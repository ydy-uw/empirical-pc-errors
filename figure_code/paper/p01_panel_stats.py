"""Per-panel statistics for the paper figures (pervasiveness, outliers, floors).

For every formation panel (239 month ends x P in {500,1000,2000,3000}) this rebuilds
the 63-day return panel from the frozen CRSP extract and the frozen universe, exactly
as companion/code/rawpanel.py does, and records

  * the centered dual spectrum theta_1..theta_62 (W = Yc Yc^T/(N P)),
  * the skewness-rule count k (Jung et al. 2018, alpha = 0.10, stored by the pipeline),
  * A_j = ell_k/theta_j for j = 1,2,3, ell_k the mean of theta_{k+1..N-1},
  * the subspace number S = sum_{j<=k} A_j and S/k,
  * for the sample directions h_1..h_3: participation 1/(P sum_i h_ij^4) and the peak
    share max_i |h_ij| (h_j unit length),
  * the number of daily returns above +100% inside the window and the largest one,
  * the same floors and screens on the panel clipped to [-100%, +100%], with the count
    held at the raw panel's k (so each raw/clipped pair differs only by the returns),
  * the plug-in second-order bias of A_j (figure_code/theory.py, Proposition B of the
    outline) with p_eff = P/(1 + c^2), c the split-sample coefficient of variation of
    residual variances (companion/out/idio_cv.json, computed on raw panels), and the
    relative eigengaps (theta_j - theta_{j+1})/theta_j for j = 1, 2 (near-degeneracy).

Output: figure_code/out/panel_stats.csv (one row per panel; aggregates only, no
security identifiers).  Runtime: about 2 minutes.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, P_GRID, N_OBS, COMP_OUT  # noqa: E402
from theory import bias_second_order  # noqa: E402
import paths, rawpanel  # noqa: E402  (companion/code)


def jung_counts(alpha=0.10):
    fs = paths.year_files(paths.ROLLING, "factor_selections.parquet")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[np.isclose(d.alpha, alpha)]
    d["formation_date"] = pd.to_datetime(d.formation_date)
    return d.set_index(["formation_date", "n_assets"]).selected_factors


def dual_stats(A, k):
    """Centered dual spectrum and the first three right singular vectors."""
    n, p = A.shape
    Ac = A - A.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    theta = s ** 2 / (n * p)
    theta = theta[: n - 1]                         # n-1 non-degenerate directions
    ell = theta[k:].mean()
    H = Vt[:3].T                                   # p x 3, unit columns
    part = 1.0 / (p * (H ** 4).sum(axis=0))
    peak = np.abs(H).max(axis=0)
    return theta, ell, part, peak


def main():
    t0 = time.time()
    K = jung_counts()
    U = rawpanel.universes()
    W = rawpanel.wide()
    cv = json.load(open(COMP_OUT / "idio_cv.json"))
    rows = []
    for (fd, P), sel in U.groupby(["formation_date", "n_assets"]):
        permnos = sel.sort_values("rank").permno.to_numpy()
        dts = rawpanel.window_dates(fd, N_OBS)
        X = W.loc[pd.DatetimeIndex(dts), permnos].to_numpy(float)
        if np.isnan(X).any():
            raise RuntimeError(f"missing returns in {fd:%Y-%m} P={P}")
        k = int(K.loc[(fd, P)])
        th, ell, part, peak = dual_stats(X, k)
        Xc = np.clip(X, -1.0, 1.0)
        thc, ellc, partc, peakc = dual_stats(Xc, k)
        big = X > 1.0
        row = dict(formation_date=fd, n_assets=P, k=k, ell=ell, ell_clip=ellc,
                   n_ext=int(big.sum()), n_ext_names=int(big.any(axis=0).sum()),
                   max_ret=float(X.max()),
                   S=float((ell / th[:k]).sum()), S_clip=float((ellc / thc[:k]).sum()),
                   theta_bulk_sd_over_mean=float(th[k:].std() / th[k:].mean()))
        peff = P / (1 + cv[str(P)]["split"] ** 2)
        b_raw = bias_second_order(ell / th[:k], peff, N_OBS - 1)
        b_clip = bias_second_order(ellc / thc[:k], peff, N_OBS - 1)
        row["p_eff"] = peff
        for j in range(2):
            row[f"relgap{j+1}{j+2}"] = (th[j] - th[j + 1]) / th[j]
            row[f"relgap{j+1}{j+2}_clip"] = (thc[j] - thc[j + 1]) / thc[j]
        for j in range(3):
            row[f"bias{j+1}"] = b_raw[j] if j < k else np.nan
            row[f"bias{j+1}_clip"] = b_clip[j] if j < k else np.nan
            row[f"theta{j+1}"] = th[j]
            row[f"theta{j+1}_clip"] = thc[j]
            row[f"A{j+1}"] = ell / th[j] if j < k else np.nan
            row[f"A{j+1}_clip"] = ellc / thc[j] if j < k else np.nan
            row[f"part{j+1}"] = part[j]; row[f"peak{j+1}"] = peak[j]
            row[f"part{j+1}_clip"] = partc[j]; row[f"peak{j+1}_clip"] = peakc[j]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["n_assets", "formation_date"])
    for j in (1, 2, 3):
        for suf in ("", "_clip"):
            df[f"loc{j}{suf}"] = (df[f"part{j}{suf}"] < 0.01) | (df[f"peak{j}{suf}"] > 0.5)
    df.to_csv(OUT / "panel_stats.csv", index=False)
    print(f"{len(df)} panels in {time.time()-t0:.0f}s -> {OUT/'panel_stats.csv'}")


if __name__ == "__main__":
    main()
