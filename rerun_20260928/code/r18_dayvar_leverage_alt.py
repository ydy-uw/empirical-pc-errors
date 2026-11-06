"""How much of the relation between the day variance and leverage does PCA estimation itself produce?

The leverage sum_{j<=k} U_tj^2 is estimated from the same panel as the day variance, and the leading
components lean toward days with a large specific variance.  So d_t correlates with the estimated
leverage even when the day variances are unrelated to the factor path.  This measures that effect in the
calibrated simulation and tests the data against it.  Per month (cleaned, clipped P = 3,000 panels,
63-day window, k = 3, as in Section 6.1):
  path    : the month's day-variance path, as in assumption3.run_month: s_t^2 = max(1 + c (d_t/mean d - 1),
            0.02) renormalized to mean one, c^2 = 1 - cv0^2/cv^2 (cv0: mean corrected CV of the 99
            constant-variance panels of r15, lev_month.csv; cv: the data's corrected CV);
  month   : 20 panels with s_t^2 on its own days (the relation of the data's day variances with the
            factor path kept); systematic part r12.planted() with that path;
  shuffled: 99 panels, each with s_t^2 permuted over the days by a fresh permutation (so unrelated to the
            factor path) and its own r12.planted() calibration;
  per panel: the Pearson correlation over the 63 days of d_t with the estimated leverage; the co-movement
            ratio of r17 (tr(P Dt P)/k) / (tr(Dt)/(N-1)), Dt = M diag(d) M, P = U_k U_k'; the CV of d_t;
            the correlation of the planted s_t^2 with the estimated and with the planted leverage
            (sum_{j<=k} of the planted fit's left singular vectors squared);
  tests   : Monte Carlo p-values of the data's correlation and ratio against the 99 shuffled panels;
  profile : d_t / mean(d) in the leverage bins of r15, the first shuffled panel of each month.
Output (aggregates only): rerun_20260928/work/figure_code/out/checks_20261011/
  alt_month.csv, alt_profiles.csv, alt_summary.json
Run from the project root:  emp/.venv/Scripts/python.exe rerun_20260928/code/r18_dayvar_leverage_alt.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r12_null_checks as r12  # noqa: E402
import r15_leverage_check as r15  # noqa: E402

OUT = r15.OUT
R_MONTH, R_SHUF = 20, 99
K = 3


def sim_path(fit, s2, path, rng):
    A3 = r12.A3
    eps = rng.standard_t(A3.TDF, size=fit.shape) / np.sqrt(A3.TDF / (A3.TDF - 2))
    Y = np.clip(fit + eps * np.sqrt(s2)[None, :] * np.sqrt(path)[:, None], -r12.CLIP, r12.CLIP)
    return Y - Y.mean(axis=0, keepdims=True)


def measures(Y):
    N = Y.shape[0]
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    E = Y - (U[:, :K] * s[:K]) @ Vt[:K]
    lev = (U[:, :K] ** 2).sum(axis=1)
    d = (E ** 2).mean(axis=1) / (1.0 - 1.0 / N - lev)
    M = np.eye(N) - 1.0 / N
    Dt = M @ (d[:, None] * M)
    P = U[:, :K] @ U[:, :K].T
    ratio = float((np.trace(P @ Dt @ P) / K) / (np.trace(Dt) / (N - 1)))
    return lev, d, float(np.corrcoef(d, lev)[0, 1]), ratio, r15.cv(d)


def one(args):
    mi, fd, cv0 = args
    Y = np.clip(r12._D["panels"][fd]["X"], -r12.CLIP, r12.CLIP)
    Y = Y - Y.mean(axis=0, keepdims=True)
    N = Y.shape[0]
    lev0, d, corr0, ratio0, cv = measures(Y)
    c = np.sqrt(max(0.0, 1.0 - cv0 ** 2 / cv ** 2))
    path = np.maximum(1.0 + c * (d / d.mean() - 1.0), 0.02); path /= path.mean()
    row = dict(formation_date=fd, shrink=c, data_corr=corr0, data_ratio=ratio0, data_cv=cv)
    prof = []
    for arm, R in (("month", R_MONTH), ("shuffled", R_SHUF)):
        rng = np.random.default_rng([20261012, mi, 0 if arm == "month" else 1])
        res = []
        for r in range(R):
            pth = path if arm == "month" else path[rng.permutation(N)]
            if arm == "shuffled" or r == 0:
                fit, s2, _ = r12.planted(Y, K, pth, rng)
                Uf = np.linalg.svd(fit, full_matrices=False)[0][:, :K]
                lev_true = (Uf ** 2).sum(axis=1)
            lev, ds, corr, ratio, cvs = measures(sim_path(fit, s2, pth, rng))
            res.append((corr, ratio, cvs, np.corrcoef(pth, lev)[0, 1], np.corrcoef(pth, lev_true)[0, 1]))
            if arm == "shuffled" and r == 0:
                b = np.digitize(lev, r15.BINS) - 1
                prof += [("shuffled", int(b[t]), float(lev[t]), float(ds[t] / ds.mean())) for t in range(N)]
        a = np.array(res)
        for j, q in enumerate(("corr", "ratio", "cv", "corr_path_lev", "corr_path_levtrue")):
            row[f"{arm}_{q}"] = float(a[:, j].mean())
        if arm == "shuffled":
            row["shuffled_corr_max"] = float(a[:, 0].max()); row["shuffled_ratio_max"] = float(a[:, 1].max())
            row["shuffled_ratio_share_ge1"] = float((a[:, 1] >= 1).mean())
            row["p_corr"] = float((1 + (a[:, 0] >= corr0).sum()) / (R + 1))
            row["p_ratio"] = float((1 + (a[:, 1] >= ratio0).sum()) / (R + 1))
    return row, prof


def main():
    lm = pd.read_csv(OUT / "lev_month.csv", parse_dates=["formation_date"]).sort_values("formation_date")
    t0 = time.time()
    with Pool(16, initializer=r12._init) as pool:
        res = pool.map(one, [(i, pd.Timestamp(fd), cv0) for i, (fd, cv0) in
                             enumerate(zip(lm.formation_date, lm.sim_mean_cv_corr))], chunksize=2)
    m = pd.DataFrame([r[0] for r in res]); m.to_csv(OUT / "alt_month.csv", index=False)
    pr = pd.DataFrame([p for r in res for p in r[1]], columns=["source", "bin", "lev", "d_rel"])
    pr.groupby(["source", "bin"]).agg(n=("lev", "size"), lev_median=("lev", "median"),
                                      d_rel_median=("d_rel", "median")).reset_index().to_csv(OUT / "alt_profiles.csv", index=False)
    med = lambda col: float(m[col].median())
    summ = dict(months=len(m), shrink=med("shrink"), data_corr=med("data_corr"), data_ratio=med("data_ratio"),
                **{f"{a}_{q}": med(f"{a}_{q}") for a in ("month", "shuffled")
                   for q in ("corr", "ratio", "cv", "corr_path_lev", "corr_path_levtrue")},
                shuffled_ratio_share_ge1=med("shuffled_ratio_share_ge1"),
                p_corr_le_001=int((m.p_corr <= 0.01).sum()), p_corr_le_005=int((m.p_corr <= 0.05).sum()),
                p_ratio_le_001=int((m.p_ratio <= 0.01).sum()), p_ratio_le_005=int((m.p_ratio <= 0.05).sum()),
                data_corr_gt_shuf_mean=int((m.data_corr > m.shuffled_corr).sum()),
                data_ratio_gt_shuf_mean=int((m.data_ratio > m.shuffled_ratio).sum()),
                p_corr_median=med("p_corr"), p_ratio_median=med("p_ratio"))
    (OUT / "alt_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"done in {time.time() - t0:.0f}s"); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
