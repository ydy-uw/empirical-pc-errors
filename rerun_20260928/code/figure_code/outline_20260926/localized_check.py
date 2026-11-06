# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""What the localized sample directions load on (Section 3 of the 26 September outline).

For every untreated panel whose sample direction j (j = 1, 2, 3, within the skewness
count) fails the pervasiveness screen, find the stock with the largest absolute entry
of h_j and record the largest absolute daily return of that stock in the window, and
whether that return is the largest in the whole panel; the same for the directions
that are still localized after clipping at +-50% (with the stock's untreated return
and whether it reaches the clip).  Also records, for the clipped
panels at P = 3,000 with k = 5 factors removed, the ratio of the mean residual variance
of the largest 10% of stocks to the mean over all stocks (used in Section 4).

Inputs: the licensed CRSP extract (read-only) and the per-panel screen results in
outlier_cleaning/experiments/00_raw/results/panels_main.csv.
Output (aggregates only): figure_code/out/outline_20260926/localized_check.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[4]
import os  # noqa: E402
WORK = Path(os.environ.get("HLHH_RERUN_OUT", ROOT / "rerun_20260928" / "work"))
sys.path.insert(0, str(ROOT / "companion" / "code"))
import rawpanel

OUT = WORK / "figure_code" / "out" / "outline_20260926"
N_OBS = 63


def main():
    U = rawpanel.universes(); W = rawpanel.wide()
    raw = pd.read_csv(WORK / "outlier_cleaning/experiments/00_raw/results/panels_main.csv",
                      parse_dates=["formation_date"])
    res = {}
    for P in (500, 3000):
        rec = []
        g = raw[raw.p == P]
        for _, r in g.iterrows():
            js = [j for j in (1, 2, 3) if r.k_skew >= j and bool(r[f"loc{j}"])]
            if not js:
                continue
            sel = U[(U.n_assets == P) & (U.formation_date == r.formation_date)].sort_values("rank")
            X = W.loc[pd.DatetimeIndex(rawpanel.window_dates(r.formation_date, N_OBS)),
                      sel.permno.to_numpy()].to_numpy(float)
            Xc = X - X.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            colmax = np.abs(X).max(axis=0)
            for j in js:
                i = int(np.argmax(np.abs(Vt[j - 1])))
                rec.append(dict(j=j, peak=float(np.abs(Vt[j - 1, i])), top_ret=float(colmax[i]),
                                is_panel_max=bool(colmax[i] >= colmax.max() - 1e-12),
                                rank_in_panel=int((colmax > colmax[i]).sum()) + 1))
        d = pd.DataFrame(rec)
        res[str(P)] = {str(j): dict(months=int((d.j == j).sum()),
                                    top_stock_ret_gt_100=float((d[d.j == j].top_ret > 1.0).mean()),
                                    top_stock_ret_gt_50=float((d[d.j == j].top_ret > 0.5).mean()),
                                    top_stock_is_panel_max=float(d[d.j == j].is_panel_max.mean()),
                                    top_stock_rank_median=float(d[d.j == j].rank_in_panel.median()),
                                    top_stock_ret_median=float(d[d.j == j].top_ret.median()))
                       for j in (1, 2, 3) if (d.j == j).any()}
    # the same for the clipped panels (setting c050): localized directions that survive the clip
    cln = pd.read_csv(WORK / "outlier_cleaning/experiments/01_fixed_clip/results/panels_c050.csv",
                      parse_dates=["formation_date"])
    resc = {}
    for P in (500, 3000):
        rec = []
        g = cln[cln.p == P]
        for _, r in g.iterrows():
            js = [j for j in (1, 2, 3) if r.k_ref >= j and bool(r[f"loc{j}"])]
            if not js:
                continue
            sel = U[(U.n_assets == P) & (U.formation_date == r.formation_date)].sort_values("rank")
            X = W.loc[pd.DatetimeIndex(rawpanel.window_dates(r.formation_date, N_OBS)),
                      sel.permno.to_numpy()].to_numpy(float)
            Xk = np.clip(X, -0.5, 0.5)
            Xc = Xk - Xk.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            cmax_raw = np.abs(X).max(axis=0); cmax_clip = np.abs(Xk).max(axis=0)
            for j in js:
                i = int(np.argmax(np.abs(Vt[j - 1])))
                rec.append(dict(j=j, at_clip=bool(cmax_raw[i] >= 0.5), raw_ret=float(cmax_raw[i]),
                                is_panel_max_clipped=bool(cmax_clip[i] >= cmax_clip.max() - 1e-12),
                                rank_clipped=int((cmax_clip > cmax_clip[i]).sum()) + 1))
        d = pd.DataFrame(rec)
        resc[str(P)] = {str(j): dict(months=int((d.j == j).sum()),
                                     top_stock_hits_clip=float(d[d.j == j].at_clip.mean()),
                                     top_stock_raw_ret_median=float(d[d.j == j].raw_ret.median()),
                                     top_stock_raw_ret_q25=float(d[d.j == j].raw_ret.quantile(0.25)),
                                     top_stock_is_panel_max=float(d[d.j == j].is_panel_max_clipped.mean()))
                        for j in (1, 2, 3) if len(d) and (d.j == j).any()}
    res["clipped"] = resc
    # residual variance of the largest decile relative to the mean, clipped, P = 3,000, k = 5
    ratios = []
    months = sorted(U[U.n_assets == 3000].formation_date.unique())
    for fd in months:
        sel = U[(U.n_assets == 3000) & (U.formation_date == fd)].sort_values("rank")
        X = np.clip(W.loc[pd.DatetimeIndex(rawpanel.window_dates(pd.Timestamp(fd), N_OBS)),
                          sel.permno.to_numpy()].to_numpy(float), -0.5, 0.5)
        Xc = X - X.mean(axis=0, keepdims=True)
        Uf, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        E = Xc - (Uf[:, :5] * s[:5]) @ Vt[:5]
        v = (E ** 2).mean(axis=0)
        ratios.append(v[:300].mean() / v.mean())
    res["top_decile_resid_var_ratio"] = dict(median=float(np.median(ratios)),
                                             q25=float(np.quantile(ratios, 0.25)),
                                             q75=float(np.quantile(ratios, 0.75)), months=len(ratios))
    (OUT / "localized_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
