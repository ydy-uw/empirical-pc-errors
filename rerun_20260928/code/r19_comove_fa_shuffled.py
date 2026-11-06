"""The factor-analytic co-movement ratio against panels whose day variances are unrelated to the factor path.

The co-movement share of Section 6.2 (\\dvComoveShare) is the share of months in which heavy.fa's ratio
(tr(Vf' Dt Vf)/k) / (tr(Dt)/n), Dt = M diag(dd) M, Vf the top k eigenvectors of W - Dt, dd the fitted day
variances of fa_floor, is at least one (clipped P = 3,000 panels, k = the month's skewness count).  The
factor space and the day variances are estimated from the same panel, so the ratio can exceed one when the
day variances are unrelated to the factor path.  Per month:
  data    : heavy.fa's ratio on the clipped, demeaned panel (reproduces fa_c050.csv);
  shuffled: 39 panels, each with the month's day-variance path of r18 (s_t^2 from the corrected day
            variances at k = 3) permuted over the days by a fresh permutation, systematic part
            r12.planted() at the month's skewness count with that path; heavy.fa's ratio on each;
  tests   : Monte Carlo p-value (1 + #{shuffled >= data}) / 40 of the data's ratio against the 39 panels; share of
            panels with ratio >= 1; share of shuffled fits that reach fa_floor's 1,000-iteration limit.
Output (aggregates only): rerun_20260928/work/figure_code/out/checks_20261011/fa_shuffled_month.csv,
  fa_shuffled_summary.json
Run from the project root:  emp/.venv/Scripts/python.exe rerun_20260928/code/r19_comove_fa_shuffled.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "outlier_cleaning"))
import r12_null_checks as r12  # noqa: E402
import r15_leverage_check as r15  # noqa: E402
import r18_dayvar_leverage_alt as r18  # noqa: E402

OUT = r15.OUT
R = 39
FA_CSV = ROOT / "rerun_20260928" / "work" / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results" / "fa_c050.csv"


def init():
    r12._init()
    from p04_daily_volatility import fa_floor
    globals()["fa_floor"] = fa_floor


def fa_ratio(Y, k):
    """heavy.fa's co-movement ratio; Y is already clipped and demeaned."""
    N, P = Y.shape; n = N - 1
    Wd = Y @ Y.T / (n * P)
    _, _, _, _, dd, it = fa_floor(Wd, k)
    Mc = np.eye(N) - 1.0 / N
    Dt = Mc @ (dd[:, None] * Mc)
    _, Vf = np.linalg.eigh(Wd - Dt); Vf = Vf[:, -k:]
    return float((np.trace(Vf.T @ Dt @ Vf) / k) / (np.trace(Dt) / n)), int(it)


def one(args):
    mi, fd, cv0, k = args
    Y = np.clip(r12._D["panels"][fd]["X"], -r12.CLIP, r12.CLIP)
    Y = Y - Y.mean(axis=0, keepdims=True)
    N = Y.shape[0]
    _, d, _, _, cv = r18.measures(Y)
    c = np.sqrt(max(0.0, 1.0 - cv0 ** 2 / cv ** 2))
    path = np.maximum(1.0 + c * (d / d.mean() - 1.0), 0.02); path /= path.mean()
    ratio0, it0 = fa_ratio(Y, k)
    rng = np.random.default_rng([20261012, mi, 3])
    rs, its = [], []
    for _ in range(R):
        pth = path[rng.permutation(N)]
        fit, s2, _ = r12.planted(Y, k, pth, rng)
        r_, it_ = fa_ratio(r18.sim_path(fit, s2, pth, rng), k)
        rs.append(r_); its.append(it_)
    rs = np.array(rs); its = np.array(its)
    return dict(formation_date=fd, k=k, data_ratio=ratio0, data_iter=it0, shuf_mean=float(rs.mean()),
                shuf_share_ge1=float((rs >= 1).mean()), shuf_q95=float(np.quantile(rs, 0.95)),
                p=float((1 + (rs >= ratio0).sum()) / (R + 1)), shuf_nonconv=float((its >= 1000).mean()))


def main():
    lm = pd.read_csv(OUT / "lev_month.csv", parse_dates=["formation_date"]).sort_values("formation_date")
    fa = pd.read_csv(FA_CSV, parse_dates=["formation_date"])
    fa = fa[fa.n_assets == 3000].set_index("formation_date")
    t0 = time.time()
    args = [(i, pd.Timestamp(fd), cv0, int(fa.loc[fd, "k"])) for i, (fd, cv0) in
            enumerate(zip(lm.formation_date, lm.sim_mean_cv_corr))]
    with Pool(16, initializer=init) as pool:
        rows = pool.map(one, args, chunksize=2)
    m = pd.DataFrame(rows)
    m["fa_csv_ratio"] = fa.loc[m.formation_date, "comove_ratio"].values
    m.to_csv(OUT / "fa_shuffled_month.csv", index=False)
    summ = dict(months=len(m), max_abs_diff_vs_fa_csv=float((m.data_ratio - m.fa_csv_ratio).abs().max()),
                data_share_ge1=float((m.data_ratio >= 1).mean()), data_ratio_median=float(m.data_ratio.median()),
                shuf_share_ge1_median=float(m.shuf_share_ge1.median()), shuf_share_ge1_mean=float(m.shuf_share_ge1.mean()),
                shuf_ratio_median=float(m.shuf_mean.median()), data_nonconv=float((m.data_iter >= 1000).mean()),
                shuf_nonconv_overall=float(m.shuf_nonconv.mean()),
                data_gt_shuf_mean=int((m.data_ratio > m.shuf_mean).sum()),
                p_le_005=int((m.p <= 0.05).sum()), p_le_0025=int((m.p <= 0.025).sum()), p_median=float(m.p.median()))
    (OUT / "fa_shuffled_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"done in {time.time() - t0:.0f}s"); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
