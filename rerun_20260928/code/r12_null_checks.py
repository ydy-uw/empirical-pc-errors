"""Two checks on the Section 6 test and the Section 4 residual correlation (cleaned P = 3,000 panels, clipped at +-50%).

(a) Per-month Monte Carlo test of a constant day variance (Section 6.1, Figure 7a).
    For each formation month, 63-day window, k = 3 (as in Figure 7a): the statistic is the
    coefficient of variation over the 63 days of the leverage-corrected mean squared residual
    after k principal components (assumption3.py: decompose).  Under the null each stock's
    specific variance is the same on every day.  We simulate R = 99 panels from the month's own
    calibrated fit with a constant day variance (assumption3.py: the same calibration, the same
    simulate(); planted() is reproduced verbatim below), compute the statistic on each with the
    same estimator, and report the Monte Carlo p-value (1 + #{sim >= data}) / (R + 1).

(b) Can PCA residuals show a large-cap correlation block by construction?  The residuals are
    estimates, so we simulate.  For each month we simulate one panel from the
    month's rank-k fit (k = the month's skewness count) with independent specific returns and a
    constant day variance (same calibration), remove q = 2 and q = k principal components, and
    compute the mean pairwise residual correlation among the largest 10% of stocks, exactly as
    for the data (p02_residual_blocks.block_matrix, decile 1 diagonal).  The data values are
    computed the same way on the same panels.

Output (aggregates only): rerun_20260928/work/figure_code/out/checks_20261006/{a_cv_test.csv, b_block_null.csv}
Run from the project root:  emp/.venv/Scripts/python.exe rerun_20260928/code/r12_null_checks.py
"""
from __future__ import annotations
import os, sys, pickle, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "rerun_20260928" / "work"
sys.path.insert(0, str(ROOT / "rerun_20260928" / "code" / "figure_code" / "outline_20260927"))
sys.path.insert(0, str(ROOT / "figure_code" / "paper"))
sys.path.insert(0, str(ROOT / "companion" / "code"))
OUT = WORK / "figure_code" / "out" / "checks_20261006"
CACHE = WORK / "outlier_cleaning" / "_cache" / "panels_P3000.pkl"
COUNTS = WORK / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results" / "panels_c050.csv"
CLIP, R = 0.5, 99
_D = {}


def _init():
    global A3, block_matrix
    import assumption3 as A3  # noqa: F401  (decompose, estimates, simulate)
    from p02_residual_blocks import block_matrix  # noqa: F401
    globals()["A3"], globals()["block_matrix"] = A3, block_matrix
    _D["panels"] = pickle.load(open(CACHE, "rb"))


def planted(Y, k, path, rng):
    """assumption3.run_month's planted(), verbatim (calibrated rank-k signal and stock variances)."""
    theta, Uf, s, Vt, dvar, sig2 = A3.decompose(Y, k)
    n = Y.shape[0] - 1
    ell = theta[k:].mean(); Vk = Vt[:k].T
    lam = np.maximum(theta[:k] - ell * (Uf[:, :k] ** 2 * path[:, None]).sum(axis=0), 1e-12)
    s2 = sig2.copy(); m = min(k, 5)
    for _ in range(2):
        fit = (Uf[:, :k] * np.sqrt(n * Y.shape[1] * lam)) @ Vt[:k]
        pilot = A3.simulate(fit, Vk, s2, path, k, rng, False)
        s2 = s2 * ell / pilot["ell"]
        th_sim = np.array([pilot[f"theta{j+1}"] for j in range(m)])
        lam[:m] = np.maximum(lam[:m] - (th_sim - theta[:m]), 1e-3 * lam[:m])
    return (Uf[:, :k] * np.sqrt(n * Y.shape[1] * lam)) @ Vt[:k], s2, Vk


def sim_panel(fit, s2, rng, N):
    eps = rng.standard_t(A3.TDF, size=fit.shape) / np.sqrt(A3.TDF / (A3.TDF - 2))
    Y = np.clip(fit + eps * np.sqrt(s2)[None, :], -CLIP, CLIP)
    return Y - Y.mean(axis=0, keepdims=True)


def top_decile_corr(Y, q):
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    E = Y - (U[:, :q] * s[:q]) @ Vt[:q]
    p = Y.shape[1]
    dec = np.minimum((np.arange(p) * 10) // p, 9)          # columns in cap-rank order
    return float(block_matrix(E, dec)[0, 0])


def one_month(args):
    mi, fd, kskew = args
    X = _D["panels"][fd]["X"]
    Y = np.clip(X, -CLIP, CLIP); Y = Y - Y.mean(axis=0, keepdims=True)
    N = Y.shape[0]; ones = np.ones(N)
    # (a)
    rng = np.random.default_rng([20261006, mi, 1])
    cv_data = A3.estimates(Y, 3, False)["cv_dayvar"]
    fit, s2, Vk = planted(Y, 3, ones, rng)
    cvs = np.array([A3.simulate(fit, Vk, s2, ones, 3, rng, False)["cv_dayvar"] for _ in range(R)])
    a = dict(formation_date=fd, cv_data=cv_data, cv_sim_mean=float(cvs.mean()), cv_sim_sd=float(cvs.std(ddof=1)),
             cv_sim_max=float(cvs.max()), p_mc=float((1 + (cvs >= cv_data).sum()) / (R + 1)))
    # (b)
    rng2 = np.random.default_rng([20261006, mi, 2])
    k = int(kskew)
    fitk, s2k, _ = planted(Y, k, ones, rng2)
    Ys = sim_panel(fitk, s2k, rng2, N)
    b = dict(formation_date=fd, k=k, data_q2=top_decile_corr(Y, 2), data_qk=top_decile_corr(Y, k),
             sim_q2=top_decile_corr(Ys, 2), sim_qk=top_decile_corr(Ys, k))
    return a, b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(COUNTS, parse_dates=["formation_date"])
    ks = c[c.n_assets == 3000].set_index("formation_date").k_skew
    months = sorted(ks.index)
    t0 = time.time()
    with Pool(16, initializer=_init) as pool:
        res = pool.map(one_month, [(i, pd.Timestamp(fd), ks.loc[fd]) for i, fd in enumerate(months)], chunksize=4)
    a = pd.DataFrame([r[0] for r in res]); b = pd.DataFrame([r[1] for r in res])
    a.to_csv(OUT / "a_cv_test.csv", index=False); b.to_csv(OUT / "b_block_null.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
    print("(a) months:", len(a), " median data CV %.3f, median simulated mean CV %.3f, median sim sd %.4f"
          % (a.cv_data.median(), a.cv_sim_mean.median(), a.cv_sim_sd.median()))
    print("    share of months with p_mc = 0.01 (data above all 99):", round((a.p_mc <= 0.01).mean(), 3),
          "; p_mc <= 0.05:", round((a.p_mc <= 0.05).mean(), 3), "; min ratio data/sim max:", round((a.cv_data / a.cv_sim_max).min(), 2))
    print("(b) median top-decile corr (%):  data q=2 %.2f  q=k %.2f | sim q=2 %.2f  q=k %.2f"
          % tuple(100 * b[c_].median() for c_ in ("data_q2", "data_qk", "sim_q2", "sim_qk")))
    print("    sim q=k 90th pct %.2f, max %.2f; data q=k min %.2f" % (100 * b.sim_qk.quantile(.9), 100 * b.sim_qk.max(), 100 * b.data_qk.min()))


if __name__ == "__main__":
    main()
