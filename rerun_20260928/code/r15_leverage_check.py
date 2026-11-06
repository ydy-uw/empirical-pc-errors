"""Leverage of PCA residuals and the day-variance test of Section 6.1.

The residuals after k components are those of each stock's regression on the k estimated factor
paths: for the demeaned N x P panel, E = (I - U_k U_k^T) Y with U_k the N x k leading dual
eigenvectors.  Day t's residuals are deflated by its leverage h_t = 1/N + sum_{j<=k} U_tj^2 (the
1/N from demeaning): with constant specific variances, mean_i E_ti^2 -> delta^2 (1 - h_t) as P grows
(conditional on the factor path).  Days with large factor moves have high leverage, so the raw
cross-sectional mean of squared residuals g_t varies over the window under a constant specific
variance.  The day variance is d_t = g_t / (1 - h_t).

Per month (cleaned, clipped P = 3,000 panels, 63-day window, k = 3, as in Section 6.1):
  data   : g_t, d_t, leverage h_t; CV of g and of d; corr(g_t, 1 - h_t); corr(d_t, h_t);
           largest leverage sum_j U_tj^2 and g on that day relative to the mean of g;
  null   : R = 99 panels from the month's calibrated constant-variance fit (r12_null_checks:
           planted(), sim_panel(); t_5 noise, independent stocks), the same quantities on each;
           Monte Carlo p-values for the raw and the corrected statistic;
  permutation null on leverage-standardized residuals: each stock's E_ti / sqrt(1 - h_t) permuted
           over the days independently of the other stocks; the statistic CV over days of the mean
           of the squared permuted values, which is the statistic d_t under the permutation;
           99 permutations; k = 3 and k = the month's skewness count.
  profiles: g_t / mean(g) and d_t / mean(d) against h_t, pooled over months in leverage bins,
           for the data and for one simulated panel per month.
Output (aggregates only): rerun_20260928/work/figure_code/out/checks_20261011/
  lev_month.csv, lev_profiles.csv, summary.json
Run from the project root:  emp/.venv/Scripts/python.exe rerun_20260928/code/r15_leverage_check.py
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r12_null_checks as r12  # noqa: E402  (planted, sim_panel, CACHE, COUNTS, CLIP)

OUT = ROOT / "rerun_20260928" / "work" / "figure_code" / "out" / "checks_20261011"
R = 99
K = 3
BINS = np.array([0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.30, 1.0])


def init():
    r12._init()


def decomp(Y, k):
    N = Y.shape[0]
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    E = Y - (U[:, :k] * s[:k]) @ Vt[:k]
    h = 1.0 / N + (U[:, :k] ** 2).sum(axis=1)
    g = (E ** 2).mean(axis=1)
    return E, h, g, g / (1.0 - h)


def cv(x):
    return float(x.std() / x.mean())


def stats(Y, k):
    E, h, g, d = decomp(Y, k)
    lev = h - 1.0 / Y.shape[0]
    t = int(np.argmax(lev))
    return dict(cv_raw=cv(g), cv_corr=cv(d), corr_g_1mh=float(np.corrcoef(g, 1 - h)[0, 1]),
                corr_d_h=float(np.corrcoef(d, h)[0, 1]), lev_max=float(lev.max()),
                g_at_max_rel=float(g[t] / g.mean()), d_at_max_rel=float(d[t] / d.mean())), (E, h, g, d)


def perm_null(E, h, rng):
    S = E / np.sqrt(1.0 - h)[:, None]
    S2 = S ** 2
    out = np.empty(R)
    for r in range(R):
        idx = np.argsort(rng.random(S2.shape), axis=0)
        m = np.take_along_axis(S2, idx, axis=0).mean(axis=1)
        out[r] = m.std() / m.mean()
    return out


def one(args):
    mi, fd, kskew = args
    Y = np.clip(r12._D["panels"][fd]["X"], -r12.CLIP, r12.CLIP)
    Y = Y - Y.mean(axis=0, keepdims=True)
    N = Y.shape[0]
    row = dict(formation_date=fd, k_skew=int(kskew))
    s_data, (E, h, g, d) = stats(Y, K)
    row.update({f"data_{a}": b for a, b in s_data.items()})
    # constant-variance null, same calibration as Figure 7a / r12 (a)
    rng = np.random.default_rng([20261011, mi, 1])
    fit, s2, _ = r12.planted(Y, K, np.ones(N), rng)
    sims = []
    prof_sim = None
    for r in range(R):
        Ys = r12.sim_panel(fit, s2, rng, N)
        st, parts = stats(Ys, K)
        sims.append(st)
        if r == 0:
            prof_sim = parts
    sims = pd.DataFrame(sims)
    for c in ("cv_raw", "cv_corr", "corr_g_1mh", "corr_d_h", "lev_max", "g_at_max_rel", "d_at_max_rel"):
        row[f"sim_mean_{c}"] = float(sims[c].mean())
    row["sim_max_cv_raw"] = float(sims.cv_raw.max()); row["sim_max_cv_corr"] = float(sims.cv_corr.max())
    row["p_raw"] = float((1 + (sims.cv_raw >= row["data_cv_raw"]).sum()) / (R + 1))
    row["p_corr"] = float((1 + (sims.cv_corr >= row["data_cv_corr"]).sum()) / (R + 1))
    # permutation null on leverage-standardized residuals
    for tag, k in (("k3", K), ("kown", int(kskew))):
        rngp = np.random.default_rng([20261011, mi, 2, k])
        Ek, hk, gk, dk = decomp(Y, k)
        nul = perm_null(Ek, hk, rngp)
        row[f"perm_{tag}_cv_data"] = cv(dk); row[f"perm_{tag}_mean"] = float(nul.mean())
        row[f"perm_{tag}_max"] = float(nul.max())
        row[f"perm_{tag}_p"] = float((1 + (nul >= cv(dk)).sum()) / (R + 1))
    # profiles for the figure
    prof = []
    for src, (Ee, hh, gg, dd) in (("data", (E, h, g, d)), ("sim", prof_sim)):
        b = np.digitize(hh - 1.0 / N, BINS) - 1
        for t in range(N):
            prof.append((src, int(b[t]), float(hh[t] - 1.0 / N), float(gg[t] / gg.mean()), float(dd[t] / dd.mean())))
    return row, prof


def main():
    c = pd.read_csv(r12.COUNTS, parse_dates=["formation_date"])
    ks = c[c.n_assets == 3000].set_index("formation_date").k_skew
    t0 = time.time()
    with Pool(16, initializer=init) as pool:
        res = pool.map(one, [(i, pd.Timestamp(fd), ks.loc[fd]) for i, fd in enumerate(sorted(ks.index))], chunksize=4)
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame([r[0] for r in res]); m.to_csv(OUT / "lev_month.csv", index=False)
    pr = pd.DataFrame([p for r in res for p in r[1]], columns=["source", "bin", "lev", "g_rel", "d_rel"])
    prof = pr.groupby(["source", "bin"]).agg(n=("lev", "size"), lev_median=("lev", "median"),
                                             g_rel_median=("g_rel", "median"), d_rel_median=("d_rel", "median")).reset_index()
    prof.to_csv(OUT / "lev_profiles.csv", index=False)
    med = lambda col: float(m[col].median())
    summ = dict(
        months=len(m),
        data=dict(cv_raw=med("data_cv_raw"), cv_corr=med("data_cv_corr"), corr_g_1mh=med("data_corr_g_1mh"),
                  corr_d_h=med("data_corr_d_h"), share_corr_d_h_pos=float((m.data_corr_d_h > 0).mean()),
                  lev_max=med("data_lev_max"), g_at_max_rel=med("data_g_at_max_rel"), d_at_max_rel=med("data_d_at_max_rel")),
        null=dict(cv_raw=med("sim_mean_cv_raw"), cv_corr=med("sim_mean_cv_corr"), corr_g_1mh=med("sim_mean_corr_g_1mh"),
                  corr_d_h=med("sim_mean_corr_d_h"), lev_max=med("sim_mean_lev_max"),
                  g_at_max_rel=med("sim_mean_g_at_max_rel"), d_at_max_rel=med("sim_mean_d_at_max_rel")),
        test=dict(raw_p001=int((m.p_raw <= 0.01).sum()), corr_p001=int((m.p_corr <= 0.01).sum()),
                  min_ratio_raw=float((m.data_cv_raw / m.sim_max_cv_raw).min()),
                  min_ratio_corr=float((m.data_cv_corr / m.sim_max_cv_corr).min())),
        perm=dict(k3_mean=med("perm_k3_mean"), k3_p001=int((m.perm_k3_p <= 0.01).sum()),
                  k3_min_ratio=float((m.perm_k3_cv_data / m.perm_k3_max).min()),
                  kown_mean=med("perm_kown_mean"), kown_p001=int((m.perm_kown_p <= 0.01).sum()),
                  kown_p_max=float(m.perm_kown_p.max()),
                  kown_min_ratio=float((m.perm_kown_cv_data / m.perm_kown_max).min())))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"done in {time.time() - t0:.0f}s"); print(json.dumps(summ, indent=1))
    print(prof.to_string())


if __name__ == "__main__":
    main()
