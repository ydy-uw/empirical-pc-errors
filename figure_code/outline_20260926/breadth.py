"""Breadth experiment for the 26 September outline (Section 4).

Inside each clipped 3,000-stock formation panel, draw nested random subsets of
p = 375, 750, 1,500 and 3,000 stocks and compute the centred dual spectrum
theta_1..theta_62.  Random subsets hold the composition of the universe fixed, so any
change with p comes from breadth alone.

For each subset we record
  e_j = theta_j / ell_20 - 1   (j = 1..15), the excess of eigenvalue j over the bulk,
        with ell_20 the mean of theta_21..theta_62 (a level that does not depend on k);
  A_j(k) = mean(theta_{k+1..62}) / theta_j for k = 2, 3, 5 and j = 1..3, the estimate
        of Theorem 2 at a fixed count.

Benchmarks (computed on the same subsets), built from the month's own data.  The
clipped panel is split into its rank-2 PCA fit and the residual E, and one of two
things is done to E before adding it back to the fit:
  twofactor       each stock's residual series is shuffled in time independently: this
                  keeps every stock's residual distribution and removes both the
                  cross-sectional dependence of the residuals and any common day-to-day
                  change in residual dispersion;
  twofactor_flip  every residual is multiplied by an independent random sign: this keeps
                  the size of every residual, hence each day's residual dispersion, and
                  removes the cross-sectional dependence of the residuals.
  twofactor_rotate each stock's residual series is rotated in time by an independent
                  random circular shift: this keeps every stock's own serial dependence
                  (apart from one wrap-around point) and removes the cross-sectional
                  dependence of the residuals and any common day-to-day change in
                  residual dispersion.
Data minus twofactor_flip measures dependence in the residuals beyond two factors, across
stocks or over time; twofactor_rotate minus twofactor isolates the part due to serial
dependence within each stock; twofactor_flip minus twofactor measures the effect of
day-to-day changes in residual dispersion (a departure from Assumption 3).  (A benchmark
that flips the sign of each stock's whole residual series is not informative: it leaves
the spectrum of the residual part unchanged.)

Also recorded per month, on the full 3,000-stock residual after two factors: the pooled
lag-1 autocorrelation of the residuals (sum of e_t e_{t-1} over sum of e_t^2), the median
of the stock-level lag-1 autocorrelations, and, for the largest 10% of stocks by
capitalization, the mean pairwise residual correlation and the ratio of their mean
residual variance to the mean over all stocks.  Output: breadth_resid.csv.  Random subsets of one
universe show whether structure is present in that universe; they cannot show how it
would grow along a nested sequence of stocks, which is what Assumptions 2' and 4 are about.

Inputs: the licensed CRSP extract through companion/code/rawpanel.py (read-only).
Output (aggregates only, no identifiers): figure_code/out/outline_20260926/
    breadth_panels.csv (one row per month x draw x p x kind) and breadth.csv (medians).
Run from the project root:  OPENBLAS_NUM_THREADS=1 python figure_code/outline_20260926/breadth.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "companion" / "code"))
import rawpanel

OUT = ROOT / "figure_code" / "out" / "outline_20260926"
P_FULL, N_OBS = 3000, 63
SIZES = (375, 750, 1500, 3000)
DRAWS = 8
JMAX = 15
KS = (2, 3, 5)
CLIP = 0.5
K_NULL = 2


def dual(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    return (s ** 2 / (X.shape[0] * X.shape[1]))[: N_OBS - 1]


def stats(th):
    d = {f"e{j+1}": th[j] / th[20:].mean() - 1.0 for j in range(JMAX)}
    for k in KS:
        ell = th[k:].mean()
        for j in range(3):
            d[f"A{j+1}_k{k}"] = ell / th[j]
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    U = rawpanel.universes(); W = rawpanel.wide()
    rows, resid = [], []
    t0 = time.time()
    months = sorted(U[U.n_assets == P_FULL].formation_date.unique())
    for mi, fd in enumerate(months):
        sel = U[(U.n_assets == P_FULL) & (U.formation_date == fd)].sort_values("rank")
        dts = rawpanel.window_dates(pd.Timestamp(fd), N_OBS)
        X = W.loc[pd.DatetimeIndex(dts), sel.permno.to_numpy()].to_numpy(float)
        Y = np.clip(X, -CLIP, CLIP)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        Uf, s, Vt = np.linalg.svd(Yc, full_matrices=False)
        fit = (Uf[:, :K_NULL] * s[:K_NULL]) @ Vt[:K_NULL]
        E = Yc - fit
        top = E[:, : P_FULL // 10]
        Z = (top - top.mean(0)) / top.std(0)
        C = Z.T @ Z / N_OBS
        m = C.shape[0]
        v = (E ** 2).mean(axis=0)
        num = (E[1:] * E[:-1]).sum(axis=0); den = (E ** 2).sum(axis=0)
        resid.append(dict(formation_date=pd.Timestamp(fd),
                          rho1_pooled=num.sum() / den.sum(),
                          rho1_median=float(np.median(num / den)),
                          top_corr=(C.sum() - np.trace(C)) / (m * (m - 1)),
                          top_var_ratio=v[: P_FULL // 10].mean() / v.mean()))
        rng = np.random.default_rng(20260926 + mi)
        rng2 = np.random.default_rng(20270926 + mi)
        tt = np.arange(N_OBS)[:, None]
        for d in range(DRAWS):
            Es = np.empty_like(E)
            for i in range(P_FULL):
                Es[:, i] = E[rng.permutation(N_OBS), i]
            Ynull = fit + Es
            Yflip = fit + E * rng.choice((-1.0, 1.0), size=E.shape)
            shifts = rng2.integers(0, N_OBS, size=P_FULL)
            Yrot = fit + np.take_along_axis(E, (tt - shifts[None, :]) % N_OBS, axis=0)
            perm = rng.permutation(P_FULL)
            for p in SIZES:
                cols = perm[:p]
                for kind, M in (("data", Yc), ("twofactor", Ynull), ("twofactor_flip", Yflip),
                                ("twofactor_rotate", Yrot)):
                    rows.append(dict(formation_date=pd.Timestamp(fd), draw=d, p=p, kind=kind,
                                     **stats(dual(M[:, cols]))))
        if mi % 20 == 0:
            print(f"{mi+1}/{len(months)} months, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(resid).to_csv(OUT / "breadth_resid.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "breadth_panels.csv", index=False)
    # median over draws within a month, then median and quartiles over months
    g = df.groupby(["kind", "p", "formation_date"]).median(numeric_only=True).drop(columns="draw")
    summ = []
    for (kind, p), h in g.groupby(level=[0, 1]):
        for c in h.columns:
            v = h[c]
            summ.append(dict(kind=kind, p=p, stat=c, median=v.median(),
                             q25=v.quantile(0.25), q75=v.quantile(0.75)))
    pd.DataFrame(summ).to_csv(OUT / "breadth.csv", index=False)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
