"""Residual size blocks on data against two nulls: is the residual that of a strict
factor model?

For each formation month at P = 500 and P = 3,000 (all 239 months):

  data      remove m principal components from the window-demeaned 63 x P panel,
            standardize each security's residual over the window, and average the
            residual correlations within and between capitalization deciles
            (decile 1 = largest P/10 names by company capitalization at formation).
            The within-decile average excludes the diagonal.  This is the statistic of
            companion Section 7 (it reproduces companion/out heat maps).
  permute   the same pipeline after permuting each security's residual in time
            (keeps every marginal, destroys all cross-sectional dependence).
  strict FM a strict k-factor model built from the panel itself: systematic part equal
            to the panel's own rank-k PCA fit (k = skewness-rule count), specific
            returns independent Gaussian with each security's own residual variance.
            The identical pipeline is applied to 4 simulated panels per month.  Any
            block the strict model produces comes only from PCA estimation error
            (the leakage Theorem 2 quantifies), not from residual dependence.

m runs over {1,2,3,5,8,12,20,40}.  Output: figure_code/out/residual_blocks.csv with
the decile-1 block mean, the whole-matrix root-mean-square of the 10 x 10 block
matrix and the corner block (largest x smallest decile), per month, m, P and arm;
and residual_blocks_matrix_m5.npz with median 10 x 10 matrices at m = 5.
Seeds are integers derived from (year, month, P, replicate).  Runtime ~6 minutes.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, N_OBS  # noqa: E402
import rawpanel  # noqa: E402
from p01_panel_stats import jung_counts  # noqa: E402

M_GRID = (1, 2, 3, 5, 8, 12, 20, 40)
R_STRICT, R_PERM = 4, 2


def block_matrix(E, dec):
    """10 x 10 mean residual correlation between deciles (diagonal excluded)."""
    n = E.shape[0]
    Z = E - E.mean(axis=0)
    sd = np.sqrt((Z ** 2).mean(axis=0))
    Z = Z / sd
    G = np.zeros((E.shape[1], 10)); G[np.arange(E.shape[1]), dec] = 1.0
    T = Z @ G                                    # n x 10: sums of standardized residuals
    S = T.T @ T / n                              # sum_{i in g, j in h} corr_ij
    cnt = G.sum(axis=0)
    B = S / np.outer(cnt, cnt)
    d = np.diag(S) - cnt                         # remove corr_ii = 1
    B[np.diag_indices(10)] = d / (cnt * (cnt - 1))
    return B


def residuals(A, m):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return A - (U[:, :m] * s[:m]) @ Vt[:m]


def summarize(B):
    return B[0, 0], np.sqrt((B ** 2).mean()), B[0, 9]


def main():
    t0 = time.time()
    K = jung_counts()
    U = rawpanel.universes()
    W = rawpanel.wide()
    rows, mats = [], {}
    for P in (500, 3000):
        for fd, sel in U[U.n_assets == P].groupby("formation_date"):
            sel = sel.sort_values("rank")
            permnos = sel.permno.to_numpy()
            dec = np.minimum((np.arange(P) * 10) // P, 9)   # rank order = cap order
            X = W.loc[pd.DatetimeIndex(rawpanel.window_dates(fd, N_OBS)), permnos].to_numpy(float)
            A = X - X.mean(axis=0)
            k = int(K.loc[(fd, P)])
            Uu, s, Vt = np.linalg.svd(A, full_matrices=False)
            sig = (Uu[:, :k] * s[:k]) @ Vt[:k]
            resid_var = ((A - sig) ** 2).mean(axis=0)
            seed = fd.year * 10000 + fd.month * 100 + (P // 500)
            for m in M_GRID:
                E = A - (Uu[:, :m] * s[:m]) @ Vt[:m]
                B = block_matrix(E, dec)
                rows.append(dict(formation_date=fd, n_assets=P, k=k, m=m, arm="data",
                                 **dict(zip(("d1", "rms", "corner"), summarize(B)))))
                if m == 5:
                    mats.setdefault((P, "data"), []).append(B)
                for r in range(R_PERM):
                    rng = np.random.default_rng([seed, m, r, 1])
                    Ep = np.take_along_axis(E, rng.permuted(np.tile(np.arange(A.shape[0])[:, None], (1, P)), axis=0), axis=0)
                    Bp = block_matrix(Ep, dec)
                    rows.append(dict(formation_date=fd, n_assets=P, k=k, m=m, arm="permute", rep=r,
                                     **dict(zip(("d1", "rms", "corner"), summarize(Bp)))))
            for r in range(R_STRICT):
                rng = np.random.default_rng([seed, r, 2])
                Y = sig + rng.standard_normal(A.shape) * np.sqrt(resid_var)
                Y = Y - Y.mean(axis=0)
                Uy, sy, Vty = np.linalg.svd(Y, full_matrices=False)
                for m in M_GRID:
                    E = Y - (Uy[:, :m] * sy[:m]) @ Vty[:m]
                    Bs = block_matrix(E, dec)
                    rows.append(dict(formation_date=fd, n_assets=P, k=k, m=m, arm="strict_fm", rep=r,
                                     **dict(zip(("d1", "rms", "corner"), summarize(Bs)))))
                    if m == 5 and r == 0:
                        mats.setdefault((P, "strict_fm"), []).append(Bs)
        print(f"P={P} done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "residual_blocks.csv", index=False)
    np.savez(OUT / "residual_blocks_matrix_m5.npz",
             **{f"P{P}_{arm}": np.median(np.array(v), axis=0) for (P, arm), v in mats.items()})
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
