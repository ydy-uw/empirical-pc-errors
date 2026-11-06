# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""The slower diagnostics, run on the main setting of each treatment.

blocks  residual size block: remove m principal components from the demeaned treated
        panel, standardize each name's residual, and average residual correlations
        within and between capitalization deciles (figure_code/paper/p02).  Decile 1 is
        the largest tenth of the names that remain in the panel.
fa      the factor-analytic estimate A_j^FA under day-varying specific variance
        (figure_code/paper/p04, function fa_floor), with the fit's diagnostics and the
        co-movement ratio of Proposition C.
oos     the disjoint-window check (figure_code/paper/p05): top-m directions estimated on
        two non-overlapping 63-day windows over the names common to both universes;
        realized subspace distance against the simple bound max(sum A, sum A').
        Each window is treated separately, exactly as a formation panel would be.
"""
from __future__ import annotations
import sys
import numpy as np
from oc_common import ROOT, N_OBS
sys.path.insert(0, str(ROOT / "figure_code" / "paper"))
from p02_residual_blocks import block_matrix          # noqa: E402
from p04_daily_volatility import fa_floor             # noqa: E402

M_BLOCK = (1, 3, 5, 8, 12, 20, 40)


def blocks(Xt, k):
    A = Xt - Xt.mean(axis=0)
    p = A.shape[1]
    dec = np.minimum((np.arange(p) * 10) // p, 9)          # columns are in cap-rank order
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    out = []
    for m in sorted(set(M_BLOCK) | {k}):
        E = A - (U[:, :m] * s[:m]) @ Vt[:m]
        B = block_matrix(E, dec)
        out.append(dict(m=m, at_own_k=(m == k), d1=float(B[0, 0]),
                        rms=float(np.sqrt((B ** 2).mean())), corner=float(B[0, 9])))
    return out


def fa(Xt, k):
    A_ = Xt - Xt.mean(axis=0)
    N, P = A_.shape; n = N - 1
    Wd = A_ @ A_.T / (n * P)
    A, Af, th, Uu, dd, it = fa_floor(Wd, k)
    lev = (Uu[:, :k] ** 2).sum(axis=1) + 1 / N
    Mc = np.eye(N) - 1.0 / N
    Dt = Mc @ (dd[:, None] * Mc)
    _, Vf = np.linalg.eigh(Wd - Dt); Vf = Vf[:, -k:]
    comove = float((np.trace(Vf.T @ Dt @ Vf) / k) / (np.trace(Dt) / n))
    row = dict(k=k, iterations=it, converged=bool(it < 1000),
               day_var_cv=float(dd.std() / dd.mean()), max_day_share=float(dd.max() / dd.sum()),
               max_leverage=float(lev.max()), comove_ratio=comove)
    for j in range(min(3, k)):
        row[f"A{j+1}"] = float(A[j]); row[f"Afa{j+1}"] = float(Af[j])
    return row


def window_stats(Xt, k, m):
    A_ = Xt - Xt.mean(axis=0)
    N, P = A_.shape
    Wd = A_ @ A_.T / ((N - 1) * P)
    A, Af, th, U, _, _ = fa_floor(Wd, k)
    H = A_.T @ U[:, :m]; H /= np.linalg.norm(H, axis=0)
    return A[:m], Af[:m], H


def oos_pair(X1, X2, k1, k2):
    """Realized subspace distance and the simple bounds for one pair of treated windows
    over the same names (columns aligned)."""
    m = min(k1, k2)
    A1, Af1, H1 = window_stats(X1, k1, m)
    A2, Af2, H2 = window_stats(X2, k2, m)
    C = H1.T @ H2
    return dict(m=m, realized_sub=float(m - (C ** 2).sum()),
                simple_A=float(max(A1.sum(), A2.sum())), simple_FA=float(max(Af1.sum(), Af2.sum())),
                realized_1=float(1 - C[0, 0] ** 2),
                bound_1_A=float(1 - (1 - A1[0]) * (1 - A2[0])),
                bound_1_FA=float(1 - (1 - Af1[0]) * (1 - Af2[0])))
