"""How far is A_j from the out-of-subspace error at finite p, and what sets the size
of the gap?  A check of the effective-cross-section heuristic behind the proposed
Theorem 2' (approximate factor models).

Heuristic.  Write R for the dual Gram of the specific returns projected off the loading
span.  To first order in p^{-1/2}, A_j - OE_j = (ell - rho_j)/theta_j is a linear
functional of the fluctuation of R, with mean zero and

    sd(A_j - OE_j)  ~=  OE_j * sqrt( 2 (1 + 1/(n-k)) / p_eff ),
    p_eff = (tr Sigma_Z)^2 / tr(Sigma_Z^2),

where Sigma_Z is the cross-sectional covariance of the specific returns.  p_eff = p for
equal, independent specific variances; p/(1+c^2) when the variances have coefficient
of variation c; smaller again with cross-sectional dependence.  The theorem needs only
p_eff -> infinity (Chamberlain-Rothschild boundedness is sufficient, not necessary).
At second order the error has a negative mean (a winner's curse over the n-k bulk
directions; figure_code/theory.py, bias_second_order):

    E[A_j - OE_j] ~= -(OE_j/p_eff) [ 2(n-k) a_j + sum_i a_i + 2 sum_{l!=j} (1/a_j - 1/a_l)^-1 - 2 OE_j ],
    a_j = OE_j/(1-OE_j).

Bias over sd is about -sqrt(2) (n-k) a_j / sqrt(p_eff): at n - k = 59 and p_eff near
1,000 the bias of the weakest direction is comparable to its standard deviation.
Pr(A_j > OE_j) is then about Phi(bias/sd) (column reversal_pred).

Simulation.  k = 3 pervasive factors with signal eigenvalues and noise level taken from
cross-panel medians of the P = 3,000 panels with k >= 3 (theta_j - ell, delta^2 = n ell; factor path
whitened as in the companion's audit), N = 63 observations demeaned (n = 62), and four
specific-return designs:
  spherical     iid N(0, delta^2)
  dispersed     independent, variances lognormal with the CRSP coefficient of variation
                c = 1.49 (companion out/idio_cv.json, split-sample, P = 3,000)
  dispersed+block  as above, plus equicorrelation rho = 0.077 among the first 10% of
                names (the decile-1 residual block measured on CRSP after 5 components)
  dispersed+weak   as 'dispersed' plus one non-pervasive factor loading on 10% of the
                names, eigenvalue in Sigma_Z equal to p^{1/2} times the mean specific
                variance (so p_eff grows, but slowly)
p in {250, 500, 1000, 2000, 3000, 6000}, 400 replicates per cell, count fixed at k = 3.
Output: figure_code/out/finite_p_error.csv (gap_sd_pred: first order; gap_mean_pred:
second-order bias).  Runtime ~3 minutes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT  # noqa: E402
from theory import bias_second_order, reversal_rate  # noqa: E402

N = 63; K = 3; REPS = 400
P_GRID = (250, 500, 1000, 2000, 3000, 6000)
CV = 1.49; RHO = 0.077


def calibration():
    d = pd.read_csv(OUT / "panel_stats.csv")
    g = d[(d.n_assets == 3000) & (d.k >= 3)]
    th = np.array([g.theta1.median(), g.theta2.median(), g.theta3.median()])
    # bulk level: the median of the panels' own ell (at their selected count)
    ell = float(np.median(g.ell))
    return th - ell, ell


def sigma_design(design, p, rng, delta2):
    """Return (function drawing Z as p x N, p_eff)."""
    if design == "spherical":
        var = np.full(p, delta2)
    else:
        s2 = np.log(1 + CV ** 2)
        var = delta2 * rng.lognormal(-s2 / 2, np.sqrt(s2), p)
        var *= delta2 / var.mean()
    sd = np.sqrt(var)
    nb = p // 10
    g = np.zeros(p); g[:nb] = 1.0
    extra = 0.0
    if design == "dispersed+weak":
        lam_w = np.sqrt(p) * delta2              # eigenvalue of the weak factor in Sigma_Z
        extra = lam_w / nb                       # per-name loading^2 on the block
    # trace and Frobenius norm of Sigma_Z
    trS = var.sum() + extra * nb
    frob2 = (var ** 2).sum() + (extra * nb) ** 2 + 2 * extra * (var[:nb]).sum()
    if design == "dispersed+block":
        # corr = RHO inside the block: Sigma_ij = RHO sd_i sd_j (i != j in block)
        sb = sd[:nb]
        frob2 += RHO ** 2 * ((sb ** 2).sum() ** 2 - (sb ** 4).sum())

    def draw(r):
        Z = r.standard_normal((p, N)) * sd[:, None]
        if design == "dispersed+block":
            # equicorrelation via a common component inside the block, variance-preserving
            c = r.standard_normal(N)
            Z[:nb] = sd[:nb, None] * (np.sqrt(1 - RHO) * Z[:nb] / sd[:nb, None] + np.sqrt(RHO) * c)
        if design == "dispersed+weak":
            Z[:nb] += np.sqrt(extra) * r.standard_normal(N)
        return Z
    return draw, trS ** 2 / frob2


def main():
    lam, ell = calibration()
    n = N - 1
    delta2 = n * ell
    rows = []
    for di, design in enumerate(("spherical", "dispersed", "dispersed+block", "dispersed+weak")):
        for p in P_GRID:
            rng = np.random.default_rng([p, di, 20260925])
            draw, peff = sigma_design(design, p, rng, delta2)
            Q, _ = np.linalg.qr(rng.standard_normal((p, K)))          # true loading span
            # whitened factor path: Phi Phi^T / n = diag(lam) exactly, demeaned
            Fr = rng.standard_normal((K, N)); Fr -= Fr.mean(axis=1, keepdims=True)
            L = np.linalg.cholesky(Fr @ Fr.T / n)
            Phi = np.sqrt(lam)[:, None] * np.linalg.solve(L, Fr)
            S = np.sqrt(p) * Q @ Phi                                  # B Phi with B^T B / p = I
            gaps, oes, As = [], [], []
            for r in range(REPS):
                rr = np.random.default_rng([p, di, r, 7])
                Y = S + draw(rr)
                Y = Y - Y.mean(axis=1, keepdims=True)
                Wd = Y.T @ Y / (n * p)
                th, U = np.linalg.eigh(Wd); th = th[::-1][:n]; U = U[:, ::-1]
                ell_r = th[K:].mean()
                A = ell_r / th[:K]
                H = Y @ U[:, :K]; H /= np.linalg.norm(H, axis=0)
                OE = 1 - ((Q.T @ H) ** 2).sum(axis=0)
                gaps.append(A - OE); oes.append(OE); As.append(A)
            gaps, oes, As = map(np.array, (gaps, oes, As))
            bpred = bias_second_order(oes.mean(axis=0), peff, n)
            rpred = reversal_rate(oes.mean(axis=0), peff, n)
            for j in range(K):
                pred = oes[:, j].mean() * np.sqrt(2 * (1 + 1 / (n - K)) / peff)
                rows.append(dict(design=design, p=p, p_eff=peff, j=j + 1,
                                 OE=oes[:, j].mean(), A=As[:, j].mean(),
                                 gap_mean=gaps[:, j].mean(), gap_sd=gaps[:, j].std(ddof=1),
                                 gap_sd_pred=pred, gap_mean_pred=bpred[j], reversal_pred=rpred[j],
                                 reversal=(gaps[:, j] > 0).mean()))
            print(design, p, round(peff), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "finite_p_error.csv", index=False)


if __name__ == "__main__":
    main()
