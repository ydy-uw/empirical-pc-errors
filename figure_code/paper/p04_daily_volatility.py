"""Day-to-day specific variance (a failure of Assumption 3) and the out-of-subspace estimate.

Mechanism.  If day t's specific returns have cross-sectional variance delta_t^2, the dual
matrix of the window-demeaned panel converges (p -> infinity, n, k fixed) to

    W_inf = S + M D M,   S = systematic dual (rank k),  D = diag(delta_t^2)/n,
    M = I - 11'/N (demeaning).

The identity A_j - OE_j = (ell - rho_j)/theta_j holds at every p, and in the limit
rho_j -> u_j' D u_j (specific energy on the days direction j uses) while
ell -> [tr(P_perp S) + tr(P_perp M D M)]/(n - k).  They differ unless D is a multiple of
the identity, so A_j is then an inconsistent estimate of OE_j.  (If factor returns are
large on high-variance days the aggregate bias is provably downward; see the outline,
Proposition C.)

Estimator studied here (factor-analytic dual, "A^FA").  Fit W ~ L L' + M diag(d) M with
L of rank k and d >= 0 by principal-axis iteration: given d, L L' is the rank-k part of
W - M diag(d) M; given L, d solves the least-squares problem in vec form (non-negative).
Then A_j^FA = u_j' diag(d) u_j / theta_j, with (theta_j, u_j) the eigenpairs of W.
In the limit this recovers OE_j exactly whenever the decomposition is identified.  (A
one-step version that fits d to W - sum_j theta_j u_j u_j' is NOT consistent: that
residual also contains P_perp S P_perp.  It is not used.)

Part 1 (simulation): k = 3 factors with signal eigenvalues and noise level from the
cross-panel medians of the P = 3,000 panels with k >= 3 (as p03).  Four arms:
  constant        day variances all equal (Assumption 3 holds)
  dayvar          day variances lognormal, coefficient of variation 0.6
  dayvar+comove   ... and factor returns scaled by the same day volatility
  realistic       name-level variance dispersion c = 1.49, standardized t5 noise,
                  day-variance CV 0.5, co-movement
For each arm, 8 draws of (day variances, factor path) are held FIXED across
p in {500, 3000, 20000} (so the p-direction is a genuine sweep), 40 replicates each,
plus the exact p = infinity value computed from W_inf.
Part 2 (data, --data): A_j and A_j^FA on all 956 CRSP panels, raw and clipped, with the
fit's diagnostics and the co-movement ratio (b/k)/(T/n) of Proposition C computed from
the fitted day variances (>= 1: the factor path's days carry at least the average
specific variance, so A_j understates in theta-weighted aggregate in the limit).
Outputs: figure_code/out/daily_vol_sim.csv, daily_vol_data.csv (aggregates only).
Runtime (single core, OPENBLAS_NUM_THREADS=1): about 2 minutes for the simulation and
about 10 minutes for --data.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, N_OBS  # noqa: E402

K = 3
_CACHE = {}


def _design(N):
    """Centring matrix M, the Gram matrix M o M of the regressors and its Cholesky factor.

    The least-squares problem  min_d || R - M diag(d) M ||_F  has normal equations
    (M o M) d = diag(M R M), because <m_s m_s', m_t m_t'> = M_st^2 and m_s' R m_s =
    (M R M)_ss.  M o M (Hadamard square) is positive definite, so the solution is unique.
    """
    if N not in _CACHE:
        M = np.eye(N) - np.ones((N, N)) / N
        G = M * M
        _CACHE[N] = (M, G, np.linalg.cholesky(G))
    return _CACHE[N]


def _mdm(d):
    """M diag(d) M without forming the product: d_s 1{s=t} - (d_s + d_t)/N + sum(d)/N^2."""
    N = len(d)
    return np.diag(d) - (d[:, None] + d[None, :]) / N + d.sum() / N ** 2


def _fit_d(R, G, C):
    N = R.shape[0]
    r1 = R.sum(axis=1)
    b = np.diag(R) - 2 * r1 / N + r1.sum() / N ** 2       # diag(M R M)
    d = np.linalg.solve(G, b)
    if (d < 0).any():                                   # non-negative version, same objective
        d = nnls(C.T, np.linalg.solve(C, b))[0]
    return d


def fa_floor(Wd, k, iters=1000, tol=1e-8, d0=None):
    """(A_j, A_j^FA, theta, U, d, iterations) for the N x N dual matrix of a demeaned panel.

    Principal-axis iteration for W ~ L L' + M diag(d) M (L of rank k, d >= 0).  Stops when
    no day variance moves by more than tol times their mean; 'iterations' = iters means
    the fit did not converge.  Non-convergence can happen when one day's specific variance
    rivals a factor (the decomposition is then still locally identified, but the default
    start, every d_t equal to the Theorem 2 noise level, lies outside the basin of
    attraction; checks/crisis_day_check.py) and when single-name outliers look like
    volatile days.
    d0 overrides the start.
    """
    N = Wd.shape[0]; n = N - 1
    _, G, C = _design(N)
    th, U = np.linalg.eigh(Wd); th = th[::-1][:n]; U = U[:, ::-1][:, :n]
    ell = th[k:].mean(); Uk = U[:, :k]
    d = np.full(N, ell * n / (N - 1)) if d0 is None else np.asarray(d0, float).copy()
    for it in range(iters):
        e, V = np.linalg.eigh(Wd - _mdm(d))
        Lk = V[:, -k:] * np.sqrt(np.maximum(e[-k:], 0))
        d_new = _fit_d(Wd - Lk @ Lk.T, G, C)
        if np.max(np.abs(d_new - d)) <= tol * np.abs(d).mean():
            d = d_new; break
        d = d_new
    A = ell / th[:k]
    Afa = (Uk ** 2 * d[:, None]).sum(axis=0) / th[:k]
    return A, Afa, th, U, d, it + 1


# ---------------------------------------------------------------- simulation
def calibration():
    d = pd.read_csv(OUT / "panel_stats.csv")
    g = d[(d.n_assets == 3000) & (d.k >= 3)]
    ell = float(np.median(g.ell))
    th = np.array([g.theta1.median(), g.theta2.median(), g.theta3.median()])
    return th - ell, ell


ARMS = {"constant": dict(cv=0.0, gamma=0, names=0.0, tdf=None),
        "dayvar": dict(cv=0.6, gamma=0, names=0.0, tdf=None),
        "dayvar+comove": dict(cv=0.6, gamma=1, names=0.0, tdf=None),
        "realistic": dict(cv=0.5, gamma=1, names=1.49, tdf=5)}


def simulate(draws=8, reps=40, p_grid=(500, 3000, 20000)):
    lam, ell = calibration()
    N = N_OBS; n = N - 1; delta2 = n * ell
    M, _, _ = _design(N)
    rows = []
    for ai, (arm, a) in enumerate(ARMS.items()):
        for dr in range(draws):
            rng = np.random.default_rng([ai, dr, 2026])
            if a["cv"] > 0:
                s2 = np.log(1 + a["cv"] ** 2); v = rng.lognormal(-s2 / 2, np.sqrt(s2), N); v /= v.mean()
            else:
                v = np.ones(N)
            Fr = rng.standard_normal((K, N)) * v ** (a["gamma"] / 2)
            Fr -= Fr.mean(axis=1, keepdims=True)
            Lc = np.linalg.cholesky(Fr @ Fr.T / n)
            Phi = np.sqrt(lam)[:, None] * np.linalg.solve(Lc, Fr)
            # exact p = infinity values
            S = Phi.T @ Phi / n
            Dt = M @ np.diag(delta2 * v / n) @ M
            Ainf, AFinf, thi, Ui, _, _ = fa_floor(S + Dt, K)
            OEinf = np.array([Ui[:, j] @ Dt @ Ui[:, j] for j in range(K)]) / thi[:K]
            for j in range(K):
                rows.append(dict(arm=arm, draw=dr, p=np.inf, j=j + 1, A=Ainf[j], A_fa=AFinf[j], OE=OEinf[j]))
            for p in p_grid:
                rp = np.random.default_rng([ai, dr, p, 7])
                Q, _ = np.linalg.qr(rp.standard_normal((p, K)))
                sig_names = np.ones(p)
                if a["names"] > 0:
                    s2n = np.log(1 + a["names"] ** 2)
                    sig_names = rp.lognormal(-s2n / 2, np.sqrt(s2n), p); sig_names /= sig_names.mean()
                Ssig = np.sqrt(p) * Q @ Phi
                acc = []
                for r in range(reps):
                    rr = np.random.default_rng([ai, dr, p, r, 11])
                    if a["tdf"]:
                        E = rr.standard_t(a["tdf"], (p, N)) / np.sqrt(a["tdf"] / (a["tdf"] - 2))
                    else:
                        E = rr.standard_normal((p, N))
                    Y = Ssig + E * np.sqrt(delta2 * sig_names)[:, None] * np.sqrt(v)[None, :]
                    Y = Y - Y.mean(axis=1, keepdims=True)
                    Wd = Y.T @ Y / (n * p)
                    A, Af, th, U, _, _ = fa_floor(Wd, K)
                    H = Y @ U[:, :K]; H /= np.linalg.norm(H, axis=0)
                    OE = 1 - ((Q.T @ H) ** 2).sum(axis=0)
                    acc.append(np.r_[A, Af, OE])
                acc = np.array(acc).mean(axis=0)
                for j in range(K):
                    rows.append(dict(arm=arm, draw=dr, p=p, j=j + 1, A=acc[j], A_fa=acc[K + j], OE=acc[2 * K + j]))
            print(arm, dr, flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / "daily_vol_sim.csv", index=False)
    return out


# ---------------------------------------------------------------- data
def data():
    import rawpanel
    from p01_panel_stats import jung_counts
    Kc = jung_counts(); U = rawpanel.universes(); W = rawpanel.wide()
    rows = []
    for (fd, P), sel in U.groupby(["formation_date", "n_assets"]):
        permnos = sel.sort_values("rank").permno.to_numpy()
        X = W.loc[pd.DatetimeIndex(rawpanel.window_dates(fd, N_OBS)), permnos].to_numpy(float)
        k = int(Kc.loc[(fd, P)])
        for tag, Xa in (("raw", X), ("clip", np.clip(X, -1, 1))):
            A_ = Xa - Xa.mean(axis=0)
            n = A_.shape[0] - 1
            Wd = A_ @ A_.T / (n * P)
            A, Af, th, Uu, dd, it = fa_floor(Wd, k)
            lev = (Uu[:, :k] ** 2).sum(axis=1) + 1 / Wd.shape[0]
            # co-movement condition of Proposition C, with the fitted day variances:
            # b/k >= T/n, b = tr(P_F Dt), T = tr Dt, Dt = M diag(d) M, P_F = the factor
            # path's day space (top-k eigenvectors of W - Dt)
            Mc = np.eye(len(dd)) - 1.0 / len(dd)
            Dt = Mc @ (dd[:, None] * Mc)
            _, Vf = np.linalg.eigh(Wd - Dt); Vf = Vf[:, -k:]
            comove = float((np.trace(Vf.T @ Dt @ Vf) / k) / (np.trace(Dt) / n))
            row = dict(formation_date=fd, n_assets=P, k=k, arm=tag, iterations=it,
                       converged=bool(it < 1000), day_var_cv=float(dd.std() / dd.mean()),
                       max_day_share=float(dd.max() / dd.sum()), zero_days=int((dd <= 0).sum()),
                       max_leverage=float(lev.max()), comove_ratio=comove,
                       min_rel_gap=float(np.min((th[:min(3, k)] - th[1:min(3, k) + 1]) / th[:min(3, k)])))
            for j in range(min(3, k)):
                row[f"A{j+1}"] = A[j]; row[f"Afa{j+1}"] = Af[j]
            rows.append(row)
        if len(rows) % 200 == 0:
            print("data", len(rows) // 2, "panels", flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / "daily_vol_data.csv", index=False)
    return out


if __name__ == "__main__":
    t0 = time.time()
    if "--skip-sim" not in sys.argv:
        s = simulate()
        deg = lambda x: np.degrees(np.arcsin(np.sqrt(np.clip(x, 0, 1))))
        g = s.groupby(["arm", "p", "j"])[["A", "A_fa", "OE"]].mean()
        print(deg(g).round(2).to_string()); print(f"sim done {time.time()-t0:.0f}s")
    if "--data" in sys.argv:
        dd = data(); print(f"data done {time.time()-t0:.0f}s")
