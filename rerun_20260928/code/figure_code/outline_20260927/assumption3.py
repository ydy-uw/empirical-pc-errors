# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""Date-invariance of the specific variances (Assumption 2 of the theory paper, v25) on the clipped P = 3,000 panels.

Two questions.
  (1) How much does the cross-sectional average specific variance move from day to day inside
      a window, and does a shorter window make it steadier?
  (2) What does the observed movement do to the Theorem 2 estimate A_j = ell/theta_j?

Day variance.  For a demeaned N x P panel Y with rank-k PCA fit U_k S_k V_k' and residual E, the
residual dispersion of day t is d_t = mean_i E_ti^2.  The fit absorbs part of the noise on days
with high leverage, so d_t is divided by Q_tt = 1 - 1/N - sum_{j<=k} U_tj^2 (the diagonal of the
residual projector; under a constant day variance E d_t = delta^2 Q_tt).  The day-variance
dispersion of a window is the coefficient of variation over days of delta_t^2 := d_t / Q_tt.

Calibrated simulation (per month).  Systematic part: the month's own rank-k PCA fit with its
singular values shrunk to sqrt(n P lambda_j), starting from lambda_j = theta_j - ell * sum_t U_tj^2 s_t^2,
the signal net of the specific energy that the arm's day-variance path s_t^2 puts on direction j's
days (theta_j - ell for a constant day variance), then corrected by two pilot simulations so that
the simulated theta_1..theta_5 and the bulk level ell match the data in both arms (this removes the
finite-p upward bias of the sample eigenvalues; the stock variances are rescaled by a common factor); the loading span is span(V_k) and the factor path has the realized timing.  Specific
returns Z_ti = sigma_i s_t eps_ti with sigma_i^2 the stock's residual variance, eps standardized
t_5, and the simulated panel clipped at +/-50% like the data;
    const   s_t = 1 (the specific variances do not change over the window);
    dayvar  s_t^2 = 1 + c (delta_t^2 / mean delta^2 - 1), the month's own day-variance path,
            shrunk by c = sqrt(max(0, 1 - CV_const^2 / CV_data^2)) so that the measured
            dispersion net of its sampling noise matches the data (method of moments).
On every simulated panel: A_j, the factor-analytic estimate A_j^FA (principal-axis fit of the dual
matrix as a rank-k part plus a separate specific variance for each day, fa_floor of
figure_code/paper/p04_daily_volatility.py), and the true OE_j = 1 - ||V_k' h_j||^2.  Also the
rotation of h_j inside the planted span away from its own planted direction v_j (used for
Assumptions 5-6 in separation.py's summary).

Designs.
  core   N = 63 (the paper's window), k = the month's skewness count, 4 replicates per arm, with
         the factor-analytic estimate.
  sweep  N in {21, 42, 63, 126, 252} days ending at the formation date, k = 3 for all N, stocks of
         the month's 3,000 with a return on every day of the window; data, 1 const replicate and
         2 dayvar replicates, without the factor-analytic estimate.

Inputs: the licensed CRSP extract through companion/code/rawpanel.py (read-only) and
outlier_cleaning/experiments/01_fixed_clip/results/panels_c050.csv (counts).
Outputs (aggregates only, no identifiers): figure_code/out/outline_20260927/
    a3_core.csv   one row per month x source (data, const, dayvar) x replicate
    a3_sweep.csv  one row per month x N x source x replicate
Run from the project root:  OPENBLAS_NUM_THREADS=1 python figure_code/outline_20260927/assumption3.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[4]
import os  # noqa: E402
WORK = Path(os.environ.get("HLHH_RERUN_OUT", ROOT / "rerun_20260928" / "work"))
sys.path.insert(0, str(ROOT / "companion" / "code"))
sys.path.insert(0, str(ROOT / "figure_code"))
sys.path.insert(0, str(ROOT / "figure_code" / "paper"))
import rawpanel  # noqa: E402
from p04_daily_volatility import fa_floor  # noqa: E402

OUT = WORK / "figure_code" / "out" / "outline_20260927"
CLP = WORK / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results"
P, CLIP, TDF = 3000, 0.5, 5
CORE_N, CORE_REPS = 63, 4
SWEEP_N, SWEEP_K = (21, 42, 63, 126, 252), 3
J = 3


def decompose(Y, k):
    """Demeaned panel -> (theta, U, S, Vt, day-variance path delta_t^2, sigma_i^2)."""
    N = Y.shape[0]; n = N - 1
    Uf, s, Vt = np.linalg.svd(Y, full_matrices=False)
    theta = (s ** 2 / (n * Y.shape[1]))[:n]
    E = Y - (Uf[:, :k] * s[:k]) @ Vt[:k]
    Q = 1.0 - 1.0 / N - (Uf[:, :k] ** 2).sum(axis=1)
    dvar = (E ** 2).mean(axis=1) / Q
    sig2 = (E ** 2).sum(axis=0) / Q.sum()
    return theta, Uf, s, Vt, dvar, sig2


def estimates(Y, k, fa, Vk=None):
    """A_j (and A_j^FA, OE_j and in-span rotation when Vk is given) for j <= min(k, 3)."""
    N = Y.shape[0]; n = N - 1
    theta, Uf, s, Vt, dvar, _ = decompose(Y, k)
    out = dict(cv_dayvar=float(dvar.std() / dvar.mean()), ell=float(theta[k:].mean()))
    m = min(k, J)
    for j in range(m):
        out[f"A{j+1}"] = float(theta[k:].mean() / theta[j])
    for j in range(5):
        out[f"theta{j+1}"] = float(theta[j])
    if fa:
        Wd = Y @ Y.T / (n * Y.shape[1])
        _, Afa, _, _, _, it = fa_floor(Wd, k)
        for j in range(m):
            out[f"Afa{j+1}"] = float(Afa[j])
        out["fa_iter"] = int(it)
    if Vk is not None:
        H = Vt[:m].T                                   # sample directions h_j (unit, P-vectors)
        C = Vk.T @ H                                   # k x m coordinates in the planted span
        inside = (C ** 2).sum(axis=0)
        for j in range(m):
            out[f"OE{j+1}"] = float(1.0 - inside[j])
            out[f"rot{j+1}"] = float(np.degrees(np.arccos(min(1.0, abs(C[j, j]) / np.sqrt(inside[j])))))
    return out


def simulate(fit, Vk, sig2, path, k, rng, fa):
    N, Pn = fit.shape
    eps = rng.standard_t(TDF, size=(N, Pn)) / np.sqrt(TDF / (TDF - 2))
    Z = eps * np.sqrt(sig2)[None, :] * np.sqrt(path)[:, None]
    Y = np.clip(fit + Z, -CLIP, CLIP)                  # the same treatment as the data
    Y = Y - Y.mean(axis=0, keepdims=True)
    return estimates(Y, k, fa, Vk)


def run_month(Xfull, k, rng, n_days, reps, fa):
    """Data row plus const and dayvar simulation rows for one window."""
    X = Xfull[-n_days:]
    X = X[:, ~np.isnan(X).any(axis=0)]
    Y = np.clip(X, -CLIP, CLIP); Y = Y - Y.mean(axis=0, keepdims=True)
    theta, Uf, s, Vt, dvar, sig2 = decompose(Y, k)
    # planted signal: the fit with its noise share removed, so that the simulated theta_j match
    # the data.  To first order theta_j = lambda_j + ell * sum_t U_tj^2 s_t^2, where s_t^2 (mean one)
    # is the arm's day-variance path; so lambda_j = theta_j - ell * sum_t U_tj^2 s_t^2, which is
    # theta_j - ell for a constant day variance.
    n = Y.shape[0] - 1
    ell = theta[k:].mean()
    Vk = Vt[:k].T

    def planted(path):
        # first-order start, then two pilot simulations that remove the remaining finite-p gaps
        # between the simulated and the observed theta_1..theta_min(k,5) and bulk level ell
        lam = np.maximum(theta[:k] - ell * (Uf[:, :k] ** 2 * path[:, None]).sum(axis=0), 1e-12)
        s2 = sig2.copy()
        m = min(k, 5)
        for _ in range(2):
            fit = (Uf[:, :k] * np.sqrt(n * Y.shape[1] * lam)) @ Vt[:k]
            pilot = simulate(fit, Vk, s2, path, k, rng, False)
            s2 = s2 * ell / pilot["ell"]
            th_sim = np.array([pilot[f"theta{j+1}"] for j in range(m)])
            lam[:m] = np.maximum(lam[:m] - (th_sim - theta[:m]), 1e-3 * lam[:m])
        return (Uf[:, :k] * np.sqrt(n * Y.shape[1] * lam)) @ Vt[:k], s2
    rows = [dict(source="data", rep=0, p=Y.shape[1], **estimates(Y, k, fa))]
    cv_data = rows[0]["cv_dayvar"]
    ones = np.ones(Y.shape[0])
    fit0, s20 = planted(ones)
    const = [simulate(fit0, Vk, s20, ones, k, rng, fa) for _ in range(reps)]
    cv_const = float(np.mean([c["cv_dayvar"] for c in const]))
    c = np.sqrt(max(0.0, 1.0 - cv_const ** 2 / cv_data ** 2))
    path = np.maximum(1.0 + c * (dvar / dvar.mean() - 1.0), 0.02)
    path /= path.mean()
    dv_reps = reps if reps > 1 else 2
    fit1, s21 = planted(path)
    dayv = [simulate(fit1, Vk, s21, path, k, rng, fa) for _ in range(dv_reps)]
    rows += [dict(source="const", rep=r, p=Y.shape[1], shrink=c, **x) for r, x in enumerate(const)]
    rows += [dict(source="dayvar", rep=r, p=Y.shape[1], shrink=c, **x) for r, x in enumerate(dayv)]
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(CLP / "panels_c050.csv", parse_dates=["formation_date"])
    counts = counts[counts.p == P].set_index("formation_date").k_skew
    U, W = rawpanel.universes(), rawpanel.wide()
    months = sorted(U[U.n_assets == P].formation_date.unique())
    core, sweep = [], []
    t0 = time.time()
    for mi, fd in enumerate(months):
        fd = pd.Timestamp(fd)
        sel = U[(U.n_assets == P) & (U.formation_date == fd)].sort_values("rank").permno.to_numpy()
        dts = rawpanel.window_dates(fd, max(SWEEP_N))
        Xfull = W.reindex(index=pd.DatetimeIndex(dts), columns=sel).to_numpy(float)
        k = int(counts.loc[fd])
        rng = np.random.default_rng(20260927 + mi)
        for r in run_month(Xfull, k, rng, CORE_N, CORE_REPS, fa=True):
            core.append(dict(formation_date=fd, k=k, **r))
        for n_days in SWEEP_N:
            rng2 = np.random.default_rng([20260928, mi, n_days])
            for r in run_month(Xfull, SWEEP_K, rng2, n_days, 1, fa=False):
                sweep.append(dict(formation_date=fd, N=n_days, k=SWEEP_K, **r))
        if mi % 10 == 0:
            print(f"{mi+1}/{len(months)} months, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(core).to_csv(OUT / "a3_core.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "a3_sweep.csv", index=False)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
