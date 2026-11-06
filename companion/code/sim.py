"""Finite-P audit of the Theorem-2 estimator.

Exact algebra used throughout (model Y = b*Phi + Z, p x n, rank-k loadings b):

    W      = Y'Y/(np) = C C' + R,     C = (b'Y)'/sqrt(np),  R = Y'(I-bb')Y/(np)
    theta_j, u_j       jth eigenpair of W
    rho_j  = u_j' R u_j
    OE_j   = sin^2 angle(h_j, span(b)) = rho_j / theta_j      (exact, any p)
    A_j    = ell / theta_j,  ell = (tr W - sum_{i<=m} theta_i)/(n-m)

    =>  A_j - OE_j = (ell - rho_j)/theta_j
        ell - rho_j = [mean_{i>m} rho_i - rho_j]  +  (1/(n-m)) sum_{i>m} ||C'u_i||^2
                       \-- selection tilt (<=0 typically) --/   \-- signal in bulk (>=0) --/
"""
from __future__ import annotations
import hashlib
import numpy as np

N_OBS = 63


def stable_seed(base, *parts):
    payload = "|".join([str(base), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def bartlett_wishart(rng, reps, dim, df):
    if df < dim:
        raise ValueError("Bartlett needs df >= dim")
    low = np.zeros((reps, dim, dim))
    r, c = np.tril_indices(dim, -1)
    low[:, r, c] = rng.standard_normal((reps, len(r)))
    for i in range(dim):
        low[:, i, i] = np.sqrt(rng.chisquare(df - i, size=reps))
    return low @ np.swapaxes(low, 1, 2)


def factor_draws(rng, reps, n, lam, dist, whiten=False):
    """Draw the n x k factor block with row covariance diag(lam).

    With ``whiten=True`` the *realised* Gram of each replicate is forced to
    diag(lam) exactly, so the path-conditional asymptotic floor equals the
    calibration target ell/theta_j rather than fluctuating around it.
    """
    k = len(lam)
    if dist == "gaussian":
        z = rng.standard_normal((reps, n, k))
    elif dist == "t6":
        z = rng.standard_t(6, size=(reps, n, k)) / np.sqrt(1.5)
    else:
        raise ValueError(dist)
    if whiten:
        q, _ = np.linalg.qr(z)                        # (reps, n, k), orthonormal columns
        z = q * np.sqrt(float(n))
    return z * np.sqrt(lam)[None, None, :]


def _diagnostics(W, C, R, k_true, m_list, n=N_OBS, keep_vecs=False):
    """Return per-replicate diagnostics for each reported factor and each k_used."""
    vals, vecs = np.linalg.eigh(W)                     # ascending
    theta = vals[:, ::-1]                              # descending, (reps, n)
    U = vecs[:, :, ::-1]                               # (reps, n, n)
    rho = np.einsum("rni,rnm,rmi->ri", U, R, U, optimize=True)
    sig = np.einsum("rnk,rni->rki", C, U, optimize=True)
    sig = np.sum(sig ** 2, axis=1)                     # ||C' u_i||^2, (reps, n)
    # identity check: theta_i = ||C'u_i||^2 + rho_i
    assert np.abs(theta - (sig + rho)).max() < 1e-10, np.abs(theta - (sig + rho)).max()
    trW = theta.sum(axis=1)
    rep = min(3, k_true)
    OEall = rho[:, :k_true] / theta[:, :k_true]        # every retained direction
    OE = OEall[:, :rep]
    out = {}
    for m in m_list:
        ell = (trW - theta[:, :m].sum(axis=1)) / (n - m)
        A = ell[:, None] / theta[:, :rep]
        Aall = ell[:, None] / theta[:, :k_true]
        tilt = rho[:, m:].mean(axis=1)[:, None] - rho[:, :rep]
        leak = sig[:, m:].mean(axis=1)[:, None] * np.ones((1, rep))
        assert np.abs((ell[:, None] - rho[:, :rep]) - (tilt + leak)).max() < 1e-12
        trail = theta[:, m:]
        out[m] = dict(A=A, ell=ell, tilt=tilt, leak=leak,
                      Asum=Aall.sum(axis=1), OEsum=OEall.sum(axis=1),
                      bulk_disp=trail.std(axis=1, ddof=1) / trail.mean(axis=1))
    # relative eigengap of each reported direction, a path-level covariate
    gap = np.empty_like(theta[:, :rep])
    for j in range(rep):
        gap[:, j] = (theta[:, j] - theta[:, j + 1]) / theta[:, j]
    res = dict(theta=theta[:, :rep], rho=rho[:, :rep], OE=OE, per_m=out, rep=rep,
               gap=gap, OEsum=OEall.sum(axis=1), trW=trW,
               OEall=OEall, theta_all=theta)
    if keep_vecs:
        res["U"] = U
    return res


def simulate_spherical(theta_fit, ell_fit, p, *, reps, seed, dist="gaussian",
                       m_list=None, n=N_OBS, factor_seed=None, whiten=False,
                       keep_vecs=False):
    """Wishart representation of spherical idiosyncratic noise (exact, no p x p matrix).

    `factor_seed` fixes the realised factor path independently of p, so that a
    p-sweep grows the cross-section along one path, as the theorem does.
    """
    theta_fit = np.asarray(theta_fit, float)
    k = len(theta_fit)
    lam = theta_fit - ell_fit
    if np.any(lam <= 0):
        raise ValueError("signal eigenvalues must exceed the bulk")
    if p - k < n:
        raise ValueError("need p-k >= n")
    d2 = n * ell_fit
    rng = np.random.default_rng(seed)
    frng = rng if factor_seed is None else np.random.default_rng(factor_seed)
    Csig = factor_draws(frng, reps, n, lam, dist, whiten=whiten) / np.sqrt(n)
    Cpar = np.sqrt(d2 / (n * p)) * rng.standard_normal((reps, n, k))
    C = Csig + Cpar
    R = (d2 / (n * p)) * bartlett_wishart(rng, reps, n, p - k)
    W = C @ np.swapaxes(C, 1, 2) + R
    out = _diagnostics(W, C, R, k, m_list or [k], n=n, keep_vecs=keep_vecs)
    # path-conditional asymptotic floor: a_j(F) = (d2/n) / (lam_hat_j + d2/n)
    lam_hat = np.linalg.eigvalsh(Csig @ np.swapaxes(Csig, 1, 2))[:, ::-1][:, :out["rep"]]
    out["a_path"] = (d2 / n) / (lam_hat + d2 / n)
    return out


def simulate_explicit(theta_fit, ell_fit, p, *, reps, seed, dist="gaussian",
                      cv=0.0, m_list=None, n=N_OBS, batch=25, noise="gaussian",
                      whiten=False):
    """Explicit p x n idiosyncratic draw.

    ``cv`` is the cross-sectional coefficient of variation of the per-security
    idiosyncratic variance; ``noise`` selects the idiosyncratic law
    ("gaussian" or "t5", standardised to unit variance)."""
    theta_fit = np.asarray(theta_fit, float)
    k = len(theta_fit)
    lam = theta_fit - ell_fit
    d2 = n * ell_fit
    rng = np.random.default_rng(seed)
    s2log = np.log1p(cv ** 2)
    acc = []
    done = 0
    while done < reps:
        r = min(batch, reps - done)
        done += r
        v = np.exp(rng.normal(-0.5 * s2log, np.sqrt(s2log), size=(r, p))) if cv > 0 \
            else np.ones((r, p))
        v = v / v.mean(axis=1, keepdims=True)
        sd = np.sqrt(d2 * v)[:, :, None]                      # (r, p, 1)
        if noise == "gaussian":
            E = rng.standard_normal((r, p, n))
        elif noise == "t5":
            E = rng.standard_t(5, size=(r, p, n)) / np.sqrt(5 / 3)
        else:
            raise ValueError(noise)
        Z = sd * E
        Fs = factor_draws(rng, r, n, lam, dist, whiten=whiten)  # (r, n, k)
        # Haar orthonormal loading frame, independent of sigma
        G = rng.standard_normal((r, p, k))
        b, _ = np.linalg.qr(G)                                # (r, p, k)
        Y = np.sqrt(p) * (b @ np.swapaxes(Fs, 1, 2)) + Z      # (r, p, n)
        Bt = np.swapaxes(b, 1, 2) @ Y                         # (r, k, n)
        C = np.swapaxes(Bt, 1, 2) / np.sqrt(n * p)
        W = np.swapaxes(Y, 1, 2) @ Y / (n * p)
        R = W - C @ np.swapaxes(C, 1, 2)
        acc.append(_diagnostics(W, C, R, k, m_list or [k], n=n))
    if len(acc) == 1:
        return acc[0]
    out = dict(theta=np.concatenate([a["theta"] for a in acc]),
               rho=np.concatenate([a["rho"] for a in acc]),
               OE=np.concatenate([a["OE"] for a in acc]),
               gap=np.concatenate([a["gap"] for a in acc]),
               OEsum=np.concatenate([a["OEsum"] for a in acc]),
               trW=np.concatenate([a["trW"] for a in acc]),
               rep=acc[0]["rep"], per_m={})
    for m in acc[0]["per_m"]:
        out["per_m"][m] = {kk: np.concatenate([a["per_m"][m][kk] for a in acc])
                           for kk in acc[0]["per_m"][m]}
    return out
