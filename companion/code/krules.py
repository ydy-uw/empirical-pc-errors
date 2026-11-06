"""Factor-count rules, including the Marchenko-Pastur family.

Conventions.  A panel is an N x p matrix of returns (observations by securities).
Everything is CENTERED: Ac = A - colmean over time, so the dual matrix
W = Ac Ac' /(N p) has rank at most N-1.  Its ordered eigenvalues are theta_1 >= ...
and the bulk average after removing m spikes is

    ell_m = (tr W - sum_{i<=m} theta_i) / (N - 1 - m),

which is the same estimator Theorem 2 uses.  Under the theorem's noise model the
non-degenerate eigenvalues of the pure-noise dual matrix are those of a Wishart in
d = N-1-m dimensions with p degrees of freedom, scaled so that their mean is ell_m;
the Marchenko-Pastur law therefore places them in ell_m [(1-sqrt(d/p))^2,
(1+sqrt(d/p))^2].
"""
import numpy as np

from sim import bartlett_wishart


MODE_CODE = {"plain": 0, "dayvol": 1}


def stable_rng(*keys):
    """Deterministic generator from a tuple of keys.

    Python randomises the hash of str and bytes per process, so a seed built with
    hash() on anything containing a string is not reproducible across runs.  This
    maps every key to an integer first (strings through MODE_CODE) and feeds them
    to a SeedSequence, which is reproducible by construction.
    """
    ints = [MODE_CODE[k] if isinstance(k, str) else int(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(ints))


def centered_eigs(A, kmax=30):
    """Ordered eigenvalues of W = Ac Ac'/(N p) (all N of them, last one ~0)."""
    A = np.asarray(A, float)
    N, p = A.shape
    Ac = A - A.mean(axis=0, keepdims=True)
    ev = np.linalg.eigvalsh(Ac @ Ac.T / (N * p))[::-1]
    return np.clip(ev, 0.0, None)


def bulk_mean(ev, m, N=None):
    """ell_m with the centered N-1-m denominator."""
    N = len(ev) if N is None else N
    d = N - 1 - m
    return float((ev.sum() - ev[:m].sum()) / d) if d > 0 else np.nan


# --------------------------------------------------------------------------
# Marchenko-Pastur family
# --------------------------------------------------------------------------
def mp_edge(ev, p, N=None, kmax=20, tw=False):
    """Iterative Marchenko-Pastur upper-edge rule.

    Increment m while theta_{m+1} exceeds the MP upper edge of a bulk whose mean
    is ell_m in d = N-1-m dimensions.  This is the eigenvalue-clipping rule of the
    random-matrix literature on correlation matrices (Laloux et al. 1999), with the
    noise level re-fitted after each removal so leftover signal does not inflate it.
    With tw=True the edge carries the Tracy-Widom finite-size correction.  At fixed
    d that correction is the wrong order -- the fluctuation of theta_1/ell is
    O(p^{-1/2}), not O(p^{-2/3}) -- and on these panels it changes the count in
    more than half of them without changing any conclusion.
    """
    ev = np.asarray(ev, float); N = len(ev) if N is None else N
    for m in range(kmax + 1):
        d = N - 1 - m
        if d <= 1 or m + 1 >= len(ev): return m
        ell = bulk_mean(ev, m, N)
        c = d / p
        edge = (1.0 + np.sqrt(c)) ** 2
        if tw:
            # (1+sqrt(c))^2 + p^{-2/3} (1+sqrt(c))^{4/3} c^{-1/6} * s, s = TW1 0.95 pt
            edge += 0.9793 * p ** (-2.0 / 3.0) * (1 + np.sqrt(c)) ** (4.0 / 3.0) * c ** (-1.0 / 6.0)
        if ev[m] <= ell * edge:
            return m
    return kmax


_GOE_CACHE = {}


def goe_quantile(d, q=0.95, draws=20000, seed=7):
    """Quantile of lambda_1(W_d) - mean(lambda(W_d)) for a d x d GOE.

    With Z p x d of i.i.d. entries of variance sigma^2 and S = Z'Z/p, at fixed d
    and p -> infinity  sqrt(p)(S/sigma^2 - I) => W_d, a GOE with unit off-diagonal
    and variance-2 diagonal.  Hence theta_1/ell = 1 + (lambda_1 - mean)/sqrt(p) +
    o(p^{-1/2}), which is the fluctuation the Marchenko-Pastur edge -- an
    O(p^{-1/2}) object only in proportional asymptotics -- does not describe when
    d stays fixed.
    """
    key = (int(d), float(q), int(draws), int(seed))
    if key in _GOE_CACHE: return _GOE_CACHE[key]
    rng = np.random.default_rng(seed + int(d))
    out = np.empty(draws)
    for i in range(draws):
        A = rng.standard_normal((d, d))
        W = (A + A.T) / np.sqrt(2.0)
        e = np.linalg.eigvalsh(W)
        out[i] = e[-1] - e.mean()
    v = float(np.quantile(out, q))
    _GOE_CACHE[key] = v
    return v


_WIS_CACHE = {}


def wishart_quantile(d, p, q=0.95, draws=3000, seed=11):
    """Exact finite-(d,p) Gaussian null quantile of theta_1/ell.

    The nonzero eigenvalues of the noise part of a centred dual matrix are those of
    a Wishart in d dimensions with p degrees of freedom.  Simulating that Wishart
    gives the exact null distribution of the ratio every edge rule thresholds, with
    no asymptotics in either d or p.
    """
    key = (int(d), int(p), float(q), int(draws))
    if key in _WIS_CACHE: return _WIS_CACHE[key]
    rng = np.random.default_rng(seed + 1000 * int(d) + int(p))
    out = np.empty(draws)
    step = 200
    done = 0
    while done < draws:
        r = min(step, draws - done)
        W = bartlett_wishart(rng, r, int(d), int(p))
        e = np.linalg.eigvalsh(W)[:, ::-1]
        out[done:done + r] = e[:, 0] / e.mean(axis=1)
        done += r
    v = float(np.quantile(out, q))
    _WIS_CACHE[key] = v
    return v


def wishart_edge(ev, p, N=None, kmax=20, q=0.95, draws=3000):
    """Sequential edge at the exact finite-(d,p) Gaussian null quantile."""
    ev = np.asarray(ev, float); N = len(ev) if N is None else N
    for m in range(kmax + 1):
        d = N - 1 - m
        if d <= 1 or m + 1 >= len(ev): return m
        if ev[m] <= bulk_mean(ev, m, N) * wishart_quantile(d, p, q, draws):
            return m
    return kmax


def mp_edge_size(d, p, draws=3000, seed=11):
    """Type-I rate of one Marchenko-Pastur edge step under the exact Gaussian null."""
    rng = np.random.default_rng(seed + 7919 * int(d) + int(p))
    hit = 0; done = 0; step = 200
    edge = (1.0 + np.sqrt(d / p)) ** 2
    while done < draws:
        r = min(step, draws - done)
        W = bartlett_wishart(rng, r, int(d), int(p))
        e = np.linalg.eigvalsh(W)[:, ::-1]
        hit += int(np.sum(e[:, 0] / e.mean(axis=1) > edge))
        done += r
    return hit / draws


def onatski_ed(ev, kmax=20, iters=20):
    """Onatski (2010) 'edge distribution' estimator.

    Calibrates the eigenvalue-gap threshold from the curvature of the bulk, so it
    needs no variance estimate.  Uses the same fourth-power-of-two-thirds design
    matrix as the original.
    """
    ev = np.asarray(ev, float); n = len(ev)
    j = kmax + 1
    r = 0
    for _ in range(iters):
        hi = min(j + 4, n - 1)
        idx = np.arange(j, hi + 1)
        if len(idx) < 2: return r
        x = (idx - 1.0) ** (2.0 / 3.0)
        y = ev[idx - 1]
        b = np.polyfit(x, y, 1)[0]
        delta = 2.0 * abs(b)
        gaps = ev[:kmax] - ev[1:kmax + 1]
        hits = np.flatnonzero(gaps >= delta)
        rn = int(hits[-1] + 1) if hits.size else 0
        if rn == r: return r
        r = rn; j = r + 1
    return r


def permutation_edge(A, alpha=0.05, kmax=20, B=20, rng=None):
    """Finite-sample Marchenko-Pastur edge, calibrated by cross-sectional permutation.

    The asymptotic edge is a p -> infinity, N -> infinity statement.  Here N=63 is
    fixed, so we replace it by its exact finite-sample analogue: permute each
    security's own time series independently, which destroys cross-sectional
    dependence while preserving every marginal, and read off the 1-alpha quantile of
    theta_{m+1}/ell_m.  This is Horn's parallel analysis run sequentially on the
    ratio the theorem actually uses.
    """
    A = np.asarray(A, float); N, p = A.shape
    rng = np.random.default_rng(0) if rng is None else rng
    ratios = np.empty((B, kmax + 1))
    for b in range(B):
        Ab = permute_columns(A, rng)
        e = centered_eigs(Ab)
        for m in range(kmax + 1):
            ratios[b, m] = e[m] / bulk_mean(e, m, N) if N - 1 - m > 1 else np.inf
    thr = np.quantile(ratios, 1 - alpha, axis=0)
    ev = centered_eigs(A)
    for m in range(kmax + 1):
        if N - 1 - m <= 1 or m + 1 >= len(ev): return m, thr
        if ev[m] <= bulk_mean(ev, m, N) * thr[m]:
            return m, thr
    return kmax, thr


# --------------------------------------------------------------------------
# Information-criterion and ratio rules (for reference)
# --------------------------------------------------------------------------
def bai_ng(ev, p, N=None, kmax=20):
    ev = np.asarray(ev, float); N = len(ev) if N is None else N
    tot = ev.sum()
    V = np.array([tot - ev[:m].sum() for m in range(kmax + 1)])
    V = np.clip(V, 1e-300, None)
    g1 = (N + p) / (N * p) * np.log(N * p / (N + p))
    g2 = (N + p) / (N * p) * np.log(min(N, p))
    m = np.arange(0, kmax + 1)          # k=0 is admissible, as in Bai-Ng (2002)
    ic1 = np.log(V) + m * g1
    ic2 = np.log(V) + m * g2
    return int(np.argmin(ic1)), int(np.argmin(ic2))


def ahn_horenstein(ev, kmax=20, mock=True):
    """Eigenvalue ratio of Ahn and Horenstein (2013).

    With ``mock=True`` the rule carries the mock eigenvalue the original defines,
    mu_0 = sum(ev)/log(min(n,p)), so that k=0 is admissible; without it the rule
    cannot return 0 and its size against a factorless panel is one by construction.
    """
    ev = np.asarray(ev, float)
    d = np.clip(ev[:kmax + 1], 1e-300, None)
    er = ev[:kmax] / d[1:kmax + 1]
    if not mock:
        return int(np.argmax(er) + 1)
    n_eff = np.count_nonzero(ev > 1e-14)
    mu0 = ev.sum() / np.log(max(n_eff, 2))
    er0 = mu0 / max(ev[0], 1e-300)
    allr = np.concatenate([[er0], er])
    return int(np.argmax(allr))


def count_all(A, p=None, N=None, kmax=20, B=20, rng=None, alpha=0.05, jung=True):
    """Every rule on one panel.  Returns a dict of counts."""
    from jung_ref import jung_factor_count
    A = np.asarray(A, float); N, p = A.shape
    ev = centered_eigs(A)
    b1, b2 = bai_ng(ev, p, N, kmax)
    out = dict(k_mpedge=mp_edge(ev, p, N, kmax), k_onatski=onatski_ed(ev, kmax),
               k_baing1=b1, k_baing2=b2, k_ahnhoren=ahn_horenstein(ev, kmax))
    kp, thr = permutation_edge(A, alpha=alpha, kmax=kmax, B=B, rng=rng)
    out["k_perm"] = kp
    out["_thr"] = thr
    if jung:
        for a in (0.05, 0.10, 0.20):
            s = jung_factor_count(A, alpha=a, max_factors=kmax, center=True).selected
            out[f"k_jung{int(100*a):03d}"] = -1 if s is None else int(s)
    return out


def block_corr(E, groups, G):
    """Group-by-group mean residual correlation, without forming the p x p matrix.

    E is (n x p) residual; columns are standardised to unit norm, so C = E'E has a
    unit diagonal.  For a partition into G groups the block means are
    s_g' s_h /(|g||h|) with s_g the sum of the standardised columns in group g;
    the diagonal block subtracts the |g| unit diagonal entries.
    """
    nrm = np.linalg.norm(E, axis=0)
    keep = nrm > 1e-12
    E = E[:, keep] / nrm[keep]
    g = np.asarray(groups)[keep]
    n, p = E.shape
    S = np.zeros((n, G)); cnt = np.zeros(G)
    for i in range(G):
        sel = g == i
        cnt[i] = sel.sum()
        if cnt[i]: S[:, i] = E[:, sel].sum(axis=1)
    Bm = S.T @ S
    Bm = Bm / np.outer(cnt, cnt)
    for i in range(G):
        if cnt[i] > 1:
            Bm[i, i] = (S[:, i] @ S[:, i] - cnt[i]) / (cnt[i] * (cnt[i] - 1))
        else:
            Bm[i, i] = np.nan
    tot = E.sum(axis=1)
    mean_r = float((tot @ tot - p) / (p * (p - 1)))
    Gm = E @ E.T
    mean_r2 = float((np.sum(Gm * Gm) - p) / (p * (p - 1)))
    return Bm, mean_r, mean_r2, int(p)


def residuals_after(A, m):
    """Centred panel with its top-m principal components removed (n x p)."""
    Ac = np.asarray(A, float)
    Ac = Ac - Ac.mean(axis=0, keepdims=True)
    if m <= 0: return Ac
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    return Ac - (U[:, :m] * S[:m]) @ Vt[:m]


def permute_columns(A, rng):
    """Permute each security's own time series independently."""
    return rng.permuted(np.asarray(A, float), axis=0)


def resample_residual(E, rng, mode="plain"):
    """Within-security permutation of a residual panel.

    "plain"  : permute each security's own time series independently.  Destroys all
               cross-sectional dependence; preserves every marginal exactly.
    "dayvol" : the same, but applied to day-standardised residuals which are then
               re-scaled by the original daily cross-sectional scale.  This null has
               no cross-sectional correlation yet carries the panel's own common
               time-varying volatility, so a spectrum it cannot reproduce is evidence
               of a factor rather than of a volatile day.
    """
    E = np.asarray(E, float)
    if mode == "plain":
        return permute_columns(E, rng)
    s = np.sqrt((E ** 2).mean(axis=1))
    s = np.where(s > 0, s, 1.0)
    return permute_columns(E / s[:, None], rng) * s[:, None]


def sequential_edge(A, alpha=0.05, kmax=20, B=20, rng=None, mode="plain",
                    return_path=False, full_path=False):
    """Sequential edge test against an m-factor resampling null.

    At step m the null panel keeps the data's own top-m principal components and
    replaces the residual by a cross-sectional resample of itself, so the null has
    exactly m factors and the data's own marginals (and, with mode="dayvol", its own
    daily volatility).  We stop at the first m whose observed theta_{m+1}/ell_m falls
    inside the null's 1-alpha quantile.
    """
    A = np.asarray(A, float); N, p = A.shape
    rng = np.random.default_rng(0) if rng is None else rng
    Ac = A - A.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    ev = np.clip(np.linalg.eigvalsh(Ac @ Ac.T / (N * p))[::-1], 0, None)
    obs, thr = [], []
    stop = None
    for m in range(kmax + 1):
        if N - 1 - m <= 2 or m + 1 >= len(ev): break
        o = ev[m] / bulk_mean(ev, m, N)
        sig = (U[:, :m] * S[:m]) @ Vt[:m] if m else 0.0
        E = Ac - sig
        r = np.empty(B)
        for b in range(B):
            Ab = sig + resample_residual(E, rng, mode)
            e = centered_eigs(Ab)
            r[b] = e[m] / bulk_mean(e, m, N)
        t = float(np.quantile(r, 1 - alpha))
        obs.append(float(o)); thr.append(t)
        if o <= t and stop is None:
            stop = m
            if not full_path:
                return (m, obs, thr) if return_path else m
    if stop is None: stop = kmax
    return (stop, obs, thr) if return_path else stop
