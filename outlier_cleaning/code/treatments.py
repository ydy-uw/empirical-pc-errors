"""Outlier treatments.  Each takes the raw 63 x P panel X (days x names, rank order)
and its context, and returns (Xt, keep, info):

  Xt    the treated panel (days x kept names);
  keep  boolean mask over the P names (None when every name is kept);
  info  what the treatment did: entries changed, names and days touched.

Nothing here reads or writes files; the original returns are never modified in place.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, rankdata

MAD_K = 1.4826                       # makes the MAD consistent for a normal sd


def _info(X, Xt, keep=None):
    if keep is not None and not keep.all():
        ch = np.zeros_like(X, dtype=bool); ch[:, ~keep] = True
        Xk = X[:, keep]
        chk = ~np.isclose(Xk, Xt, rtol=0, atol=1e-12)
        ch[:, keep] = chk
    else:
        ch = ~np.isclose(X, Xt, rtol=0, atol=1e-12)
    return dict(n_changed=int(ch.sum()), share_changed=float(ch.mean()),
                names_touched=int(ch.any(axis=0).sum()), days_touched=int(ch.any(axis=1).sum()),
                names_dropped=int((~keep).sum()) if keep is not None else 0)


def _robust_col_scale(X):
    """1.4826 * MAD of each column; the standard deviation where the MAD is zero."""
    med = np.median(X, axis=0)
    s = MAD_K * np.median(np.abs(X - med), axis=0)
    zero = s <= 0
    if zero.any():
        s[zero] = X[:, zero].std(axis=0)
    return med, s, int(zero.sum())


# ---------------------------------------------------------------------------------
def raw(X, ctx):
    return X, None, _info(X, X)


def fixed_clip(X, ctx, c=1.0):
    """Clip every daily simple return to [-c, +c]."""
    Xt = np.clip(X, -c, c)
    return Xt, None, _info(X, Xt)


def daily_winsor(X, ctx, q=0.001):
    """Winsorize each day's cross-section at its own q and 1-q quantiles."""
    lo = np.quantile(X, q, axis=1, keepdims=True)
    hi = np.quantile(X, 1 - q, axis=1, keepdims=True)
    Xt = np.clip(X, lo, hi)
    return Xt, None, _info(X, Xt)


def xs_mad(X, ctx, c=10.0):
    """Clip each day's cross-section at median +/- c * 1.4826 * MAD of that day."""
    med = np.median(X, axis=1, keepdims=True)
    s = MAD_K * np.median(np.abs(X - med), axis=1, keepdims=True)
    Xt = np.clip(X, med - c * s, med + c * s)
    return Xt, None, _info(X, Xt)


def ts_mad(X, ctx, c=10.0, scale="window"):
    """Clip each name's returns at its median +/- c * robust scale.

    scale='window': median and 1.4826 * MAD over the name's own 63 days in the window.
    scale='prior' : the scale from the 63 trading days before the window (ex ante);
                    names without a usable prior scale fall back to the window scale.
    """
    med, s, nzero = _robust_col_scale(X)
    nfall = 0
    if scale == "prior":
        ps = np.asarray(ctx["prior_scale"], float)
        ok = np.isfinite(ps) & (ps > 0)
        nfall = int((~ok).sum())
        s = np.where(ok, ps, s)
    Xt = np.clip(X, med - c * s, med + c * s)
    info = _info(X, Xt); info.update(zero_mad_names=nzero, fallback_names=nfall)
    return Xt, None, info


def log_returns(X, ctx):
    """Continuously compounded returns log(1 + r)."""
    Xt = np.log1p(X)
    info = _info(X, Xt)
    # every entry changes a little; also report how many change by more than 1 point
    info["n_changed_1pt"] = int((np.abs(Xt - X) > 0.01).sum())
    return Xt, None, info


def rank_normal(X, ctx, mode="scaled"):
    """Normal scores of each name's own 63 returns (average ranks for ties).

    mode='scaled'   : normal scores times the name's robust scale, so the names keep
                      their relative volatility (covariance-type PCA with Gaussian marginals);
    mode='unit'     : normal scores only (principal components of the Gaussian rank
                      correlation matrix of Boudt, Cornelissen and Croux, 2012);
    mode='standard' : control without ranks: each name demeaned and divided by its
                      standard deviation (correlation PCA).
    """
    n = X.shape[0]
    if mode == "standard":
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        return Z, None, _info(X, Z)
    R = rankdata(X, axis=0, method="average")
    Z = norm.ppf((R - 0.5) / n)
    if mode == "scaled":
        _, s, _ = _robust_col_scale(X)
        Z = Z * s
    return Z, None, _info(X, Z)


def drop_names(X, ctx, c=1.0):
    """Remove from the panel every name with a daily return above +c (or below -c)."""
    keep = ~(np.abs(X) > c).any(axis=0)
    Xt = X[:, keep]
    return Xt, keep, _info(X, Xt, keep)


def reversal_filter(X, ctx, up=1.0, rev=0.20, then_clip=None):
    """Spike-and-reversal filter.

    For consecutive trading days (t-1, t) of a name, with the day before the window as
    the first t-1, flag both returns when either exceeds +up and the two-day compound
    return (1 + r_{t-1})(1 + r_t) - 1 is below rev.  Flagged returns inside the window are
    replaced by that day's cross-sectional median.  Only days up to the formation date
    are used, so a spike on the last day of the window is never flagged.
    then_clip: optional fixed clip applied afterwards.
    """
    R = np.vstack([np.asarray(ctx["prior_ret"], float)[None, :], X])
    a, b = R[:-1], R[1:]
    with np.errstate(invalid="ignore"):
        pair = ((a > up) | (b > up)) & ((1 + a) * (1 + b) - 1 < rev)
    pair = np.nan_to_num(pair, nan=0).astype(bool)          # pair[t] couples day t-1 and day t
    flag = pair.copy()                                       # day t (in window) is flagged ...
    flag[:-1] |= pair[1:]                                    # ... or as the first leg of (t, t+1)
    med = np.median(X, axis=1, keepdims=True)
    Xt = np.where(flag, np.broadcast_to(med, X.shape), X)
    if then_clip is not None:
        Xt = np.clip(Xt, -then_clip, then_clip)
    info = _info(X, Xt); info["pairs_flagged"] = int(pair.sum())
    return Xt, None, info


FUNCS = dict(raw=raw, fixed_clip=fixed_clip, daily_winsor=daily_winsor, xs_mad=xs_mad,
             ts_mad=ts_mad, log_returns=log_returns, rank_normal=rank_normal,
             drop_names=drop_names, reversal_filter=reversal_filter)

# experiment folder -> (treatment function, settings); 'main' is the setting the report
# is built on, the others are the sensitivity settings
METHODS = {
    "00_raw": ("raw", {"main": {}}),
    "01_fixed_clip": ("fixed_clip", {"main": {"c": 1.0}, "c025": {"c": 0.25},
                                     "c050": {"c": 0.5}, "c200": {"c": 2.0}}),
    "02_daily_winsorization": ("daily_winsor", {"main": {"q": 0.001}, "q005": {"q": 0.005},
                                                "q010": {"q": 0.01}}),
    "03_cross_sectional_mad": ("xs_mad", {"main": {"c": 10.0}, "c05": {"c": 5.0},
                                          "c20": {"c": 20.0}}),
    "04_time_series_mad": ("ts_mad", {"main": {"c": 10.0}, "c05": {"c": 5.0},
                                      "c20": {"c": 20.0},
                                      "prior_c10": {"c": 10.0, "scale": "prior"}}),
    "05_log_returns": ("log_returns", {"main": {}}),
    "06_rank_transform": ("rank_normal", {"main": {"mode": "scaled"}, "unit": {"mode": "unit"},
                                          "standard": {"mode": "standard"}}),
    "07_drop_names": ("drop_names", {"main": {"c": 1.0}, "c050": {"c": 0.5},
                                     "c025": {"c": 0.25}}),
    "08_reversal_filter": ("reversal_filter", {"main": {"up": 1.0, "rev": 0.2},
                                               "up050": {"up": 0.5, "rev": 0.2},
                                               "then_clip": {"up": 1.0, "rev": 0.2,
                                                             "then_clip": 1.0}}),
}


def apply(method, setting, X, ctx):
    fname, settings = METHODS[method]
    return FUNCS[fname](X, ctx, **settings[setting])
