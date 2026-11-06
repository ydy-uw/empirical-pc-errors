# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""Per-panel diagnostics, computed identically for every treatment.

For a treated panel Xt (63 x p, days x names) this returns one dictionary of
aggregates: the factor count under five rules, the Theorem 2 estimates A_j at the
treated panel's own count and at a reference count, the subspace number, the
pervasiveness screen of the first three sample directions, the bulk dispersion, and
the residual statistics that set the finite-p accuracy (split-sample dispersion of
residual variances and residual kurtosis).  No security identifiers are returned.
"""
from __future__ import annotations
import numpy as np
from oc_common import ALPHA_SKEW, KMAX
from jung_ref import jung_factor_count
from krules import bai_ng, ahn_horenstein, onatski_ed, mp_edge

LOC_PART, LOC_PEAK = 0.01, 0.5          # the pervasiveness screen


def skew_count(Xt):
    j = jung_factor_count(Xt, alpha=ALPHA_SKEW, max_factors=KMAX, center=True)
    return int(j.selected) if (j.selected is not None and not j.censored) else KMAX  # may be 0


def core(Xt, k_ref=None):
    N, p = Xt.shape
    n = N - 1
    Ac = Xt - Xt.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    ev = np.zeros(N); ev[:len(s)] = s ** 2 / (N * p)          # centred dual spectrum, all N
    th = ev[:n]                                              # the n non-degenerate ones
    k_rule = skew_count(Xt)
    k = max(k_rule, 1)            # the estimate needs at least one factor; zeros are recorded
    b1, b2 = bai_ng(ev, p, N, 20)
    rules = dict(k_skew=k, k_bn=int(b2), k_ah=int(ahn_horenstein(ev, 20)),
                 k_ed=int(onatski_ed(ev, 20)), k_mp=int(mp_edge(ev, p, N, 20)))
    ell = lambda m: float(th[m:].mean())
    d = dict(p=p, **rules, k_skew_rule=k_rule, theta1=th[0], theta2=th[1], theta3=th[2], ell=ell(k),
             bulk_disp=float(th[k:].std() / th[k:].mean()),
             relgap12=float((th[0] - th[1]) / th[0]), relgap23=float((th[1] - th[2]) / th[1]))
    for j in range(3):
        d[f"A{j+1}"] = ell(k) / th[j] if j < k else np.nan
    d["S"] = float(sum(ell(k) / th[j] for j in range(k)))
    d["S_over_k"] = d["S"] / k
    if k_ref is not None:
        d["k_ref"] = int(k_ref)
        for j in range(3):
            d[f"A{j+1}_ref"] = ell(k_ref) / th[j] if j < k_ref else np.nan
        d["S_ref"] = float(sum(ell(k_ref) / th[j] for j in range(k_ref)))
    for r, kk in rules.items():                              # A_1 at each rule's count
        d[f"A1_at_{r[2:]}"] = ell(max(kk, 1)) / th[0]
    H = Vt[:3].T
    part = 1.0 / (p * (H ** 4).sum(axis=0)); peak = np.abs(H).max(axis=0)
    for j in range(3):
        d[f"part{j+1}"] = float(part[j]); d[f"peak{j+1}"] = float(peak[j])
        d[f"loc{j+1}"] = bool(part[j] < LOC_PART or peak[j] > LOC_PEAK)
    # residual after the own count
    E = Ac - (U[:, :k] * s[:k]) @ Vt[:k]
    v = (E ** 2).mean(axis=0)
    vo, ve = (E[0::2] ** 2).mean(axis=0), (E[1::2] ** 2).mean(axis=0)
    mu = float(v.mean())
    d["resid_var_mean"] = mu
    d["cv_naive"] = float(v.std(ddof=1) / mu)
    d["cv_split"] = float(np.sqrt(max(float(np.cov(vo, ve)[0, 1]), 0.0)) / mu)
    Z = E / np.sqrt(np.where(v > 0, v, np.nan))
    kurt = np.nanmean(Z ** 4, axis=0) - 3.0
    d["kurt_median"] = float(np.nanmedian(kurt))
    d["kurt_pooled"] = float(np.nanmean(Z ** 4) - 3.0)
    d["n_above_100"] = int((Xt > 1.0).sum())
    d["max_abs"] = float(np.abs(Xt).max())
    return d
