"""Formation panels for the outlier-treatment study.

Each panel is the 63 x P matrix of daily simple returns of the frozen top-P universe
over the 63 trading days that end on the formation date, built exactly as
companion/code/rawpanel.py builds it (columns in rank order, i.e. by capitalization).
Two pieces of context are attached for the treatments that need them:

  prior_scale  robust scale 1.4826 * MAD of each name's returns over the 63 trading
               days before the window (NaN when fewer than 40 of them are observed);
  prior_ret    each name's return on the trading day before the window (NaN if missing).

The panels are read from the original extract, which is opened read-only.  To save
time across runs they can be cached with `python panels.py build`; the cache holds
licensed returns and security identifiers, lives in outlier_cleaning/_cache (or
$HLHH_PANEL_CACHE), and must never be shared or copied into an experiment folder.
Without a cache every run rebuilds the panels from the extract (a few minutes).
"""
from __future__ import annotations
import sys, pickle, time
import numpy as np, pandas as pd
from oc_common import CACHE, P_GRID, N_OBS
import rawpanel

MIN_PRIOR = 40


def _robust_scale(M):
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0)
    cnt = np.isfinite(M).sum(axis=0)
    s = 1.4826 * mad
    s[cnt < MIN_PRIOR] = np.nan
    return s


def _build_P(P):
    U = rawpanel.universes(); W = rawpanel.wide(); cal = rawpanel.calendar()
    out = {}
    for fd, sel in U[U.n_assets == P].groupby("formation_date"):
        permnos = sel.sort_values("rank").permno.to_numpy()
        dts = rawpanel.window_dates(fd, N_OBS)
        i0 = int(np.searchsorted(cal, dts[0]))
        X = W.loc[pd.DatetimeIndex(dts), permnos].to_numpy(float)
        if np.isnan(X).any():
            raise RuntimeError(f"missing returns in {fd:%Y-%m} P={P}")
        prior = W.reindex(index=pd.DatetimeIndex(cal[max(0, i0 - N_OBS):i0]), columns=permnos).to_numpy(float)
        out[pd.Timestamp(fd)] = dict(X=X, permnos=permnos.astype(np.int64),
                                     prior_scale=_robust_scale(prior),
                                     prior_ret=prior[-1].copy() if len(prior) else np.full(P, np.nan),
                                     dates=np.asarray(dts))
    return out


def build():
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "README.txt").write_text(
        "Local cache of formation panels built from the licensed CRSP extract.\n"
        "Contains returns and security identifiers. Do not share or copy it.\n"
        "Delete it at any time; panels.py rebuilds it from emp/data.\n")
    for P in P_GRID:
        t0 = time.time()
        d = _build_P(P)
        with open(CACHE / f"panels_P{P}.pkl", "wb") as f:
            pickle.dump(d, f, protocol=4)
        print(f"P={P}: {len(d)} panels cached in {time.time()-t0:.0f}s", flush=True)


_MEM = {}


def load(P):
    """dict formation_date -> panel dict, from the cache if present."""
    if P in _MEM:
        return _MEM[P]
    f = CACHE / f"panels_P{P}.pkl"
    if f.is_file():
        with open(f, "rb") as fh:
            _MEM[P] = pickle.load(fh)
    else:
        _MEM[P] = _build_P(P)
    return _MEM[P]


def drop(P):
    _MEM.pop(P, None)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
