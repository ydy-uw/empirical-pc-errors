"""Rebuild formation panels from raw CRSP returns and the frozen universes."""
import glob, functools
import numpy as np, pandas as pd

import paths

D = str(paths.CRSP_PARQUET)   # year=YYYY/daily.parquet, from src.prepare_crsp_parquet
U = str(paths.ROLLING)        # year=YYYY/selected_universes.parquet, from src.run_rolling_experiments
N_OBS = 63

@functools.lru_cache(maxsize=1)
def returns():
    fs = paths.year_files(D, "daily.parquet")
    d = pd.concat([pd.read_parquet(f, columns=["permno", "dlycaldt", "dlyret"]) for f in fs],
                  ignore_index=True)
    d["dlycaldt"] = pd.to_datetime(d["dlycaldt"])
    d = d.dropna(subset=["dlyret"])
    return d.set_index(["dlycaldt", "permno"]).dlyret.sort_index()

@functools.lru_cache(maxsize=1)
def calendar():
    return np.array(sorted(returns().index.get_level_values(0).unique()))

@functools.lru_cache(maxsize=1)
def universes():
    fs = paths.year_files(U, "selected_universes.parquet")
    d = pd.concat([pd.read_parquet(f, columns=["formation_date", "n_assets", "rank", "permno"])
                   for f in fs], ignore_index=True)
    d["formation_date"] = pd.to_datetime(d["formation_date"])
    return d

@functools.lru_cache(maxsize=1)
def wide():
    r = returns().unstack("permno")
    return r

def window_dates(formation, n=N_OBS):
    cal = calendar()
    i = int(np.searchsorted(cal, np.datetime64(formation)))
    if cal[i] != np.datetime64(formation) or i + 1 < n:
        return None
    return cal[i + 1 - n: i + 1]

def panel(formation, n_assets, n=N_OBS, permnos=None):
    """Return the n x P formation matrix (observations x securities), or None."""
    if permnos is None:
        u = universes()
        sel = u[(u.formation_date == formation) & (u.n_assets == n_assets)]
        if sel.empty: return None, None
        permnos = sel.sort_values("rank").permno.to_numpy()
    dts = window_dates(formation, n)
    if dts is None: return None, None
    W = wide()
    cols = [c for c in permnos if c in W.columns]
    X = W.loc[pd.DatetimeIndex(dts), cols]
    return X, np.array(cols)

def spectra(X, kmax=30):
    """theta_j and the bulk average for the uncentered and centered dual matrices."""
    A = X.to_numpy(dtype=float)
    n, p = A.shape
    out = {}
    for tag, M in (("u", A), ("c", A - A.mean(axis=0, keepdims=True))):
        W = M @ M.T / (n * p)
        ev = np.linalg.eigvalsh(W)[::-1]
        out[tag] = ev
    return out
