"""Shared settings for the outlier-treatment study (outlier_cleaning/).

The study reads the licensed CRSP extract and the frozen universes through
companion/code/rawpanel.py and never writes to emp/data.  Each treatment writes
only aggregates (one row per panel, no security identifiers) to its own folder
under outlier_cleaning/experiments/.
"""
from __future__ import annotations
import os, sys, json, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent                                  # outlier_cleaning/
ROOT = STUDY.parent                                  # project root
sys.path.insert(0, str(ROOT / "companion" / "code"))
sys.path.insert(0, str(ROOT / "figure_code"))
EXP = STUDY / "experiments"
CMP = STUDY / "comparison"
# local cache of the formation panels (licensed data: stays on this machine, never shared)
CACHE = Path(os.environ.get("HLHH_PANEL_CACHE", STUDY / "_cache"))

P_GRID = (500, 1000, 2000, 3000)
N_OBS = 63
ALPHA_SKEW = 0.10
KMAX = 30
CRISES = {"gfc": ("2008-09-01", "2009-06-30"), "covid": ("2020-02-01", "2020-06-30")}


def deg(a):
    """Angle in degrees whose squared sine is a."""
    return np.degrees(np.arcsin(np.sqrt(np.clip(np.asarray(a, float), 0, 1))))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
