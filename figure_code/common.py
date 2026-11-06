"""Shared helpers for the figure scripts in figure_code/.

Paths follow the package convention: this folder sits at the project root next to
emp/, companion/ and wpca/.  Data are read from emp/data (via companion/code/paths.py),
stored results from companion/out and wpca/out.  Aggregated outputs (never
security-level rows) are written to figure_code/out/.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "companion" / "code"))
COMP_OUT = ROOT / "companion" / "out"
WPCA_OUT = ROOT / "wpca" / "out"
EMP_OUT = ROOT / "emp" / "output"
OUT = Path(os.environ.get("HLHH_FIG_OUT", ROOT / "figure_code" / "out"))
FIG_PAPER = Path(os.environ.get("HLHH_PAPER_FIGS", ROOT / "paper" / "figures"))
FIG_RECORD = Path(os.environ.get("HLHH_RECORD_FIGS", ROOT / "figure_code" / "record" / "figures"))
for d in (OUT, FIG_PAPER, FIG_RECORD):
    d.mkdir(parents=True, exist_ok=True)

P_GRID = (500, 1000, 2000, 3000)
N_OBS = 63

# one colour per direction, used in every figure
C1, C2, C3 = "#1f5a96", "#d9822b", "#3a8f5b"
DIR_COLORS = {1: C1, 2: C2, 3: C3}
GREY = "#6b6b6b"
CRISES = [("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-06-30")]


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
        "figure.dpi": 150, "lines.linewidth": 1.2,
    })


def deg(a):
    """Angle in degrees whose squared sine is a."""
    return np.degrees(np.arcsin(np.sqrt(np.clip(np.asarray(a, float), 0, 1))))


def shade_crises(ax):
    import pandas as pd
    for a, b in CRISES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="0.88", zorder=0, lw=0)


def save(fig, name, where="paper"):
    d = FIG_PAPER if where == "paper" else FIG_RECORD
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=200)


def dump(obj, name):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)
