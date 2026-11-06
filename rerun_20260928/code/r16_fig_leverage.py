"""Figure: day variance against leverage, raw and leverage-corrected (Section 6.1).

Reads the binned profiles written by r15_leverage_check.py (aggregates only) and draws
<package>/figures/fig_leverage.pdf: (a) panels simulated with a constant day variance (one per month),
(b) the data; in each, the median over date-months of the raw mean squared residual and of the
day variance, each relative to its window mean, against the date's leverage sum_{j<=k} U_tj^2 (k = 3).
Panel (a) also shows the limit of the raw mean squared residual under a constant day variance,
(1 - 1/N - l) / (1 - (1 + k)/N).  Panel (b) also shows the day variance of panels simulated with the
month's day variances shuffled over the days (first such panel of each month; alt_profiles.csv of
r18_dayvar_leverage_alt.py): the rise with the estimated leverage that estimation alone produces.
Run from the project root:  emp/.venv/Scripts/python.exe rerun_20260928/code/r16_fig_leverage.py paper/overleaf_upload_20261011
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rerun_20260928" / "work" / "figure_code" / "out" / "checks_20261011" / "lev_profiles.csv"
ALT = SRC.parent / "alt_profiles.csv"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "overleaf_upload_20261011"
N, K = 63, 3
C = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, INK2, GRID, RAWC = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8986"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def main():
    p = pd.read_csv(SRC)
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.45), sharey=True)
    for ax, src, title in ((axs[0], "sim", "(a) simulated, constant day variance"), (axs[1], "data", "(b) data")):
        g = p[p.source == src].sort_values("lev_median")
        ax.plot(g.lev_median, g.g_rel_median, "--o", ms=3, color=RAWC, lw=1.2, label="raw mean squared residual")
        ax.plot(g.lev_median, g.d_rel_median, "-o", ms=3, color=C[1], label="day variance")
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
        if src == "sim":
            x = np.linspace(0.0, g.lev_median.max(), 100)
            ax.plot(x, (1 - 1 / N - x) / (1 - (1 + K) / N), color=INK, lw=0.8, ls=(0, (1, 2)),
                    label="limit of the raw mean squared residual")
        if src == "data":
            a = pd.read_csv(ALT).sort_values("lev_median")
            ax.plot(a.lev_median, a.d_rel_median, "-.s", ms=2.6, color=C[2], lw=1.2,
                    label="day variance, simulated with the day variances shuffled over the days")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("leverage of the date")
    axs[0].set_ylabel("relative to window mean")
    h0, l0 = axs[0].get_legend_handles_labels()
    h1, l1 = axs[1].get_legend_handles_labels()
    fig.legend(h0 + h1[2:], l0 + l1[2:], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    (DST / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(DST / "figures" / "fig_leverage.pdf", bbox_inches="tight")
    fig.savefig(SRC.parent / "fig_leverage_preview.png", bbox_inches="tight", dpi=130)
    print("wrote", DST / "figures" / "fig_leverage.pdf")


if __name__ == "__main__":
    main()
