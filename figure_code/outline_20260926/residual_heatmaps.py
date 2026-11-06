"""Residual correlation heat maps for single months (Section 4 of the 26 September outline).

For four formation months at P = 3,000 the clipped (+/-50%), demeaned 63 x 3,000 panel is
reduced by its rank-q PCA fit, q = 2 and q = the month's skewness count, and the full
3,000 x 3,000 matrix of residual correlations is drawn with the stocks in order of
capitalization at formation (largest first).  Nothing is grouped by any characteristic
and nothing is averaged over months.  For display, each pixel is the mean over a 5 x 5
square of stock pairs (600 x 600 pixels per panel), diagonal included (each pixel on it
averages five self-correlations of 1 with twenty correlations between different stocks); at the
resolution of single pairs the sampling noise of a 63-day correlation (s.d. about 0.13)
hides the pattern.

Months, chosen by rule from aggregate outputs:
  typical   in each half of the sample, the month closest to the medians over months of
            the skewness count, the leading angle (skewness count) and the mean residual
            correlation among the largest 10% of stocks after two components
            (sum of absolute standardized deviations);
  unusual   the month with the highest mean residual correlation among the largest 10%
            after two components, and the month with the highest skewness count.

Inputs: the licensed CRSP extract through companion/code/rawpanel.py (read-only);
outlier_cleaning/experiments/01_fixed_clip/results/panels_c050.csv and
figure_code/out/outline_20260926/breadth_resid.csv (aggregates).
Outputs: <outline>/figures/fig_residual_heatmaps.pdf and
figure_code/out/outline_20260926/residual_heatmaps.json (per-panel summary numbers only).
Run from the project root:
  OPENBLAS_NUM_THREADS=1 python figure_code/outline_20260926/residual_heatmaps.py [outline_dir]
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "companion" / "code"))
import rawpanel  # noqa: E402

OUT = ROOT / "figure_code" / "out" / "outline_20260926"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "overleaf_upload_20260926"
CLP = ROOT / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results"
P, N_OBS, CLIP, BIN, TOP = 3000, 63, 0.5, 5, 300
VMAX = 0.25

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def choose_months():
    c = pd.read_csv(CLP / "panels_c050.csv", parse_dates=["formation_date"])
    c = c[c.p == P][["formation_date", "k_skew", "A1"]]
    r = pd.read_csv(OUT / "breadth_resid.csv", parse_dates=["formation_date"])
    m = c.merge(r[["formation_date", "top_corr"]], on="formation_date").reset_index(drop=True)
    m["ang1"] = np.degrees(np.arcsin(np.sqrt(m.A1)))
    cols = ["k_skew", "ang1", "top_corr"]
    m["dist"] = sum((m[v] - m[v].median()).abs() / m[v].std() for v in cols)
    half = len(m) // 2
    typical = [m.iloc[:half].sort_values("dist").iloc[0], m.iloc[half:].sort_values("dist").iloc[0]]
    unusual = [m.loc[m.top_corr.idxmax()], m.loc[m.k_skew.idxmax()]]
    ten = m.nlargest(10, "top_corr").formation_date
    run = dict(ten_highest_top_corr_first=str(ten.min().date()), ten_highest_top_corr_last=str(ten.max().date()),
               max_k_skew=int(m.k_skew.max()))
    return [(t.formation_date, int(t.k_skew), "typical") for t in typical] + \
           [(u.formation_date, int(u.k_skew), "unusual") for u in unusual], run


def residual_corr(fd, q, U, W):
    sel = U[(U.n_assets == P) & (U.formation_date == fd)].sort_values("rank")
    X = W.loc[pd.DatetimeIndex(rawpanel.window_dates(fd, N_OBS)), sel.permno.to_numpy()].to_numpy(float)
    Y = np.clip(X, -CLIP, CLIP)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Uf, s, Vt = np.linalg.svd(Yc, full_matrices=False)
    E = Yc - (Uf[:, :q] * s[:q]) @ Vt[:q]
    Z = E / np.sqrt((E ** 2).sum(axis=0))
    return Z.T @ Z


def main():
    (DST / "figures").mkdir(parents=True, exist_ok=True)
    months, run = choose_months()
    U, W = rawpanel.universes(), rawpanel.wide()
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.95))
    info = []
    for c, (fd, k, kind) in enumerate(months):
        for r, q in enumerate((2, k)):
            R = residual_corr(fd, q, U, W)
            img = R.reshape(P // BIN, BIN, P // BIN, BIN).mean(axis=(1, 3))   # diagonal included
            np.fill_diagonal(R, np.nan)                                        # summary numbers: pairs only
            top = float(np.nanmean(R[:TOP, :TOP]))
            dpx = np.diag(img)
            ax = axes[r, c]
            im = ax.imshow(img, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, interpolation="none",
                           extent=(0.5, P + 0.5, P + 0.5, 0.5))
            ax.set_xticks([1, 1500, 3000]); ax.set_yticks([1, 1500, 3000])
            ax.set_xticklabels(["1", "1,500", "3,000"]); ax.set_yticklabels(["1", "1,500", "3,000"])
            if c > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_xticklabels([])
                ax.set_title(f"{fd:%B %Y}, {kind}\n$q=2$; largest 10%: {top:.2f}", color=INK)
            else:
                ax.set_title(f"$q=k={q}$\nlargest 10%: {top:.2f}", color=INK)
            info.append(dict(formation_date=str(fd.date()), kind=kind, q=q, k_skew=k,
                             top_decile_mean_corr=top, mean_offdiag_corr=float(np.nanmean(R)),
                             diag_pixel_min=float(dpx.min()), diag_pixel_median=float(np.median(dpx)),
                             diag_pixel_max=float(dpx.max())))
    axes[0, 0].set_ylabel("stock, by capitalization")
    axes[1, 0].set_ylabel("stock, by capitalization")
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes((0.945, 0.2, 0.012, 0.6))
    cb = fig.colorbar(im, cax=cax, ticks=[-0.2, -0.1, 0, 0.1, 0.2], extend="max")
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_edgecolor(INK2)
    cb.set_label("residual correlation", fontsize=7)
    fig.savefig(DST / "figures" / "fig_residual_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)
    (OUT / "residual_heatmaps.json").write_text(json.dumps(dict(panels=info, selection=run), indent=1))
    for d in info:
        print(d)
    print(run)


if __name__ == "__main__":
    main()
