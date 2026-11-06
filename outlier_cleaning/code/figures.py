"""Figures for the technical reports (outlier_cleaning/overleaf_reports/figures/).

    python outlier_cleaning/code/figures.py           # every method found, then the comparison

Per method <m>: figures/<m>/{changes,counts,estimates,screen,blocks}.pdf.  The untreated
panels are drawn in grey (dashed) in every figure, the treated ones in colour.
Comparison: figures/comparison/*.pdf.  Everything is drawn from the aggregate files in
experiments/*/results/.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from oc_common import EXP, STUDY, P_GRID, CRISES, deg
from treatments import METHODS

FIG = STUDY / "overleaf_reports" / "figures"
C = {1: "#1f5a96", 2: "#d9822b", 3: "#3a8f5b"}
GREY = "#8a8a8a"
TREAT = "#1f5a96"
LABEL = {"00_raw": "untreated", "01_fixed_clip": "fixed clip", "02_daily_winsorization": "daily winsorization",
         "03_cross_sectional_mad": "cross-sectional clip", "04_time_series_mad": "time-series clip",
         "05_log_returns": "log returns", "06_rank_transform": "rank transform",
         "07_drop_names": "drop names", "08_reversal_filter": "reversal filter"}


def style():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "lines.linewidth": 1.1})


def shade(ax):
    for a, b in CRISES.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="0.9", zorder=0, lw=0)


def load(m, s="main"):
    return pd.read_csv(EXP / m / "results" / f"panels_{s}.csv", parse_dates=["formation_date"])


def save(fig, m, name):
    d = FIG / m; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf"); plt.close(fig)


def fig_changes(m, t, r):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
    fname = METHODS[m][0]
    for P, ls in ((500, "-"), (3000, "-")):
        g = t[t.n_assets == P]
        if fname == "drop_names":
            y, lab = g.names_dropped, "names removed per panel"
        elif fname == "log_returns":
            y, lab = 100 * g.n_changed_1pt / (63 * g.p), "moved >1 point (%)"
        else:
            y, lab = 100 * g.share_changed, "entries changed (%)"
        ax[0].plot(g.formation_date, y, lw=0.9, color=C[1] if P == 500 else C[2], label=f"$P={P:,}$")
    ax[0].set_ylabel(lab); ax[0].legend(frameon=False); shade(ax[0])
    ax[0].set_title("(a) what the treatment changes", loc="left")
    mm = t.merge(r[["formation_date", "n_assets", "A1", "loc1"]], on=["formation_date", "n_assets"], suffixes=("", "_raw"))
    g = mm[mm.n_assets == 3000]
    dd = deg(g.A1) - deg(g.A1_raw)
    ax[1].axhline(0, color="0.6", lw=0.6)
    ax[1].plot(g.formation_date, dd, color=TREAT, lw=0.9)
    loc = g.loc1_raw.astype(bool)
    ax[1].scatter(g.formation_date[loc], dd[loc], s=9, color=C[2], zorder=3, label="leading direction localized (untreated)")
    ax[1].set_ylabel(r"change in leading angle (degrees)"); shade(ax[1])
    ax[1].set_title(r"(b) $P=3{,}000$: treated minus untreated", loc="left")
    if loc.any():
        ax[1].legend(frameon=False, loc="upper center", fontsize=6.5)
    save(fig, m, "changes")


def fig_counts(m, t, r):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
    x = np.arange(len(P_GRID))
    pairs = ((t, TREAT, "-", "untreated"),) if m == "00_raw" else ((r, GREY, "--", "untreated"), (t, TREAT, "-", "treated"))
    for df, col, ls, lab in pairs:
        for rule, mk in (("skew", "o"), ("bn", "s")):
            med = [df[df.n_assets == P][f"k_{rule}"].median() for P in P_GRID]
            lo = [df[df.n_assets == P][f"k_{rule}"].quantile(.25) for P in P_GRID]
            hi = [df[df.n_assets == P][f"k_{rule}"].quantile(.75) for P in P_GRID]
            off = 0 if m == "00_raw" else (-0.06 if df is r else 0.06)
            ax[0].errorbar(x + off, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], color=col, ls=ls,
                           marker=mk, ms=3.5, capsize=2, lw=1,
                           label=f"{'skewness' if rule == 'skew' else 'Bai–Ng'}, {lab}")
    ax[0].set_xticks(x, [f"{P:,}" for P in P_GRID]); ax[0].set_xlabel("$P$"); ax[0].set_ylabel("count (median, IQR)")
    top = max(t.k_skew.quantile(.75), r.k_skew.quantile(.75))
    ax[0].set_ylim(0, 18.5)
    ax[0].legend(frameon=False, ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.55, 1.0),
                 handletextpad=0.3, columnspacing=0.8)
    ax[0].set_title("(a) count against universe size", loc="left")
    series = ((t, TREAT, 1.0, "untreated"),) if m == "00_raw" else ((r, GREY, 0.9, "untreated"), (t, TREAT, 1.0, "treated"))
    for df, col, lw, lab in series:
        g = df[df.n_assets == 3000]
        ax[1].step(g.formation_date, g.k_skew, where="mid", color=col, lw=lw, label=lab)
    ax[1].yaxis.get_major_locator().set_params(integer=True)
    shade(ax[1]); ax[1].set_ylabel("skewness count"); ax[1].legend(frameon=False)
    ax[1].set_title(r"(b) $P=3{,}000$, month by month", loc="left")
    save(fig, m, "counts")


def fig_estimates(m, t, r):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
    x = np.arange(len(P_GRID))
    both = ((t, "-", "o"),) if m == "00_raw" else ((r, "--", None), (t, "-", "o"))
    for j in (1, 2, 3):
        for df, ls, mk in both:
            y = [deg(df[(df.n_assets == P) & (df.k_skew >= j)][f"A{j}"].median()) for P in P_GRID]
            ax[0].plot(x, y, ls=ls, marker=mk, ms=3, color=C[j], alpha=0.55 if df is r else 1,
                       label=f"$j={j}$" if df is t else None)
    ax[0].set_xticks(x, [f"{P:,}" for P in P_GRID]); ax[0].set_xlabel("$P$")
    ax[0].set_ylabel("median angle (degrees)")
    ax[0].legend(frameon=False, title=None if m == "00_raw" else "solid: treated", title_fontsize=7)
    ax[0].set_title("(a) levels by universe size", loc="left")
    series = ((t, TREAT, 1.0, "untreated"),) if m == "00_raw" else ((r, GREY, 0.9, "untreated"), (t, TREAT, 1.0, "treated"))
    for df, col, lw, lab in series:
        g = df[df.n_assets == 3000]
        ax[1].plot(g.formation_date, deg(g.A1), color=col, lw=lw, label=lab)
    shade(ax[1]); ax[1].set_ylabel("leading angle (degrees)"); ax[1].legend(frameon=False)
    ax[1].set_title(r"(b) $P=3{,}000$, direction 1", loc="left")
    save(fig, m, "estimates")


def fig_screen(m, t, r):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
    x = np.arange(len(P_GRID))
    both = ((t, "-", "o"),) if m == "00_raw" else ((r, "--", None), (t, "-", "o"))
    for j in (1, 2, 3):
        for df, ls, mk in both:
            y = []
            for P in P_GRID:
                g = df[df.n_assets == P]
                kr = r[r.n_assets == P].set_index("formation_date").k_skew.reindex(g.formation_date).to_numpy()
                h = g[kr >= j]
                y.append(100 * (1 - h[f"loc{j}"].mean()))
            ax[0].plot(x, y, ls=ls, marker=mk, ms=3, color=C[j], alpha=0.55 if df is r else 1,
                       label=f"$j={j}$" if df is t else None)
    ax[0].set_xticks(x, [f"{P:,}" for P in P_GRID]); ax[0].set_xlabel("$P$"); ax[0].set_ylim(0, 102)
    ax[0].set_ylabel("months passing the screen (%)")
    ax[0].legend(frameon=False, title=None if m == "00_raw" else "solid: treated", title_fontsize=7, loc="lower left")
    ax[0].set_title("(a) pass share by universe size", loc="left")
    yrs = sorted(r.formation_date.dt.year.unique())
    series = ((t, "untreated"),) if m == "00_raw" else ((r, "untreated"), (t, "treated"))
    w = 0.8 / (2 * len(series))
    for i, (df, lab) in enumerate(series):
        for jj, j in enumerate((2, 3)):
            g = df[df.n_assets == 3000]
            cnt = g.groupby(g.formation_date.dt.year)[f"loc{j}"].sum().reindex(yrs, fill_value=0)
            pos = i * 2 + jj - (2 * len(series) - 1) / 2
            ax[1].bar(np.arange(len(yrs)) + pos * w, cnt, width=w, color=C[j],
                      alpha=0.35 if (df is r and m != "00_raw") else 1, label=f"$j={j}$, {lab}")
    ax[1].set_xticks(np.arange(len(yrs))[::3], [str(y) for y in yrs][::3])
    ax[1].set_ylabel("localized months per year"); ax[1].set_ylim(0, 14.5)
    ax[1].legend(frameon=False, ncol=2, fontsize=6.5, loc="upper left")
    ax[1].set_title(r"(b) $P=3{,}000$, directions 2 and 3", loc="left")
    save(fig, m, "screen")


def fig_blocks(m):
    fb = EXP / m / "results" / "blocks.csv"
    if not fb.is_file():
        return
    b = pd.read_csv(fb); br = pd.read_csv(EXP / "00_raw" / "results" / "blocks.csv")
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for a, P in zip(ax, (500, 3000)):
        pairs = ((b, TREAT, "-", "untreated"),) if m == "00_raw" else ((br, GREY, "--", "untreated"), (b, TREAT, "-", "treated"))
        for df, col, ls, lab in pairs:
            g = df[(df.n_assets == P) & df.m.isin([1, 3, 5, 8, 12, 20, 40])].drop_duplicates(["formation_date", "m"])
            med = g.groupby("m").d1.median() * 100
            q1 = g.groupby("m").d1.quantile(.25) * 100; q3 = g.groupby("m").d1.quantile(.75) * 100
            a.plot(med.index, med, color=col, ls=ls, marker="o", ms=3, label=lab)
            a.fill_between(med.index, q1, q3, color=col, alpha=0.15, lw=0)
        a.set_xscale("log"); a.set_xticks([1, 3, 5, 8, 12, 20, 40], ["1", "3", "5", "8", "12", "20", "40"])
        a.set_xlabel("components removed $m$"); a.set_title(f"$P={P:,}$", loc="left")
    ax[0].set_ylabel("decile-1 residual block (%)"); ax[0].legend(frameon=False)
    save(fig, m, "blocks")


# ------------------------------------------------------------------ comparison figures
ORDER = list(METHODS)


def summ(m, s="main"):
    f = EXP / m / "results" / f"summary_{s}.json"
    return json.load(open(f)) if f.is_file() else None


def fig_comparison():
    ms = [m for m in ORDER if summ(m)]
    S = {m: summ(m) for m in ms}
    y = np.arange(len(ms))[::-1]
    names = [LABEL[m] for m in ms]
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
    for P, mk, col in ((500, "o", C[1]), (3000, "s", C[2])):
        med = [S[m]["counts"][str(P)]["skew"]["median"] for m in ms]
        lo = [S[m]["counts"][str(P)]["skew"]["q25"] for m in ms]
        hi = [S[m]["counts"][str(P)]["skew"]["q75"] for m in ms]
        off = 0.12 if P == 500 else -0.12
        ax[0].errorbar(med, y + off, xerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt=mk, ms=3.5,
                       color=col, capsize=2, lw=0.9, label=f"$P={P:,}$")
    ax[0].set_yticks(y, names); ax[0].set_xlabel("median and IQR")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
                 handletextpad=0.2, columnspacing=0.8)
    ax[0].set_title("(a) skewness count", loc="left")
    for j in (1, 2, 3):
        v = [S[m]["estimates"]["3000"][str(j)]["median_deg"] for m in ms]
        ax[1].plot(v, y, "o", ms=3.5, color=C[j], label=f"$j={j}$")
        ax[1].axvline(S["00_raw"]["estimates"]["3000"][str(j)]["median_deg"], color=C[j], lw=0.5, ls=":")
    ax[1].set_xlabel("degrees"); ax[1].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3,
                                               bbox_to_anchor=(0.5, 1.0), handletextpad=0.1, columnspacing=0.6)
    ax[1].set_ylim(-0.8, len(ms) + 0.2)
    ax[1].set_title("(b) median angle, $P=3{,}000$", loc="left")
    for j in (1, 2, 3):
        v = [100 * S[m]["screen"]["3000"][str(j)]["pass_share"] for m in ms]
        ax[2].plot(v, y, "o", ms=3.5, color=C[j], label=f"$j={j}$")
    ax[2].set_xlabel("% of months")
    ax[2].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0),
                 handletextpad=0.1, columnspacing=0.6)
    ax[2].set_ylim(-0.8, len(ms) + 0.2)
    ax[2].set_title("(c) passing the screen, $P=3{,}000$", loc="left")
    for a in ax:
        a.grid(axis="y", color="0.92", lw=0.5)
    d = FIG / "comparison"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "main_results.pdf"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
    for P, mk, col in ((500, "o", C[1]), (3000, "s", C[2])):
        ax[0].plot([S[m]["accuracy"][str(P)]["p_eff"] for m in ms], y, mk, ms=3.5, color=col, label=f"$P={P:,}$")
        ax[1].plot([S[m]["accuracy"][str(P)]["kurt_pooled"] for m in ms], y, mk, ms=3.5, color=col)
        if "blocks" in S[ms[0]]:
            ax[2].plot([S[m]["blocks"][str(P)]["own_k"] if "blocks" in S[m] else np.nan for m in ms], y, mk,
                       ms=3.5, color=col)
    ax[0].set_yticks(y, names); ax[0].set_xlabel(r"$p_{\mathrm{eff}}$")
    ax[0].legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    for a in ax:
        a.set_ylim(-0.6, len(ms) + 0.4)
    ax[0].set_title("(a) effective cross-section", loc="left")
    ax[1].set_xlabel("pooled excess kurtosis"); ax[1].set_title("(b) residual tails", loc="left")
    ax[2].set_xlabel("% at the own count"); ax[2].set_title("(c) decile-1 residual block", loc="left")
    for a in ax:
        a.grid(axis="y", color="0.92", lw=0.5)
    fig.savefig(d / "accuracy.pdf"); plt.close(fig)

    # dose-response over every setting
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    mk = dict(zip(ORDER, "os^vDP*Xh"))
    raw = S["00_raw"]
    ax[0].axhline(100 * raw["screen"]["3000"]["3"]["pass_share"], color=GREY, ls=":", lw=0.8)
    ax[1].axhline(raw["estimates"]["3000"]["3"]["median_deg"], color=GREY, ls=":", lw=0.8)
    for a, yv, t in ((ax[0], 100 * raw["screen"]["3000"]["3"]["pass_share"], "untreated"),
                     (ax[1], raw["estimates"]["3000"]["3"]["median_deg"], "untreated")):
        a.text(60, yv + 0.3 if a is ax[1] else yv + 1.5, t, color=GREY, fontsize=6.5, ha="right")
    for m in ms:
        if m == "00_raw":
            continue
        for f in sorted((EXP / m / "results").glob("summary_*.json")):
            s = json.load(open(f)); st = f.stem.replace("summary_", "")
            share = 100 * s["touched"]["3000"]["entries_changed"] / (239 * 63 * 3000)
            if METHODS[m][0] in ("log_returns", "rank_normal", "drop_names"):
                share = 100 * s["touched"]["3000"]["share_changed_mean"]
            kw = dict(marker=mk[m], s=22 if st == "main" else 12, facecolors=C[2] if st == "main" else "none",
                      edgecolors=C[2], lw=0.8)
            ax[0].scatter(share, 100 * s["screen"]["3000"]["3"]["pass_share"], **kw)
            ax[1].scatter(share, s["estimates"]["3000"]["3"]["median_deg"], **kw)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("entries changed, $P=3{,}000$ (%)")
    ax[0].set_ylabel("direction 3 passing the screen (%)"); ax[1].set_ylabel("median third angle (degrees)")
    handles = [plt.Line2D([], [], marker=mk[m], ls="", color=C[2], label=LABEL[m]) for m in ms if m != "00_raw"]
    ax[1].legend(handles=handles, frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax[0].set_title("(a) pervasiveness", loc="left"); ax[1].set_title("(b) estimate", loc="left")
    fig.savefig(d / "dose_response.pdf"); plt.close(fig)


def main():
    style()
    r = load("00_raw")
    for m in METHODS:
        if not (EXP / m / "results" / "panels_main.csv").is_file():
            continue
        t = load(m)
        if m != "00_raw":
            fig_changes(m, t, r)
        fig_counts(m, t, r); fig_estimates(m, t, r); fig_screen(m, t, r); fig_blocks(m)
        print("figures", m)
    fig_comparison()


if __name__ == "__main__":
    main()
