"""Figures and quoted numbers for the Assumption 3 and Assumptions 5-6 sections of the 27 September
outline (overleaf_upload_20260927).  Reads aggregate outputs only:
  figure_code/out/outline_20260927/a3_core.csv, a3_sweep.csv   assumption3.py
  figure_code/out/outline_20260927/separation.csv              separation.py
Writes <outline>/figures/fig_dayvar.pdf, <outline>/figures/fig_separation.pdf,
<outline>/tables/assumptions.tex (macros) and figure_code/out/outline_20260927/exhibits_0927.json.
The exhibits of the 26 September outline are not touched (figure_code/outline_20260926/).
Run from the project root:  python figure_code/outline_20260927/make_exhibits_0927.py [outline_dir]
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure_code" / "out" / "outline_20260927"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "overleaf_upload_20260927"
FIG, TAB = DST / "figures", DST / "tables"
CRISES = [("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-06-30")]
TIE = 0.10
JW = {1: "One", 2: "Two", 3: "Three"}
NW = {21: "TwentyOne", 42: "FortyTwo", 63: "SixtyThree", 126: "OneTwoSix", 252: "TwoFiveTwo"}

C = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, INK2, GRID, RAWC = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8986"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def deg(a):
    return np.degrees(np.arcsin(np.sqrt(np.clip(np.asarray(a, float), 0, 1))))


def load(name):
    return pd.read_csv(OUT / name, parse_dates=["formation_date"])


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", color=INK)


def crisis_mask(idx):
    idx = pd.DatetimeIndex(idx)
    m = np.zeros(len(idx), bool)
    for a, b in CRISES:
        m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    M = {}

    def put(name, val):
        M[name] = val

    # ---------------------------------------------------------------- Assumption 3
    core, sweep = load("a3_core.csv"), load("a3_sweep.csv")
    cm = core.groupby(["formation_date", "source"]).mean(numeric_only=True)
    cv = cm.cv_dayvar.unstack()
    put("dvCvData", f"{cv['data'].median():.2f}"); put("dvCvFloor", f"{cv['const'].median():.2f}")
    put("dvCvSim", f"{cv['dayvar'].median():.2f}")
    put("dvShrink", f"{cm.xs('dayvar', level=1).shrink.median():.2f}")
    und, sim_err = {}, {}
    for j in (1, 2, 3):
        g = core[core.source.isin(["const", "dayvar"]) & core[f"A{j}"].notna()].copy()
        g["und"] = deg(g[f"OE{j}"]) - deg(g[f"A{j}"])
        g["faerr"] = deg(g[f"Afa{j}"]) - deg(g[f"OE{j}"])
        g["fagap"] = deg(g[f"Afa{j}"]) - deg(g[f"A{j}"])
        pm = g.groupby(["formation_date", "source"])[["und", "faerr", "fagap"]].mean()
        u = pm.und.unstack(); fe = pm.faerr.unstack()
        extra = u["dayvar"] - u["const"]
        und[j] = extra
        put(f"dvExtra{JW[j]}", f"{extra.median():.1f}")
        put(f"dvExtraCrisis{JW[j]}", f"{extra[crisis_mask(extra.index)].median():.1f}")
        put(f"dvExtraPos{JW[j]}", f"{100 * (extra > 0).mean():.0f}")
        put(f"dvUndConst{JW[j]}", f"{u['const'].median():.1f}")
        put(f"dvUndDayvar{JW[j]}", f"{u['dayvar'].median():.1f}")
        put(f"dvFaErr{JW[j]}", f"$\\num{{{fe['dayvar'].median():.1f}}}$")
        put(f"dvFaErrConst{JW[j]}", f"$\\num{{{fe['const'].median():.1f}}}$")
        put(f"dvFaErrAbs{JW[j]}", f"{abs(fe['dayvar'].median()):.1f}")
        put(f"dvFaErrConstAbs{JW[j]}", f"{abs(fe['const'].median()):.1f}")
        put(f"dvFaGapSim{JW[j]}", f"{pm.fagap.unstack()['dayvar'].median():.1f}")
        dd = cm.xs("data", level=1).dropna(subset=[f"A{j}"])
        put(f"dvFaGapData{JW[j]}", f"{np.median(deg(dd[f'Afa{j}']) - deg(dd[f'A{j}'])):.1f}")
        sim_err[j] = dict(extra=float(extra.median()), und_const=float(u["const"].median()),
                          und_dayvar=float(u["dayvar"].median()), fa_err=float(fe["dayvar"].median()))
    scln = json.load(open(ROOT / "outlier_cleaning" / "experiments" / "01_fixed_clip" / "results" / "summary_c050.json"))
    put("dvComoveShare", f"{100 * scln['fa']['3000']['comove_share']:.0f}")
    put("dvMonthsTwo", str(int(und[2].notna().sum()))); put("dvMonthsThree", str(int(und[3].notna().sum())))
    fa_nc = (core.fa_iter >= 1000).groupby(core.source).mean()
    put("dvFaNonconvData", f"{100 * fa_nc['data']:.0f}")
    put("dvFaNonconvSim", f"{100 * max(fa_nc['const'], fa_nc['dayvar']):.0f}")

    sm = sweep.groupby(["formation_date", "N", "source"]).mean(numeric_only=True)
    Ns = sorted(sweep.N.unique())
    cvN = sm.cv_dayvar.unstack().groupby(level=1)
    cv_med, cv_q25, cv_q75 = cvN.median(), cvN.quantile(0.25), cvN.quantile(0.75)
    dN = sm.xs("data", level=2)
    ang = {j: dN[f"A{j}"].groupby(level=1).apply(lambda x: float(np.median(deg(x)))) for j in (1, 2, 3)}
    extraN = {}
    for j in (1, 2, 3):
        g = sweep[sweep.source.isin(["const", "dayvar"])].copy()
        g["und"] = deg(g[f"OE{j}"]) - deg(g[f"A{j}"])
        u = g.groupby(["formation_date", "N", "source"]).und.mean().unstack()
        extraN[j] = (u["dayvar"] - u["const"]).groupby(level=1).median()
    for N in Ns:
        put(f"dvCv{NW[N]}", f"{cv_med.loc[N, 'data']:.2f}")
        for j in (1, 2):
            put(f"dvAng{JW[j]}{NW[N]}", f"{ang[j].loc[N]:.1f}")
            put(f"dvExtraN{JW[j]}{NW[N]}", f"{extraN[j].loc[N]:.1f}")
    put("dvFloorLo", f"{cv_med['const'].min():.2f}"); put("dvFloorHi", f"{cv_med['const'].max():.2f}")
    pk = dN.p.groupby(level=1).median()
    put("dvKeptOneTwoSix", f"{pk.loc[126]:,.0f}"); put("dvKeptTwoFiveTwo", f"{pk.loc[252]:,.0f}")
    put("dvKeptMinTwoFiveTwo", f"{dN.p.groupby(level=1).min().loc[252]:,.0f}")

    x = np.array(Ns, float)
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.35))
    ax[0].fill_between(x, cv_q25["data"], cv_q75["data"], color=C[1], alpha=0.15, lw=0)
    ax[0].plot(x, cv_med["data"], "-o", ms=3.5, color=C[1], label="clipped returns")
    ax[0].plot(x, cv_med["const"], "--s", ms=3, color=RAWC, lw=1.2, label="simulated, constant\nday variance")
    ax[0].set_ylim(0, None); ax[0].set_ylabel("coefficient of variation")
    ax[0].legend(loc="upper left", fontsize=6.3)
    panel_label(ax[0], "(a) Day variance")
    for j in (1, 2, 3):
        ax[1].plot(x, [ang[j].loc[N] for N in Ns], "-o", ms=3.5, color=C[j], label=f"direction {j}")
        ax[2].plot(x, [extraN[j].loc[N] for N in Ns], "-o", ms=3.5, color=C[j], label=f"direction {j}")
    ax[1].set_ylabel("angle to loading span (degrees)"); ax[1].set_ylim(0, None)
    ax[1].legend(loc="upper right", fontsize=6.3)
    panel_label(ax[1], "(b) Estimated angle")
    ax[2].axhline(0, color=INK2, lw=0.8)
    ax[2].set_ylabel("extra understatement (degrees)"); ax[2].set_ylim(-0.3, None)
    panel_label(ax[2], "(c) Cost of day variance")
    for a in ax:
        a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels([str(int(v)) for v in x]); a.minorticks_off()
        a.set_xlabel("window length (days)")
    fig.tight_layout(); fig.savefig(FIG / "fig_dayvar.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------------------------------------------------------------- Assumptions 5-6
    sep = load("separation.csv")
    d = cm.xs("data", level=1)
    gap23 = (d.theta2 - d.theta3) / d.theta2
    gap12 = (d.theta1 - d.theta2) / d.theta1
    gap34 = (d.theta3 - d.theta4) / d.theta3
    put("sepTieThreeFourAll", f"{100 * (gap34 < TIE).mean():.0f}")
    put("sepTieThreeFourKFour", f"{100 * (gap34[d.k >= 4] < TIE).mean():.0f}")
    put("sepTieThreeAnyKFour", f"{100 * ((gap23 < TIE) | (gap34 < TIE))[d.k >= 4].mean():.0f}")
    put("sepTieAll", f"{100 * (gap23 < TIE).mean():.0f}")
    put("sepTieKThree", f"{100 * (gap23[d.k >= 3] < TIE).mean():.0f}")
    put("sepGapOneMin", f"{gap12.min():.2f}"); put("sepGapOnePct", f"{100 * gap12.min():.0f}")
    put("sepGapTwoMed", f"{gap23.median():.2f}")
    tie = sep.mingap23 < TIE
    put("sepPairs", str(len(sep))); put("sepPairsTie", str(int(tie.sum())))
    for col, nm in (("ang1", "AngOne"), ("ang2", "AngTwo"), ("ang2_in23", "AngTwoPair")):
        put(f"sep{nm}Tie", f"{sep.loc[tie, col].median():.0f}")
        put(f"sep{nm}Sep", f"{sep.loc[~tie, col].median():.0f}")
    g = core[(core.source == "const") & (core.k >= 3)].copy()
    g["e2"] = deg(g.A2) - deg(g.OE2); g["e3"] = deg(g.A3) - deg(g.OE3)
    g["oe2"] = deg(g.OE2); g["oe3"] = deg(g.OE3)
    mm = g.groupby("formation_date")[["e2", "e3", "oe2", "oe3", "rot2", "rot3"]].mean()
    mm["gap"] = gap23.reindex(mm.index); mm["tie"] = (mm.gap < TIE).astype(float)
    mm["tie3"] = ((mm.gap < TIE) | (gap34.reindex(mm.index) < TIE)).astype(float)
    put("sepRotTwoTie", f"{mm.loc[mm.tie == 1, 'rot2'].median():.0f}")
    put("sepRotTwoSep", f"{mm.loc[mm.tie == 0, 'rot2'].median():.0f}")
    put("sepErrTwoTie", f"{abs(mm.loc[mm.tie == 1, 'e2'].median()):.1f}")
    put("sepErrTwoSep", f"{abs(mm.loc[mm.tie == 0, 'e2'].median()):.1f}")
    put("sepOeTwoTie", f"{mm.loc[mm.tie == 1, 'oe2'].median():.0f}")
    put("sepOeTwoSep", f"{mm.loc[mm.tie == 0, 'oe2'].median():.0f}")
    coef, se = {}, {}
    for j, tcol in ((2, "tie"), (3, "tie3")):
        X = np.c_[np.ones(len(mm)), mm[tcol], mm[f"oe{j}"], mm[f"oe{j}"] ** 2]
        yv = mm[f"e{j}"].to_numpy()
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
        r = yv - X @ b
        cov = np.linalg.inv(X.T @ X) * (r @ r) / (len(yv) - X.shape[1])
        coef[j], se[j] = b, np.sqrt(np.diag(cov))
    put("sepTieSeTwo", f"{se[2][1]:.2f}"); put("sepTieSeThree", f"{se[3][1]:.2f}")
    put("sepTieEffTwo", f"$\\num{{{coef[2][1]:.2f}}}$"); put("sepTieEffThree", f"$\\num{{{coef[3][1]:.2f}}}$")
    put("sepTieEffTwoAbs", f"{abs(coef[2][1]):.1f}"); put("sepTieEffThreeAbs", f"{abs(coef[3][1]):.1f}")
    put("sepMonthsSim", str(len(mm)))

    edges = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 1.0])
    mids = np.sqrt(np.maximum(edges[:-1], 0.02) * np.minimum(edges[1:], 0.65))
    def binned(xv, yv):
        b = pd.cut(xv, edges); return yv.groupby(b, observed=False).median().to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.45))
    ax[0].scatter(sep.mingap23, sep.ang2, s=5, color=C[2], alpha=0.35, lw=0)
    ax[0].plot(mids, binned(sep.mingap23, sep.ang2), "-o", ms=3.5, color=C[2], label="direction 2")
    ax[0].plot(mids, binned(sep.mingap23, sep.ang2_in23), "--s", ms=3, color=INK2, lw=1.2,
               label="direction 2, to last month's\nspan of directions 2 and 3")
    ax[0].set_ylabel("month-to-month angle (degrees)"); ax[0].set_ylim(0, 92)
    ax[0].legend(loc="upper right", fontsize=6.3)
    panel_label(ax[0], "(a) Stability, clipped returns")
    ax[1].plot(mids, binned(mm.gap, mm.rot2), "-o", ms=3.5, color=C[2], label="rotation of direction 2")
    ax[1].plot(mids, binned(mm.gap, mm.e2), "-s", ms=3, color=C[2], lw=1.1, alpha=0.6, label=r"$A_2$ minus $\mathrm{OE}_2$ (as angles)")
    ax[1].plot(mids, binned(mm.gap, mm.e3), "-^", ms=3, color=C[3], lw=1.1, label=r"$A_3$ minus $\mathrm{OE}_3$ (as angles)")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_ylabel("degrees"); ax[1].legend(loc="upper right", fontsize=6.3)
    panel_label(ax[1], "(b) Calibrated simulation")
    for a in ax:
        a.set_xscale("log"); a.set_xlim(0.018, 0.7)
        a.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); a.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
        a.minorticks_off(); a.axvline(TIE, color=RAWC, lw=0.8, ls=":")
        a.set_xlabel(r"relative gap $(\theta_2-\theta_3)/\theta_2$")
    fig.tight_layout(); fig.savefig(FIG / "fig_separation.pdf", bbox_inches="tight"); plt.close(fig)

    lines = ["% generated by figure_code/outline_20260927/make_exhibits_0927.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    (TAB / "assumptions.tex").write_text("\n".join(lines) + "\n")
    (OUT / "exhibits_0927.json").write_text(json.dumps(dict(macros=M, sim=sim_err), indent=1))
    print("wrote fig_dayvar.pdf, fig_separation.pdf,", len(M), "macros")


if __name__ == "__main__":
    main()
