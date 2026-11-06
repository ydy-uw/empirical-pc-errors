"""Figures, tables and quoted numbers for the 26 September outline.

Everything is drawn from aggregate outputs that already exist in the project:
  outlier_cleaning/experiments/00_raw/results/         untreated panels (benchmark)
  outlier_cleaning/experiments/01_fixed_clip/results/  clipped panels; the main
      specification is the setting c050 (every daily return clipped to [-50%, +50%])
  figure_code/out/outline_20260926/breadth.csv,
      breadth_resid.csv                                breadth.py (Sections 4 and 6)
  figure_code/out/outline_20260926/localized_check.json  localized_check.py (Sections 3 and 5)
No licensed data is read here.  Output:
  <outline>/figures/fig_*.pdf, <outline>/tables/{numbers,headline,accuracy}.tex,
  figure_code/out/outline_20260926/exhibits.json
Run from the project root:  python figure_code/outline_20260926/make_exhibits.py [outline_dir]
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figure_code"))
from theory import sd_first_order  # noqa: E402

EXP = ROOT / "outlier_cleaning" / "experiments"
RAW = EXP / "00_raw" / "results"
CLP = EXP / "01_fixed_clip" / "results"
OUT = ROOT / "figure_code" / "out" / "outline_20260926"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "overleaf_upload_20260926"
FIG, TAB = DST / "figures", DST / "tables"
P_GRID = (500, 1000, 2000, 3000)
PW = dict(zip(P_GRID, ("Five", "Thou", "TwoThou", "Three")))   # macro suffix by universe size
JW = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 10: "Ten"}
N_DUAL = 62
# crisis windows of formation months (outlier_cleaning/code/oc_common.py)
CRISES = [("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-06-30")]

# palette: first four categorical slots of the validated reference palette
C = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, INK2, GRID, RAWC = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8986"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
W2 = (6.5, 2.45)


def deg(a):
    return np.degrees(np.arcsin(np.sqrt(np.clip(np.asarray(a, float), 0, 1))))


def load(path):
    return pd.read_csv(path, parse_dates=["formation_date"])


def shade(ax):
    for a, b in CRISES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d9d8d3", alpha=0.6, lw=0, zorder=0)


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", color=INK)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def block_ci(values, block=12, draws=4000, seed=20260926):
    """95% percentile interval for the median over months from a circular block bootstrap.
    values: one entry per formation month in calendar order, NaN where undefined."""
    v = np.asarray(values, float); T = len(v)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=(draws, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(draws, -1)[:, :T]
    meds = np.nanmedian(v[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def spectra_estimates(npz_path, panels):
    """A_j at a fixed count k and at each panel's own skewness count, from stored spectra."""
    d = np.load(npz_path)
    sp = pd.DataFrame({"formation_date": pd.to_datetime(d["formation_date"]), "n_assets": d["n_assets"]})
    th = d["theta"]
    sp = sp.merge(panels[["formation_date", "n_assets", "k_skew"]], on=["formation_date", "n_assets"], how="left")
    out = sp.copy()
    for k in (2, 3):
        ell = th[:, k:].mean(axis=1)
        for j in range(1, k + 1):
            out[f"A{j}_k{k}"] = ell / th[:, j - 1]
    own = np.array([th[i, int(k):].mean() for i, k in enumerate(sp.k_skew)])
    kk = sp.k_skew.to_numpy()
    s3 = np.where(kk >= 3, (own / th[:, 0] + own / th[:, 1] + own / th[:, 2]) / 3, np.nan)
    out["S3_over_3"] = s3
    out["S2_over_2"] = np.where(kk >= 2, (own / th[:, 0] + own / th[:, 1]) / 2, np.nan)
    ell2 = th[:, 2:].mean(axis=1)
    out["S2_over_2_k2"] = (ell2 / th[:, 0] + ell2 / th[:, 1]) / 2
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    raw = load(RAW / "panels_main.csv")
    cln = load(CLP / "panels_c050.csv")
    sraw = json.load(open(RAW / "summary_main.json"))
    scln = json.load(open(CLP / "summary_c050.json"))
    ctx = json.load(open(ROOT / "outlier_cleaning" / "comparison" / "context.json"))
    locchk = json.load(open(OUT / "localized_check.json"))
    N = {"macros": {}}

    def put(name, val):
        N["macros"][name] = val

    # ------------------------------------------------------------ Figure 2: the clip level
    levels = [("none", RAW / "summary_main.json"), (r"$\pm$200%", CLP / "summary_c200.json"),
              (r"$\pm$100%", CLP / "summary_main.json"), (r"$\pm$50%", CLP / "summary_c050.json"),
              (r"$\pm$25%", CLP / "summary_c025.json")]
    rows = []
    for lab, sf in levels:
        s = json.load(open(sf))
        t = s.get("touched", {}).get("3000", {})
        rows.append(dict(level=lab, share=100 * t.get("share_changed_mean", 0.0),
                         **{f"pass{j}": 100 * s["screen"]["3000"][str(j)]["pass_share"] for j in (1, 2, 3)},
                         **{f"ang{j}": s["estimates"]["3000"][str(j)]["median_deg"] for j in (1, 2, 3)},
                         **{f"pass{j}_500": 100 * s["screen"]["500"][str(j)]["pass_share"] for j in (1, 2, 3)}))
    dose = pd.DataFrame(rows)
    x = np.arange(len(dose))
    ticks = [f"{r.level}\n{('%.3f' % r.share) if r.share >= 0.001 else ('%.4f' % r.share) if r.share > 0 else '0'}%"
             for r in dose.itertuples()]
    fig, ax = plt.subplots(1, 2, figsize=W2)
    for j in (1, 2, 3):
        ax[0].plot(x, dose[f"pass{j}"], "-o", ms=4, color=C[j], label=f"direction {j}")
        ax[1].plot(x, dose[f"ang{j}"], "-o", ms=4, color=C[j], label=f"direction {j}")
    for a in ax:
        a.axvline(3, color=INK2, lw=0.8, ls=":")
        a.set_xticks(x); a.set_xticklabels(ticks)
        a.set_xlabel("clip level and share of entries changed", color=INK2)
    ax[0].set_ylabel("pass rate (% of months)"); ax[0].set_ylim(35, 106)
    ax[0].text(2.93, 103.5, "chosen", fontsize=6.5, color=INK2, ha="right")
    ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.35))
    panel_label(ax[0], "(a) Pervasiveness screen, P = 3,000")
    ax[1].set_ylabel("median angle (degrees)"); ax[1].set_ylim(10, 42)
    panel_label(ax[1], "(b) Median out-of-subspace angle, P = 3,000")
    save(fig, "fig_clip_level.pdf")
    N["dose"] = dose.to_dict(orient="records")
    r50 = dose.iloc[3]; r25 = dose.iloc[4]; r100 = dose.iloc[2]
    put("passTwoCHundred", f"{r100.pass2:.0f}"); put("passThreeCHundred", f"{r100.pass3:.0f}")
    put("passTwoCQuarterFive", f"{r25.pass2_500:.0f}"); put("passThreeCQuarterFive", f"{r25.pass3_500:.0f}")
    put("angOneCQuarter", f"{r25.ang1:.1f}")
    put("chgRatioQuarter", f"{r25.share / r50.share:.0f}")

    # ------------------------------------------------------------ Figure 3: counting factors
    rules = [("skew", "skewness", C[1]), ("bn", "Bai–Ng ICp2", C[2]),
             ("ah", "eigenvalue ratio", C[3]), ("ed", "Onatski ED", C[4])]
    fig, ax = plt.subplots(1, 2, figsize=W2, gridspec_kw=dict(width_ratios=[1, 1.5]))
    for i, (r, lab, col) in enumerate(rules):
        med = [scln["counts"][str(P)][r]["median"] for P in P_GRID]
        lo = [scln["counts"][str(P)][r]["q25"] for P in P_GRID]
        hi = [scln["counts"][str(P)][r]["q75"] for P in P_GRID]
        xs = np.log10(P_GRID) + (i - 1.5) * 0.035
        ax[0].errorbar(xs, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="-o", ms=3.5,
                       color=col, lw=1.3, capsize=0, elinewidth=1, label=lab)
    rs = [sraw["counts"][str(P)]["skew"]["median"] for P in P_GRID]
    ax[0].plot(np.log10(P_GRID), rs, ls="--", color=RAWC, lw=1.1, label="skewness, untreated")
    ax[0].set_xticks(np.log10(P_GRID)); ax[0].set_xticklabels([f"{P:,}" for P in P_GRID])
    ax[0].set_xlabel("universe size P"); ax[0].set_ylabel("number of factors")
    ax[0].set_ylim(0, 12.5); ax[0].legend(loc="upper right", fontsize=6.5)
    panel_label(ax[0], "(a) Counts by rule (median, IQR)")
    g = cln[cln.p == 3000].sort_values("formation_date")
    shade(ax[1])
    ax[1].step(g.formation_date, g.k_skew, where="mid", color=C[1], lw=1.2, label="skewness")
    ax[1].step(g.formation_date, g.k_bn, where="mid", color=C[2], lw=1.0, label="Bai–Ng ICp2")
    ax[1].set_ylabel("number of factors"); ax[1].set_ylim(0, 19); ax[1].set_yticks([0, 4, 8, 12, 16])
    ax[1].legend(loc="upper left", ncol=2)
    panel_label(ax[1], "(b) Counts by month, P = 3,000")
    save(fig, "fig_counts.pdf")
    for P in P_GRID:
        put(f"kSkew{PW[P]}", f"{scln['counts'][str(P)]['skew']['median']:.0f}")
        put(f"kSkewRaw{PW[P]}", f"{sraw['counts'][str(P)]['skew']['median']:.0f}")
        put(f"kBn{PW[P]}", f"{scln['counts'][str(P)]['bn']['median']:.0f}")
        put(f"kBnRaw{PW[P]}", f"{sraw['counts'][str(P)]['bn']['median']:.0f}")
        put(f"kEd{PW[P]}", f"{scln['counts'][str(P)]['ed']['median']:.0f}")
        put(f"agree{PW[P]}", f"{100 * scln['counts'][str(P)]['agree_with_raw']:.0f}")
        put(f"kChange{PW[P]}", f"{100 * scln['counts'][str(P)]['changes_month_to_month']:.0f}")
        put(f"intSkew{PW[P]}", f"{scln['interval'][str(P)]['skew']:.1f}")
        put(f"intBn{PW[P]}", f"{scln['interval'][str(P)]['bn']:.1f}")
    kmax = g.nlargest(2, "k_skew")
    N["count_peaks"] = [(str(r.formation_date.date()), int(r.k_skew)) for r in kmax.itertuples()]

    # ------------------------------------------------------------ Figure 4: eigenvalue ladder and residual block
    bd = pd.read_csv(OUT / "breadth.csv", keep_default_na=False)
    kinds = ("data", "twofactor_flip", "twofactor", "twofactor_rotate")
    ex = {k: bd[(bd.kind == k) & bd.stat.str.match(r"^e\d+$")].pivot(index="stat", columns="p", values="median")
          for k in kinds}
    angs = bd[bd.stat.str.startswith("A")].pivot_table(index=["kind", "stat"], columns="p", values="median")
    pb = 3000
    J = np.arange(1, 16)
    lad = {k: np.array([ex[k].loc[f"e{j}", pb] for j in J]) for k in kinds}
    fig, ax = plt.subplots(1, 2, figsize=W2)
    ax[0].plot(J, lad["data"], "-o", ms=3.5, color=C[1], lw=1.5, label="clipped returns")
    ax[0].plot(J[2:], lad["twofactor_flip"][2:], "--s", ms=3, color=C[2], lw=1.2,
               label="two factors + residuals with random signs")
    ax[0].plot(J[2:], lad["twofactor"][2:], ":^", ms=3, color=RAWC, lw=1.2,
               label="two factors + residuals shuffled in time")
    ax[0].set_yscale("log"); ax[0].minorticks_off()
    ax[0].set_yticks([0.5, 1, 2, 5, 10, 20]); ax[0].set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax[0].set_xticks([1, 2, 3, 5, 10, 15])
    ax[0].set_xlabel("eigenvalue index j"); ax[0].set_ylabel(r"excess over the bulk, $\theta_j/\ell_{20}-1$")
    ax[0].legend(loc="upper right", fontsize=6.2)
    panel_label(ax[0], "(a) Eigenvalue excess, P = 3,000")
    blk_c = pd.read_csv(CLP / "blocks_c050.csv"); blk_r = pd.read_csv(RAW / "blocks.csv")
    for P, mk, col in ((500, "s", C[2]), (3000, "o", C[1])):
        for dfb, ls, lab in ((blk_c, "-", f"P = {P:,}"), (blk_r, "--", None)):
            h = dfb[(dfb.n_assets == P) & dfb.m.isin([1, 3, 5, 8, 12, 20, 40])].drop_duplicates(["formation_date", "m"])
            m = h.groupby("m").d1.median() * 100
            ax[1].plot(m.index, m.values, ls=ls, marker=mk, ms=3, color=col if ls == "-" else RAWC,
                       lw=1.3 if ls == "-" else 0.9, label=lab)
    ax[1].plot([], [], ls="--", color=RAWC, label="untreated")
    ax[1].set_xscale("log"); ax[1].set_xticks([1, 3, 5, 8, 12, 20, 40])
    ax[1].set_xticklabels(["1", "3", "5", "8", "12", "20", "40"]); ax[1].minorticks_off()
    ax[1].set_ylim(0, 11.5)
    ax[1].set_xlabel("principal components removed, q"); ax[1].set_ylabel("mean residual correlation (%)")
    ax[1].legend(loc="upper right", fontsize=6.5)
    panel_label(ax[1], "(b) Residual correlation, largest 10%")
    save(fig, "fig_breadth_residual.pdf")
    xs_ratio = lad["data"][2:10] / lad["twofactor_flip"][2:10]            # j = 3..10
    dv_ratio = lad["twofactor_flip"][2:10] / lad["twofactor"][2:10]
    ser_ratio = lad["twofactor_rotate"][2:10] / lad["twofactor"][2:10]      # serial dependence only
    put("serRatioLo", f"{ser_ratio.min():.2f}"); put("serRatioHi", f"{ser_ratio.max():.2f}")
    put("ladRotThree", f"{lad['twofactor_rotate'][2]:.2f}")
    br = pd.read_csv(OUT / "breadth_resid.csv")
    put("rhoOnePooled", f"$\\num{{{br.rho1_pooled.median():.3f}}}$")
    put("rhoOneMedian", f"$\\num{{{br.rho1_median.median():.3f}}}$")
    put("topCorrKTwo", f"{100 * br.top_corr.median():.1f}")
    put("topVarRatioKTwo", f"{br.top_var_ratio.median():.2f}")
    N["ladder_P3000"] = {k: v.tolist() for k, v in lad.items()}
    for j in (3, 5, 10):
        put(f"lad{JW[j]}", f"{lad['data'][j - 1]:.2f}")
        put(f"ladFlip{JW[j]}", f"{lad['twofactor_flip'][j - 1]:.2f}")
        put(f"ladShuf{JW[j]}", f"{lad['twofactor'][j - 1]:.2f}")
    put("ladOne", f"{lad['data'][0]:.1f}"); put("ladTwo", f"{lad['data'][1]:.1f}")
    put("xsRatioLo", f"{xs_ratio.min():.1f}"); put("xsRatioHi", f"{xs_ratio.max():.1f}")
    put("dvRatioLo", f"{dv_ratio.min():.1f}"); put("dvRatioHi", f"{dv_ratio.max():.1f}")
    small = min(int(c) for c in ex["data"].columns)
    xs_small = np.array([ex["data"].loc[f"e{j}", small] / ex["twofactor_flip"].loc[f"e{j}", small] for j in range(3, 11)])
    put("xsRatioSmallLo", f"{xs_small.min():.1f}"); put("xsRatioSmallHi", f"{xs_small.max():.1f}")
    for j in (1, 2, 3):
        put(f"subAng{JW[j]}KTwoSmall", f"{deg(angs.loc[('data', f'A{j}_k2'), 375]):.1f}")
        put(f"subAng{JW[j]}KTwoBig", f"{deg(angs.loc[('data', f'A{j}_k2'), 3000]):.1f}")
    rv = locchk["top_decile_resid_var_ratio"]
    put("topVarRatio", f"{rv['median']:.2f}")
    bs = scln["blocks"]
    for P in (500, 3000):
        put(f"blockOwn{PW[P]}", f"{bs[str(P)]['own_k']:.1f}")
        put(f"blockFortyM{PW[P]}", f"{bs[str(P)]['m40']:.1f}")
        put(f"blockFiveM{PW[P]}", f"{bs[str(P)]['m5']:.1f}")
    put("blockExcess", f"{N_DUAL * br.top_corr.median() * 0.1 * br.top_var_ratio.median():.2f}")
    gap = lad["data"][2:10] - lad["twofactor_flip"][2:10]
    put("xsGapLo", f"{gap.min():.2f}"); put("xsGapHi", f"{gap.max():.2f}")
    for P in P_GRID:
        gg = cln[(cln.p == P) & (cln.k_skew >= 3)]
        put(f"nearDeg{PW[P]}", f"{100 * (gg.relgap23 < 0.1).mean():.0f}")
        put(f"nearDegAll{PW[P]}", f"{100 * (cln[cln.p == P].relgap23 < 0.1).mean():.0f}")

    # ------------------------------------------------------------ Figure 5: pervasiveness
    fig, ax = plt.subplots(1, 2, figsize=W2)
    for j in (1, 2, 3):
        pc = [100 * scln["screen"][str(P)][str(j)]["pass_share"] for P in P_GRID]
        pr = [100 * sraw["screen"][str(P)][str(j)]["pass_share"] for P in P_GRID]
        ax[0].plot(np.log10(P_GRID), pc, "-o", ms=4, color=C[j], label=f"direction {j}")
        ax[0].plot(np.log10(P_GRID), pr, "--", marker="o", ms=3, mfc="white", color=C[j], lw=1.1)
    ax[0].plot([], [], "--", color=RAWC, label="untreated (dashed)")
    ax[0].set_xticks(np.log10(P_GRID)); ax[0].set_xticklabels([f"{P:,}" for P in P_GRID])
    ax[0].set_xlabel("universe size P"); ax[0].set_ylabel("pass rate (% of months)")
    ax[0].set_ylim(35, 103); ax[0].legend(loc="lower left")
    panel_label(ax[0], "(a) Pass rate by universe size")
    for j, dfp, kcol, ls, lab in ((2, cln, "k_ref", "-", "direction 2, clipped"), (3, cln, "k_ref", "-", "direction 3, clipped"),
                                  (2, raw, "k_skew", "--", "direction 2, untreated"), (3, raw, "k_skew", "--", "direction 3, untreated")):
        h = dfp[(dfp.p == 3000) & (dfp[kcol] >= j)][f"part{j}"]
        v = np.sort(h.values); y = np.arange(1, len(v) + 1) / len(v)
        ax[1].plot(v, y, ls=ls, color=C[j], lw=1.3 if ls == "-" else 1.0, label=lab)
    ax[1].axvline(0.01, color=INK2, lw=0.8, ls=":")
    ax[1].text(0.0115, 0.27, "1% of stocks", fontsize=6.3, color=INK2, ha="left")
    ax[1].set_xscale("log"); ax[1].set_xlabel("participation, N(h)/P"); ax[1].set_ylabel("cumulative share of months")
    ax[1].legend(loc="upper left", fontsize=6.3)
    panel_label(ax[1], "(b) Participation, P = 3,000")
    save(fig, "fig_pervasive.pdf")
    for P in P_GRID:
        for j in (1, 2, 3):
            put(f"pass{JW[j]}{PW[P]}", f"{100 * scln['screen'][str(P)][str(j)]['pass_share']:.0f}")
            put(f"passRaw{JW[j]}{PW[P]}", f"{100 * sraw['screen'][str(P)][str(j)]['pass_share']:.0f}")
            put(f"loc{JW[j]}{PW[P]}", f"{scln['screen'][str(P)][str(j)]['localized']}")
            put(f"locRaw{JW[j]}{PW[P]}", f"{sraw['screen'][str(P)][str(j)]['localized']}")
            put(f"part{JW[j]}{PW[P]}", f"{100 * scln['screen'][str(P)][str(j)]['participation_median']:.0f}")
            put(f"partRaw{JW[j]}{PW[P]}", f"{100 * sraw['screen'][str(P)][str(j)]['participation_median']:.1f}")
    # localized months: what the direction loads on
    for P in ("500", "3000"):
        for j in ("1", "2", "3"):
            if j in locchk[P]:
                q = locchk[P][j]
                put(f"locHundred{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_ret_gt_100']:.0f}")
                put(f"locFifty{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_ret_gt_50']:.0f}")
                put(f"locMax{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_is_panel_max']:.0f}")
                put(f"locRet{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_ret_median']:.0f}")
    for P in ("500", "3000"):
        for j, q in locchk["clipped"].get(P, {}).items():
            put(f"clipLocHit{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_hits_clip']:.0f}")
            put(f"clipLocMax{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_is_panel_max']:.0f}")
            put(f"clipLocRet{JW[int(j)]}{PW[int(P)]}", f"{100 * q['top_stock_raw_ret_median']:.0f}")
            put(f"clipLocN{JW[int(j)]}{PW[int(P)]}", f"{q['months']}")
    # untreated localized leading months: angle and effect of clipping
    m = raw.merge(cln, on=["formation_date", "n_assets"], suffixes=("_r", "_c"))
    g3 = m[(m.n_assets == 3000) & (m.loc1_r.astype(bool))]
    put("locOneAngRaw", f"{np.median(deg(g3.A1_r)):.1f}")
    put("locOneAngGain", f"{np.median(deg(g3.A1_c) - deg(g3.A1_r)):.1f}")
    put("locOneBelowFive", f"{int((deg(g3.A1_r) < 5).sum())}")
    for j in (2, 3):
        both = m[(m.n_assets == 3000) & (m.k_skew_r >= j) & (m.k_skew_c >= j)]
        bad = both[both[f"loc{j}_r"].astype(bool)]; ok = both[~both[f"loc{j}_r"].astype(bool)]
        put(f"gainFail{JW[j]}", f"{np.median(deg(bad[f'A{j}_c']) - deg(bad[f'A{j}_r'])):.1f}")
        put(f"gainPass{JW[j]}", f"{abs(np.median(deg(ok[f'A{j}_c']) - deg(ok[f'A{j}_r']))):.1f}")
    # effect of excluding localized months from the clipped medians
    mx = 0.0
    for P in P_GRID:
        gg = cln[cln.p == P]
        for j in (1, 2, 3):
            h = gg[gg.k_skew >= j]
            mx = max(mx, abs(deg(h[f"A{j}"].median()) - deg(h[~h[f"loc{j}"].astype(bool)][f"A{j}"].median())))
    put("exclLocMax", f"{mx:.1f}")

    # ------------------------------------------------------------ Figure 6: the size of the errors
    sp = spectra_estimates(CLP / "spectra_c050.npz", cln)
    fig, ax = plt.subplots(1, 2, figsize=W2, gridspec_kw=dict(width_ratios=[1, 1.5]))
    for j in (1, 2, 3):
        med = [scln["estimates"][str(P)][str(j)]["median_deg"] for P in P_GRID]
        lo = [scln["estimates"][str(P)][str(j)]["q25_deg"] for P in P_GRID]
        hi = [scln["estimates"][str(P)][str(j)]["q75_deg"] for P in P_GRID]
        rm = [sraw["estimates"][str(P)][str(j)]["median_deg"] for P in P_GRID]
        ax[0].fill_between(np.log10(P_GRID), lo, hi, color=C[j], alpha=0.13, lw=0)
        ax[0].plot(np.log10(P_GRID), med, "-o", ms=4, color=C[j], label=f"direction {j}")
        ax[0].plot(np.log10(P_GRID), rm, "--", color=C[j], lw=1.0)
    ax[0].plot([], [], "--", color=RAWC, label="untreated median")
    ax[0].set_xticks(np.log10(P_GRID)); ax[0].set_xticklabels([f"{P:,}" for P in P_GRID])
    ax[0].set_xlabel("universe size P"); ax[0].set_ylabel("angle to loading span (degrees)")
    ax[0].set_ylim(0, 52); ax[0].legend(loc="upper left", fontsize=6.3, ncol=2)
    panel_label(ax[0], "(a) Median and IQR over months")
    g = cln[cln.p == 3000].sort_values("formation_date")
    peff3 = scln["accuracy"]["3000"]["p_eff"]
    a1 = g.A1.to_numpy()
    sdd = np.degrees(sd_first_order(a1, peff3, N_DUAL, g.k_skew.to_numpy()) / (2 * np.sqrt(a1 * (1 - a1))))
    shade(ax[1])
    ax[1].fill_between(g.formation_date, deg(a1) - 2 * sdd, deg(a1) + 2 * sdd, color=C[1], alpha=0.25, lw=0,
                       label=r"$\pm$2 sampling sd")
    ax[1].plot(g.formation_date, deg(a1), color=C[1], lw=1.2, label="direction 1, clipped")
    gr = raw[raw.p == 3000].sort_values("formation_date")
    ax[1].plot(gr.formation_date, deg(gr.A1), color=RAWC, lw=0.8, ls="--", label="untreated")
    ax[1].set_ylabel("degrees"); ax[1].set_ylim(0, 34); ax[1].legend(loc="upper left", ncol=3, fontsize=6.3)
    panel_label(ax[1], "(b) Leading direction by month, P = 3,000")
    save(fig, "fig_errors.pdf")
    put("sdBandThree", f"{np.median(sdd):.2f}")

    # ------------------------------------------------------------ Table 2: estimates
    CI, CI24, CIk2, CIs3, CIs2, CIs2k2 = {}, {}, {}, {}, {}, {}
    widen = 0.0
    for P in P_GRID:
        gg = cln[cln.p == P].sort_values("formation_date")
        ss = sp[sp.n_assets == P].sort_values("formation_date")
        for j in (1, 2, 3):
            a = np.where(gg.k_skew >= j, deg(gg[f"A{j}"]), np.nan)
            CI[(P, j)] = block_ci(a)
            CI24[(P, j)] = block_ci(a, block=24)
            widen = max(widen, (CI24[(P, j)][1] - CI24[(P, j)][0]) - (CI[(P, j)][1] - CI[(P, j)][0]))
        for j in (1, 2):
            CIk2[(P, j)] = (float(np.median(deg(ss[f"A{j}_k2"]))), *block_ci(deg(ss[f"A{j}_k2"].to_numpy())))
        s3 = ss.S3_over_3.to_numpy()
        CIs3[P] = (float(np.nanmedian(s3)), *block_ci(s3))
        CIs2[P] = float(np.nanmedian(ss.S2_over_2)); CIs2k2[P] = float(np.nanmedian(ss.S2_over_2_k2))
    put("ciWidenMax", f"{widen:.1f}")
    def row(label, vals):
        return label + " & " + " & ".join(vals) + r"\\"
    L = [r"\begin{tabular}{@{}l cccc@{}}", r"\toprule",
         row("", [f"$P={P:,}$" for P in P_GRID]), r"\midrule",
         r"\multicolumn{5}{@{}l}{\emph{At the skewness count}}\\",
         row(r"\quad median count $k$", [f"{scln['counts'][str(P)]['skew']['median']:.0f}" for P in P_GRID])]
    for j in (1, 2, 3):
        L.append(row(rf"\quad direction {j}",
                     [f"{scln['estimates'][str(P)][str(j)]['median_deg']:.1f} [{CI[(P, j)][0]:.1f}, {CI[(P, j)][1]:.1f}]" for P in P_GRID]))
    L.append(row(r"\quad $S_2/2$", [f"{100 * CIs2[P]:.0f}\\%" for P in P_GRID]))
    L.append(row(r"\quad $S_3/3$ (if $k\ge3$)", [f"{100 * CIs3[P][0]:.0f}\\%" for P in P_GRID]))
    L += [r"\addlinespace", r"\multicolumn{5}{@{}l}{\emph{At two factors}}\\"]
    for j in (1, 2):
        L.append(row(rf"\quad direction {j}",
                     [f"{CIk2[(P, j)][0]:.1f} [{CIk2[(P, j)][1]:.1f}, {CIk2[(P, j)][2]:.1f}]" for P in P_GRID]))
    L.append(row(r"\quad $S_2/2$", [f"{100 * CIs2k2[P]:.0f}\\%" for P in P_GRID]))
    L += [r"\addlinespace", r"\multicolumn{5}{@{}l}{\emph{Pass rate of the pervasiveness screen}}\\"]
    for j in (2, 3):
        L.append(row(rf"\quad direction {j}",
                     [f"{100 * scln['screen'][str(P)][str(j)]['pass_share']:.0f}\\%" for P in P_GRID]))
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "headline.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------ Table 3: accuracy and sensitivity
    L = [r"\begin{tabular}{@{}r c ccc ccc ccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{sampling sd} & \multicolumn{3}{c}{leading bias} & \multicolumn{3}{c}{factor-analytic minus $A_j$}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
         r"$P$ & $\peff$ & 1 & 2 & 3 & 1 & 2 & 3 & 1 & 2 & 3\\", r"\midrule"]
    for P in P_GRID:
        e = scln["estimates"][str(P)]; a = scln["accuracy"][str(P)]; fa = scln["fa"].get(str(P))
        cells = [f"{P:,}", f"{a['p_eff']:,.0f}"]
        cells += [f"{e[str(j)]['sd_deg']:.2f}" for j in (1, 2, 3)]
        cells += [f"$\\num{{{e[str(j)]['bias_deg']:.2f}}}$" for j in (1, 2, 3)]
        cells += ([f"$+{fa[str(j)]['diff_deg']:.1f}$" for j in (1, 2, 3)] if fa else ["--"] * 3)
        L.append(" & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "accuracy.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------ other numbers quoted in the text
    tc = scln["touched"]
    for P in P_GRID:
        w = PW[P]
        put(f"chg{w}", f"{tc[str(P)]['entries_changed']:,}")
        put(f"chgPct{w}", f"{100 * tc[str(P)]['share_changed_mean']:.3f}")
        put(f"touched{w}", f"{tc[str(P)]['panels_touched']}")
        put(f"ext{w}", f"{int(raw[raw.p == P].n_above_100.sum()):,}")
        put(f"extPanels{w}", f"{int((raw[raw.p == P].n_above_100 > 0).sum())}")
        put(f"peff{w}", f"{scln['accuracy'][str(P)]['p_eff']:,.0f}")
        for j in (1, 2, 3):
            jj = JW[j]
            put(f"ang{jj}{w}", f"{scln['estimates'][str(P)][str(j)]['median_deg']:.1f}")
            put(f"angRaw{jj}{w}", f"{sraw['estimates'][str(P)][str(j)]['median_deg']:.1f}")
            put(f"sd{jj}{w}", f"{scln['estimates'][str(P)][str(j)]['sd_deg']:.2f}")
            put(f"bias{jj}{w}", f"$\\num{{{scln['estimates'][str(P)][str(j)]['bias_deg']:.2f}}}$")
            lo, hi = CI[(P, j)]
            put(f"ciLo{jj}{w}", f"{lo:.1f}"); put(f"ciHi{jj}{w}", f"{hi:.1f}")
            if scln["fa"].get(str(P)):
                put(f"fa{jj}{w}", f"{scln['fa'][str(P)][str(j)]['diff_deg']:.1f}")
        put(f"Sthree{w}", f"{100 * CIs3[P][0]:.0f}")
        put(f"Stwo{w}", f"{100 * CIs2[P]:.0f}"); put(f"StwoKTwo{w}", f"{100 * CIs2k2[P]:.0f}")
        put(f"Sk{w}", f"{scln['estimates'][str(P)]['S_over_k_pct']:.0f}")
        put(f"angOneKTwo{w}", f"{CIk2[(P, 1)][0]:.1f}"); put(f"angTwoKTwo{w}", f"{CIk2[(P, 2)][0]:.1f}")
        gnd = cln[(cln.p == P) & (cln.k_skew >= 2) & (cln.relgap23 >= 0.1)]
        put(f"ndExclTwo{w}", f"{np.median(deg(gnd.A2.to_numpy())):.1f}")
        snd = sp[sp.n_assets == P].merge(cln[cln.p == P][["formation_date", "relgap23"]], on="formation_date")
        put(f"ndExclTwoKTwo{w}", f"{np.median(deg(snd[snd.relgap23 >= 0.1].A2_k2.to_numpy())):.1f}")
        put(f"gfc{w}", f"{scln['crises'][str(P)]['gfc']['A1_deg']:.1f}")
        put(f"covid{w}", f"{scln['crises'][str(P)]['covid']['A1_deg']:.1f}")
        gg = cln[cln.p == P]
        put(f"residVar{w}", f"{gg.resid_var_mean.median():.3g}")
    c5, c3 = cln[cln.p == 500], cln[cln.p == 3000]
    put("residVarGrowth", f"{c3.resid_var_mean.median() / c5.resid_var_mean.median():.1f}")
    put("signalGrowth", f"{(c3.theta1 - c3.ell).median() / (c5.theta1 - c5.ell).median():.1f}")
    put("dayVarCvThree", f"{scln['fa']['3000']['day_var_cv']:.2f}")
    put("dayVarCvFive", f"{scln['fa']['500']['day_var_cv']:.2f}")
    put("faCovidThree", f"{scln['fa']['3000']['crisis_covid']['Afa1_deg']:.1f}")
    put("faGfcThree", f"{scln['fa']['3000']['crisis_gfc']['Afa1_deg']:.1f}")
    put("minRet", f"{100 * ctx['min_return']:.1f}")
    lines = ["% generated by figure_code/outline_20260926/make_exhibits.py -- do not edit"]
    for k, v in N["macros"].items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (TAB / "numbers.tex").write_text("\n".join(lines) + "\n")
    (OUT / "exhibits.json").write_text(json.dumps(N, indent=1, default=float))
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")), len(N["macros"]), "macros")


if __name__ == "__main__":
    main()
