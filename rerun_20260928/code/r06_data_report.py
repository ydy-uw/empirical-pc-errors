"""Data report for the cleaned sample of the 28 September 2026 rerun.

Reads the cleaned formation panels (panel cache built by panels.py from the cleaned universes),
the original panels (through companion/code/rawpanel.py with the frozen universes), and the
candidate tables of r02_candidates.py.  Writes only aggregates:
  rerun_20260928/report/data_report.html, figures/*.svg, and the PDF printed from the HTML
  (reports/2026-09-28_cleaned_data_report.pdf).
Run from the project root after run_all.sh:
  emp/.venv/Scripts/python.exe rerun_20260928/code/r06_data_report.py
"""
from __future__ import annotations
import os, sys, pickle, subprocess, json, html
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EMP = ROOT / "emp"
WORK = ROOT / "rerun_20260928" / "work"
CACHE = WORK / "outlier_cleaning" / "_cache"
CAND = EMP / "data/processed/clean_20260928/candidates"
NEWU = EMP / "data/processed/rolling/rolling_topcap_clean_200601_202511_20260928"
NAMES = EMP / "data/raw/wrds/crsp_names_20260928"
OUTD = ROOT / "rerun_20260928" / "report"; FIGD = OUTD / "figures"
PDF = ROOT / "reports" / "2026-09-28_cleaned_data_report.pdf"
P_GRID = (500, 1000, 2000, 3000)
CUT = 0.08
QS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
QLAB = ["min", "1", "5", "10", "25", "50", "75", "90", "95", "99", "max"]

C = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "amber": "#eda100", "gray": "#8a8986"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.4,
    "legend.frameon": False, "font.family": "DejaVu Sans", "svg.fonttype": "none"})


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGD / name, bbox_inches="tight"); plt.close(fig)
    return f"figures/{name}"


def title(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", color=INK)


# ------------------------------------------------------------------ data
def load_clean():
    return {P: pickle.load(open(CACHE / f"panels_P{P}.pkl", "rb")) for P in P_GRID}


def load_original_3000():
    os.environ.pop("HLHH_ROLLING_DIR", None)
    sys.path.insert(0, str(ROOT / "companion" / "code"))
    import rawpanel
    U = rawpanel.universes(); out = {}
    for fd in sorted(U.formation_date.unique()):
        X, cols = rawpanel.panel(pd.Timestamp(fd), 3000)
        out[pd.Timestamp(fd)] = dict(X=X.to_numpy(float), permnos=np.asarray(cols, np.int64), dates=np.asarray(X.index))
    return out


def unique_stock_days(panels):
    """Returns of the union of the panels, each (stock, day) once."""
    keys, vals = [], []
    for d in panels.values():
        dd = pd.DatetimeIndex(d["dates"]).values.astype("datetime64[D]").astype(np.int64)
        k = d["permnos"][None, :].astype(np.int64) * 100000 + dd[:, None]
        keys.append(k.ravel()); vals.append(d["X"].ravel())
    k = np.concatenate(keys); v = np.concatenate(vals)
    _, idx = np.unique(k, return_index=True)
    return v[idx]


def ann_vol(X, wins):
    if wins:
        lo = np.percentile(X, 1, axis=0); hi = np.percentile(X, 99, axis=0); X = np.clip(X, lo, hi)
    return X.std(axis=0, ddof=1) * np.sqrt(252)


def qrow(v, scale=100):
    q = np.percentile(v, QS) * scale
    return q


def fmt(x, d=2):
    return f"{x:,.{d}f}"


def table(header, rows, cls=""):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def main():
    FIGD.mkdir(parents=True, exist_ok=True)
    clean = load_clean()
    orig = load_original_3000()
    el = pd.concat([pd.read_parquet(f) for f in sorted(CAND.glob("year=*/eligible.parquet"))])
    ex = pd.read_parquet(NEWU / "exclusions.parquet")
    c = pd.read_parquet(NAMES / "stksecurityinfohist.parquet")
    c["securityenddt"] = pd.to_datetime(c.securityenddt)
    dl = c[c.delactiontype.notna() & (c.delactiontype != "N/A")].groupby("permno").agg(end=("securityenddt", "max"), act=("delactiontype", "last"))
    el = el.join(dl, on="permno")
    el["mer6"] = el.act.eq("MER") & (el.end > el.formation_date) & (el.end <= el.formation_date + pd.Timedelta(days=183))
    base = el[el.baseline_rank <= 3000]            # the original top-3000 companies
    ns = base[~base.spac]
    N = {}

    # ---------------- exclusions table
    rows = []
    for P in P_GRID:
        e = ex[ex.n_assets == P]; tot = P * len(e)
        rows.append([f"{P:,}", f"{e.spac_removed.sum():,}", f"{e.lowvol_removed.sum():,}",
                     f"{(e.spac_removed + e.lowvol_removed).sum() / tot:.2%}",
                     f"{(e.spac_removed > 0).sum()}", f"{(e.lowvol_removed > 0).sum()}",
                     f"{int(e.deepest_baseline_rank.max()):,}"])
    t_excl = table(["P", "SPAC stock-months removed", "low-vol stock-months removed", "share of stock-months replaced",
                    "months with a SPAC removed", "months with a low-vol removal", "deepest refill rank"], rows)
    e3 = ex[ex.n_assets == 3000]
    N["spac_peak"] = int(e3.spac_removed.max()); N["spac_peak_m"] = pd.Timestamp(e3.loc[e3.spac_removed.idxmax(), "formation_date"]).strftime("%B %Y")
    N["lv_mean3"] = e3.lowvol_removed.mean(); N["lv_max3"] = int(e3.lowvol_removed.max())

    # ---------------- figure 1: removals over time
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.3))
    e3 = e3.sort_values("formation_date")
    ax[0].bar(e3.formation_date, e3.spac_removed, width=25, color=C["orange"], label="SPACs")
    ax[0].bar(e3.formation_date, e3.lowvol_removed, width=25, bottom=e3.spac_removed, color=C["blue"], label="winsorized vol < 8%")
    ax[0].set_ylabel("companies removed"); ax[0].legend(loc="upper left"); title(ax[0], "(a) removed from the top 3,000, per month")
    for P, col in zip(P_GRID, (C["green"], C["amber"], C["gray"], C["blue"])):
        e = ex[ex.n_assets == P].sort_values("formation_date")
        ax[1].plot(e.formation_date, 100 * (e.spac_removed + e.lowvol_removed) / P, color=col, lw=1.1, label=f"P={P:,}")
    ax[1].set_ylabel("% of the universe replaced"); ax[1].legend(loc="upper left", ncol=2); title(ax[1], "(b) share replaced, by universe size")
    f_rem = save(fig, "fig_removed.svg")

    # ---------------- figure 2: volatility histograms and the cutoff
    bins = np.logspace(np.log10(0.005), np.log10(3.0), 80)
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5))
    groups = [(base[base.spac].vol_w, C["orange"], "SPAC"),
              (ns[ns.mer6].vol_w, C["amber"], "acquired < 6 mo."),
              (ns[~ns.mer6].vol_w, C["blue"], "other")]
    ax[0].hist([g[0] for g in groups], bins=bins, stacked=True, color=[g[1] for g in groups], label=[g[2] for g in groups], lw=0)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    for x, ls in ((0.05, ":"), (CUT, "-"), (0.10, ":")):
        ax[0].axvline(x, color=INK, lw=0.9, ls=ls)
    ax[0].set_xlabel("winsorized annualized volatility, 63 days"); ax[0].set_ylabel("stock-months (log scale)")
    ax[0].set_xticks([0.01, 0.02, 0.05, 0.08, 0.2, 0.5, 1, 2]); ax[0].set_xticklabels(["1%", "2%", "5%", "8%", "20%", "50%", "100%", "200%"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].legend(loc="upper left", fontsize=6.5, handlelength=1.2, borderaxespad=0.2)
    ax[0].set_ylim(0.7, 3e5)
    title(ax[0], "(a) original top-3,000, all 239 months")
    b2 = np.r_[0.005, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.20, 0.25, 0.3, 0.4, 0.6, 1.0]
    mid, share, cnt = [], [], []
    for a, b in zip(b2[:-1], b2[1:]):
        s = ns[(ns.vol_w > a) & (ns.vol_w <= b)]
        mid.append(np.sqrt(a * b)); share.append(100 * s.mer6.mean()); cnt.append(len(s))
    ax[1].plot(mid, share, "-o", ms=3, color=C["amber"])
    ax[1].axhline(100 * ns.mer6.mean(), color=C["gray"], lw=0.9, ls="--")
    ax[1].axvline(CUT, color=INK, lw=0.9)
    ax[1].set_xscale("log"); ax[1].set_xticks([0.02, 0.05, 0.08, 0.2, 0.5, 1]); ax[1].set_xticklabels(["2%", "5%", "8%", "20%", "50%", "100%"])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("winsorized annualized volatility"); ax[1].set_ylabel("% delisted by merger within 6 months")
    title(ax[1], "(b) non-SPAC stock-months: acquired soon after")
    f_vol = save(fig, "fig_vol_cutoff.svg")
    brow = []
    for a, b in zip(b2[:12], b2[1:13]):
        s = ns[(ns.vol_w > a) & (ns.vol_w <= b)]
        brow.append([f"{100*a:.1f}-{100*b:.1f}%", f"{len(s):,}", f"{100*s.mer6.mean():.0f}%", f"{s.permno.nunique():,}"])
    t_bins = table(["winsorized vol", "stock-months", "acquired within 6 months", "securities"], brow, "narrow")
    for cut in (0.05, 0.08, 0.10):
        s = ns[ns.vol_w < cut]
        N[f"cut{int(cut*100)}"] = dict(n=len(s), share=len(s) / len(ns), mer=s.mer6.mean(), permnos=s.permno.nunique())
    N["mer_base"] = ns.mer6.mean()
    # industries of the stock-months between 8% and 10% that are not acquired soon after
    mid8 = ns[(ns.vol_w >= CUT) & (ns.vol_w < 0.10) & ~ns.mer6]
    ci = c[["permno", "secinfostartdt", "secinfoenddt", "siccd"]].copy()
    ci["s"] = pd.to_datetime(ci.secinfostartdt); ci["e"] = pd.to_datetime(ci.secinfoenddt)
    mm = mid8.merge(ci, on="permno"); mm = mm[(mm.s <= mm.formation_date) & (mm.e >= mm.formation_date)]
    sic2 = pd.to_numeric(mm.siccd, errors="coerce") // 100
    N["mid8_n"] = len(mid8); N["mid8_util"] = (sic2 == 49).mean(); N["mid8_fin"] = sic2.between(60, 64).mean()
    N["mer_11_12"] = ns[(ns.vol_w > 0.11) & (ns.vol_w <= 0.12)].mer6.mean()
    N["mer_gt14"] = ns[ns.vol_w > 0.14].mer6.mean()
    N["spac_med_vol"] = base[base.spac].vol_w.median(); N["spac_n"] = int(base.spac.sum()); N["spac_permnos"] = base[base.spac].permno.nunique()
    N["spac_lt8"] = (base[base.spac].vol_w < CUT).mean()

    # ---------------- daily returns: all data
    allr = {P: unique_stock_days(clean[P]) for P in P_GRID}
    orr = unique_stock_days(orig)
    rows = []
    for lab, v in [(f"cleaned, P={P:,}", allr[P]) for P in P_GRID] + [("original, P=3,000", orr)]:
        q = qrow(v)
        rows.append([lab, f"{len(v)/1e6:.2f}M"] + [fmt(x, 2) for x in q] + [fmt(100 * v.mean(), 3), fmt(100 * v.std(), 2)])
    t_ret = table(["daily returns, %", "stock-days"] + QLAB + ["mean", "sd"], rows, "wide")
    v = allr[3000]
    N["q1_new"], N["q99_new"] = np.percentile(v, [1, 99]) * 100
    N["q1_old"], N["q99_old"] = np.percentile(orr, [1, 99]) * 100
    N["n3"] = len(v); N["max3"] = 100 * v.max(); N["min3"] = 100 * v.min()
    N["gt50"] = int((np.abs(v) > 0.5).sum()); N["gt100"] = int((v > 1).sum())

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
    x = np.clip(v, -0.2, 0.2)
    ax[0].hist(x, bins=np.linspace(-0.2, 0.2, 161), density=True, color=C["blue"], lw=0)
    sd_r = 1.4826 * np.median(np.abs(v - np.median(v)))
    g = np.linspace(-0.2, 0.2, 400)
    ax[0].plot(g, np.exp(-0.5 * (g / sd_r) ** 2) / (sd_r * np.sqrt(2 * np.pi)), color=C["orange"], lw=1, ls="--", label=f"normal, sd = robust scale ({100*sd_r:.1f}%)")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-3, None)
    ax[0].set_xlabel("daily return (outer bins collect |r| > 20%)"); ax[0].set_ylabel("density (log scale)")
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0].legend(loc="lower center"); title(ax[0], "(a) all stock-days, cleaned P=3,000")
    def ccdf(vals, sgn, xs):
        t = np.sort(sgn * vals)                    # share of stock-days with sgn * r > x
        return (len(t) - np.searchsorted(t, xs, side="right")) / len(t)
    for sgn, col, lab in ((1, C["green"], "gains  r > x"), (-1, C["orange"], "losses  -r > x")):
        xs = np.logspace(-3, np.log10((sgn * v).max()), 300)
        ax[1].loglog(xs, ccdf(v, sgn, xs), color=col, label=lab)
    xs = np.logspace(-3, np.log10(orr.max()), 300)
    ax[1].loglog(xs, ccdf(orr, 1, xs), color=C["gray"], lw=1, ls="--", label="gains, original panels")
    ax[1].axvline(0.5, color=INK, lw=0.8, ls=":")
    ax[1].set_xlabel("x (log scale; dotted line: the ±50% clip)"); ax[1].set_ylabel("share of stock-days")
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax[1].legend(loc="lower left"); title(ax[1], "(b) tails")
    f_ret = save(fig, "fig_returns_all.svg")

    # ---------------- individual 63-day windows
    pick = ["2006-01-31", "2008-10-31", "2014-07-31", "2020-03-31", "2021-02-26", "2025-11-28"]
    keys = sorted(clean[3000])
    sel = [min(keys, key=lambda k: abs(k - pd.Timestamp(p))) for p in pick]
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.7), sharex=True)
    wrows = []
    for ax, fd in zip(axs.ravel(), sel):
        X = clean[3000][fd]["X"].ravel()
        ax.hist(np.clip(X, -0.15, 0.15), bins=np.linspace(-0.15, 0.15, 121), density=True, color=C["blue"], lw=0)
        ax.set_yscale("log"); ax.set_ylim(1e-3, 80); ax.set_xlim(-0.21, 0.21)
        q1, q99 = np.percentile(X, [1, 99])
        ax.axvline(q1, color=C["orange"], lw=0.8, ls="--"); ax.axvline(q99, color=C["orange"], lw=0.8, ls="--")
        title(ax, pd.Timestamp(fd).strftime("window ending %b %Y"))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        wrows.append([pd.Timestamp(fd).strftime("%b %Y")] + [fmt(x, 2) for x in np.percentile(X, QS) * 100]
                     + [fmt(100 * np.median(ann_vol(clean[3000][fd]["X"], True)), 1)])
    for ax in axs[:, 0]:
        ax.set_ylabel("density (log)")
    fig.supxlabel("daily return (the outer bins collect |r| > 15%)", fontsize=8)
    f_win = save(fig, "fig_windows.svg")
    t_win = table(["window", *QLAB, "median stock vol (w), %"], wrows, "wide")

    # ---------------- single stocks: what the screens remove (anonymous examples)
    ob = el.merge(pd.DataFrame([(fd, p) for fd, d in orig.items() for p in d["permnos"]], columns=["formation_date", "permno"]),
                  on=["formation_date", "permno"])
    def closest(frame, target):
        return frame.iloc[(frame.vol_w - target).abs().argsort().iloc[0]]
    ex_spac = closest(ob[ob.spac], ob[ob.spac].vol_w.median())
    tgt = ob[~ob.spac & (ob.vol_w < CUT) & ob.mer6]
    ex_tgt = closest(tgt, tgt.vol_w.median())
    kept = ob[~ob.spac & (ob.vol_w >= CUT)]
    ex_typ = closest(kept, kept.vol_w.median())
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 1.9))
    for ax, row, lab in zip(axs, (ex_spac, ex_tgt, ex_typ), ("SPAC, removed", "acquisition target, removed", "typical stock, kept")):
        d = orig[row.formation_date]; j = int(np.flatnonzero(d["permnos"] == row.permno)[0])
        r = d["X"][:, j]
        ax.hist(r, bins=25, color=C["orange"] if "removed" in lab else C["blue"], lw=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1 if np.abs(r).max() < 0.05 else 0))
        title(ax, f"{lab}\nvol {100*row.vol_w:.1f}%, window ending {pd.Timestamp(row.formation_date):%b %Y}")
    axs[0].set_ylabel("days")
    fig.supxlabel("daily return over the 63-day window", fontsize=8)
    f_single = save(fig, "fig_single.svg")

    # ---------------- per-stock volatility in the cleaned panels
    vrows, qv = [], {}
    for lab, pans in (("cleaned P=3,000", clean[3000]), ("original P=3,000", orig)):
        vr = np.concatenate([ann_vol(d["X"], False) for d in pans.values()])
        vw = np.concatenate([ann_vol(d["X"], True) for d in pans.values()])
        qv[lab] = np.percentile(vr, QS[2:10]) * 100          # 5th to 99th percentiles
        if lab.startswith("original"):
            N["old_lt8"] = (vw < CUT).mean()
        for lab2, vv in (("raw", vr), ("winsorized 1%/99%", vw)):
            vrows.append([f"{lab}, {lab2}", f"{len(vv):,}"] + [fmt(x, 1) for x in np.percentile(vv, QS) * 100])
    t_vol = table(["annualized vol of each stock-window, %", "stock-windows"] + QLAB, vrows, "wide")
    dv = qv["cleaned P=3,000"] - qv["original P=3,000"]
    N["vshift_lo"], N["vshift_hi"] = float(dv.min()), float(dv.max())

    # ---------------- per-stock examples (anonymous)
    json.dump({k: (v if not isinstance(v, dict) else v) for k, v in N.items()}, open(OUTD / "numbers.json", "w"), indent=1, default=float)
    write_html(N, t_excl, t_bins, t_ret, t_win, t_vol, f_rem, f_vol, f_ret, f_win, f_single)


def write_html(N, t_excl, t_bins, t_ret, t_win, t_vol, f_rem, f_vol, f_ret, f_win, f_single):
    c5, c8, c10 = N["cut5"], N["cut8"], N["cut10"]
    body = f"""
<h1>The cleaned sample: SPACs and low-volatility stocks removed</h1>
<p class="sub">Data note for the 28 September 2026 rerun of the outline (overleaf_upload_20260928). CRSP daily returns, 239 month-ends from January 2006 to November 2025, 63-day windows, top-P universes with P = 500, 1,000, 2,000 and 3,000.</p>

<h2>1. What was removed and why</h2>
<p>Two kinds of security move very little and can bias the analysis. <b>SPACs</b> (special purpose acquisition companies) hold cash in trust until they merge, so their shares trade near $10 with almost no movement. <b>Pending acquisition targets</b> trade pinned to the deal price between the announcement and the closing. Both enter the top-P universes by market capitalization, the SPACs mainly in 2007&ndash;2009 and 2020&ndash;2023 (Figure 1). Both screens are applied when a universe is formed: the excluded company is replaced by the next-largest eligible company, so every universe still holds exactly P companies and the universes stay nested. With neither screen, the rebuilt selection reproduces the paper's original universes exactly in all 956 month-size cells.</p>
<p><b>SPACs.</b> A security counts as a SPAC on a formation date if its CRSP issuer name has the form of a SPAC name (&ldquo;&hellip; Acquisition Corp&rdquo;, &ldquo;&hellip; Merger Corp&rdquo;, &hellip;), or if it is linked to a SPAC IPO in SDC New Issues by its CUSIP, or carries CRSP SIC 6770 (blank checks), started trading as a SPAC and still has the name under which it listed. The merger with the target renames the company, which ends the SPAC period; the merged company stays in the sample. CRSP's SIC code alone would miss almost every SPAC after 2010, because it codes most of them 6799. At P = 3,000 the rule removes {N['spac_n']:,} stock-months ({N['spac_permnos']:,} securities), with a peak of {N['spac_peak']} companies in {N['spac_peak_m']}. Their median winsorized volatility is {100*N['spac_med_vol']:.1f}%, and {100*N['spac_lt8']:.0f}% of them would also fail the volatility screen.</p>
<p><b>Low volatility.</b> For every eligible stock and window, each of its 63 daily returns is clipped at that stock's own 1st and 99th percentile (winsorization), and the volatility of the result is annualized with &radic;252. A stock is removed from that month's universe when this volatility is below <b>8%</b>. Winsorizing matters for acquisition targets: the jump on the announcement day would otherwise keep their volatility high for the rest of the window.</p>
{t_excl}
<figure><img src="{f_rem}"><figcaption>Figure 1. Companies removed from the original top-3,000 in each formation month (a), and the share of each universe that is replaced (b).</figcaption></figure>

<h2>2. Choosing the volatility cutoff</h2>
<p>Figure 2a shows the winsorized volatility of every stock-month in the original top-3,000 universes. The main body starts rising steeply at about 8%. Below it lies a separate, nearly flat cluster of about 2,000 non-SPAC stock-months, plus the SPACs. To see what this cluster is, each stock-month is matched with CRSP's delisting record: in the cluster, most companies are delisted through a merger within six months, against {100*N['mer_base']:.1f}% of all non-SPAC stock-months (Figure 2b and the table). That share falls below one half between 8% and 9%, to {100*N['mer_11_12']:.0f}% between 11% and 12%, and to {100*N['mer_gt14']:.1f}% above 14%. The cutoff is therefore set at <b>8%</b>, the upper edge of the low-volatility cluster.</p>
<ul>
<li>Below 5%: {c5['n']:,} non-SPAC stock-months ({100*c5['share']:.2f}%), {100*c5['mer']:.0f}% of them acquired within six months.</li>
<li>Below 8% (chosen): {c8['n']:,} ({100*c8['share']:.2f}%), {100*c8['mer']:.0f}% acquired within six months.</li>
<li>Below 10%: {c10['n']:,} ({100*c10['share']:.2f}%), only {100*c10['mer']:.0f}% acquired within six months. Of the {N['mid8_n']:,} stock-months between 8% and 10% whose company is not acquired soon after, {100*N['mid8_util']:.0f}% are utilities (SIC 49) and {100*N['mid8_fin']:.0f}% banks and insurers (SIC 60&ndash;64): ordinary low-risk firms in calm periods.</li>
</ul>
<figure><img src="{f_vol}"><figcaption>Figure 2. (a) Winsorized 63-day volatility of every stock-month in the original top-3,000 universes (log scales); solid line, the 8% cutoff; dotted lines, 5% and 10%. (b) Among non-SPAC stock-months, the share delisted through a merger within six months of the formation date, by volatility bin; dashed line, the share over all stock-months.</figcaption></figure>
{t_bins}
<p>Figure 3 shows the 63 daily returns of three single stocks: a SPAC, a pending acquisition target and a typical stock, each chosen as the stock-month nearest the median winsorized volatility of its group.</p>
<figure><img src="{f_single}"><figcaption>Figure 3. Daily returns of single stocks over one 63-day window (original panels). Left and middle: removed by the screens; right: kept.</figcaption></figure>

<h2>3. Daily returns in the cleaned panels</h2>
<p>The table below uses each stock-day once, even when it falls in several overlapping windows. At P = 3,000 there are {N['n3']/1e6:.1f} million distinct stock-day returns. The ±50% clip that the paper applies before PCA is <i>not</i> applied here: these are the untreated returns. {N['gt50']:,} stock-days lie beyond ±50%, and {N['gt100']:,} are above +100%. The largest return is {N['max3']:,.0f}% and the smallest {N['min3']:.0f}%. Against the original P = 3,000 panels (last row), the distribution widens slightly: the 1st and 99th percentiles move from {N['q1_old']:.2f}% and {N['q99_old']:.2f}% to {N['q1_new']:.2f}% and {N['q99_new']:.2f}%, because the companies that replace the SPACs and targets move more than they do.</p>
{t_ret}
<figure><img src="{f_ret}"><figcaption>Figure 4. (a) Histogram of all distinct daily returns of the cleaned P = 3,000 panels (log density), with a normal density at the robust scale for comparison. (b) Share of stock-days with a gain (or loss) larger than x, log-log; dashed, gains in the original panels (almost indistinguishable).</figcaption></figure>

<h3>Individual 63-day windows</h3>
<p>Figure 5 shows the pooled returns of single formation panels at P = 3,000 (63 &times; 3,000 entries each): the first and last months, a typical month, and three unusual ones (the 2008 crisis, March 2020, and February 2021, at the height of the SPAC and meme-stock episode).</p>
<figure><img src="{f_win}"><figcaption>Figure 5. Pooled daily returns of six cleaned P = 3,000 panels (log density); dashed lines, the panel's 1st and 99th percentiles.</figcaption></figure>
{t_win}

<h2>4. Stock volatilities in the cleaned panels</h2>
<p>Annualized 63-day volatility of every stock in every P = 3,000 panel, raw and after the 1%/99% winsorization used by the screen. After cleaning, no stock-window has winsorized volatility below 8%, against {N['old_lt8']:.1%} of stock-windows in the original panels. From the 5th percentile up, the cleaned quantiles of raw volatility are {N['vshift_lo']:.1f} to {N['vshift_hi']:.1f} percentage points above the original ones.</p>
{t_vol}
<p class="note">Files: code in <code>rerun_20260928/code/</code> (spac_flags.py, r02&ndash;r04, r06); cleaned universes in <code>emp/data/processed/rolling/rolling_topcap_clean_200601_202511_20260928/</code>. All numbers are aggregates.</p>
"""
    css = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:10pt;color:#0b0b0b;max-width:7.2in;margin:0 auto;line-height:1.38}
h1{font-size:15pt;margin:0 0 4pt} h2{font-size:12pt;margin:14pt 0 4pt;border-bottom:1px solid #ccc} h3{font-size:10.5pt;margin:10pt 0 3pt}
.sub{color:#52514e;font-size:9pt;margin-top:0} p{margin:4pt 0} ul{margin:3pt 0 3pt 16pt;padding:0}
table{border-collapse:collapse;font-size:7.6pt;margin:6pt 0;width:100%} th,td{padding:1.5pt 4pt;text-align:right;border-bottom:1px solid #e6e5e0}
th{border-bottom:1px solid #52514e;font-weight:600} td:first-child,th:first-child{text-align:left}
table.narrow{width:60%} figure{margin:6pt 0;page-break-inside:avoid} img{width:100%} figcaption{font-size:8pt;color:#52514e}
.note{font-size:8pt;color:#52514e;margin-top:10pt} code{font-size:8pt}
@page{size:letter;margin:0.6in}
"""
    doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Cleaned data report</title><style>{css}</style></head><body>{body}</body></html>"
    (OUTD / "data_report.html").write_text(doc, encoding="utf-8")
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    PDF.parent.mkdir(exist_ok=True)
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", (OUTD / "data_report.html").as_uri()], check=True, timeout=180)
    print("wrote", PDF)


if __name__ == "__main__":
    main()
