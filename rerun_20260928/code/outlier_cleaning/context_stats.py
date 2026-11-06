# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""Descriptive numbers quoted in the reports' text that are not in any table.

    python outlier_cleaning/code/context_stats.py

Writes comparison/context.json and overleaf_reports/tables/context.tex (macros \\ctx...).
Reads the formation panels (aggregates only are written) and the experiment results.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from oc_common import EXP, CMP, STUDY, P_GRID, deg, dump
import panels, treatments


def load(m, s="main"):
    return pd.read_csv(EXP / m / "results" / f"panels_{s}.csv", parse_dates=["formation_date"])


def main():
    c = {}
    mins, sd_name, xs_scale, name_scale = {}, {}, {}, {}
    for P in P_GRID:
        d = panels.load(P)
        mn, sdm, xs, q = [], [], [], []
        for fd, p in d.items():
            X = p["X"]
            mn.append(X.min())
            sdm.append(np.median(X.std(axis=0)))
            med = np.median(X, axis=1, keepdims=True)
            xs.append(np.median(1.4826 * np.median(np.abs(X - med), axis=1)))
            _, s, _ = treatments._robust_col_scale(X)
            q.append(np.quantile(s, [0.1, 0.5, 0.9]))
        mins[P] = float(np.min(mn)); sd_name[P] = float(np.median(sdm)); xs_scale[P] = float(np.median(xs))
        name_scale[P] = np.median(np.array(q), axis=0).tolist()
        panels.drop(P)
    c["min_return"] = min(mins.values())
    c["median_name_sd"] = sd_name; c["xs_scale"] = xs_scale; c["name_scale_q10_q50_q90"] = name_scale

    r = load("00_raw"); cl = load("01_fixed_clip"); dr = load("07_drop_names")
    g = r[r.n_assets == 3000]
    c["panels_with_ext_3000"] = int((g.n_above_100 > 0).sum())
    x = cl.merge(r, on=["formation_date", "n_assets"], suffixes=("", "_r"))
    x = x[x.n_assets == 3000]
    for j in (2, 3):
        h = x[x[f"loc{j}"] & (x[f"k_skew_r"] >= j)]
        c[f"clip_still_loc{j}"] = int(len(h))
        c[f"clip_still_loc{j}_at_cap"] = int((h.max_abs >= 1.0 - 1e-12).sum())
        y = cl.merge(dr, on=["formation_date", "n_assets"], suffixes=("", "_d")).merge(
            r[["formation_date", "n_assets", "k_skew"]].rename(columns={"k_skew": "k_raw"}), on=["formation_date", "n_assets"])
        y = y[(y.n_assets == 3000) & y[f"loc{j}"] & (y.k_raw >= j)]
        c[f"clip_loc{j}_pass_after_drop"] = int((~y[f"loc{j}_d"]).sum()); c[f"clip_loc{j}_all"] = int(len(y))
    for s in ("main", "up050"):
        t = load("08_reversal_filter", s)
        c[f"reversal_pairs_{s}"] = int(t[t.n_assets == 3000].pairs_flagged.sum())
        c[f"reversal_entries_{s}"] = int(t[t.n_assets == 3000].n_changed.sum())
    for m in ("03_cross_sectional_mad", "04_time_series_mad"):
        t = load(m); t = t[t.n_assets == 3000].set_index("formation_date").share_changed
        c[f"share_{m[:2]}"] = dict(all=float(t.mean()), gfc=float(t["2008-09":"2009-06"].mean()),
                                   covid=float(t["2020-02":"2020-06"].mean()))
    for m in ("05_log_returns", "06_rank_transform"):
        t = load(m)
        c[f"share_changed_{m[:2]}"] = {str(P): float(t[t.n_assets == P].share_changed.mean()) for P in P_GRID}
    rk = load("06_rank_transform").merge(r, on=["formation_date", "n_assets"], suffixes=("", "_r"))
    rk = rk[rk.n_assets == 3000]
    dd = pd.Series(deg(rk.A1) - deg(rk.A1_r), index=rk.index)
    c["rank_lower_share"] = float((dd < 0).mean())
    c["rank_by_year"] = {str(k): float(v) for k, v in dd.groupby(rk.formation_date.dt.year).median().items()}
    w = load("02_daily_winsorization").merge(r, on=["formation_date", "n_assets"], suffixes=("", "_r"))
    w = w[w.n_assets == 3000]; dw = pd.Series(deg(w.A1) - deg(w.A1_r), index=w.index)
    c["winsor_lower_share"] = float((dw < 0).mean()); c["winsor_lower_1deg"] = float((dw < -1).mean())
    # eigenvalue ratios (treated / untreated), bulk level at the untreated count, medians
    ratio = {}
    for m, tag in (("01_fixed_clip", "Clip"), ("02_daily_winsorization", "Win"), ("03_cross_sectional_mad", "Xs"),
                   ("04_time_series_mad", "Ts"), ("05_log_returns", "Log"), ("06_rank_transform", "Rank")):
        t = load(m).merge(r, on=["formation_date", "n_assets"], suffixes=("", "_r"))
        for P, w in ((500, "Five"), (3000, "Three")):
            g = t[t.n_assets == P]
            ell_ref = g.A1_ref * g.theta1
            ratio[f"{tag}{w}"] = dict(ell=float((ell_ref / g.ell_r).median()), th1=float((g.theta1 / g.theta1_r).median()))
    c["ratios"] = ratio
    # report 01: the mechanism in localized months
    x = cl.merge(r, on=["formation_date", "n_assets"], suffixes=("", "_r"))
    g = x[(x.n_assets == 3000) & (x.k_skew_r >= 2)]
    locm, oth = g[g.loc2_r], g[~g.loc2_r]
    c["clip_mech"] = dict(months=int(len(locm)), th2_loc=float((locm.theta2 / locm.theta2_r).median()),
                          ell_loc=float((locm.ell / locm.ell_r).median()), th2_other=float((oth.theta2 / oth.theta2_r).median()))
    g1 = x[x.n_assets == 3000]
    d1 = pd.Series(deg(g1.A1) - deg(g1.A1_r), index=g1.index)
    c["clip_lead_max_nonloc"] = float(d1[~g1.loc1_r].abs().max())
    for j in (2, 3):
        h = x[(x.n_assets == 3000) & (x.k_skew_r >= j) & x[f"loc{j}_r"]]
        c[f"clip_rescued{j}"] = [int((~h[f"loc{j}"]).sum()), int(len(h))]
    # report 03: where the cross-sectional clip acts, by size decile (P = 3,000)
    d = panels.load(3000); cnt = np.zeros(10)
    for fd, p in d.items():
        X = p["X"]; Xt, _, _ = treatments.xs_mad(X, p, c=10.0)
        dec = np.minimum((np.arange(X.shape[1]) * 10) // X.shape[1], 9)
        np.add.at(cnt, dec, (Xt != X).sum(axis=0))
    panels.drop(3000)
    c["xs_decile_share"] = (cnt / cnt.sum()).tolist()
    # report 04: months still localized in direction 2 or 3 (screen definition: untreated count reaches j)
    t = load("04_time_series_mad").merge(r[["formation_date", "n_assets", "k_skew"]].rename(columns={"k_skew": "k_raw"}),
                                          on=["formation_date", "n_assets"])
    t = t[t.n_assets == 3000]
    still = t[(t.loc2 & (t.k_raw >= 2)) | (t.loc3 & (t.k_raw >= 3))]
    c["ts_still"] = dict(months=int(len(still)), max_abs_median=float(still.max_abs.median()),
                         share_with_ext=float((still.n_above_100 > 0).mean()))
    # report 08: leading-angle changes
    rv = load("08_reversal_filter").merge(r, on=["formation_date", "n_assets"], suffixes=("", "_r"))
    rv = rv[rv.n_assets == 3000]
    dv = pd.Series(deg(rv.A1) - deg(rv.A1_r), index=rv.index)
    big = dv[dv.abs() > 0.45]
    c["rev_lead"] = dict(months_big=int(len(big)), max=float(dv.max()),
                         other_touched_max=float(dv[(rv.n_changed > 0) & (dv.abs() <= 0.45)].abs().max()))
    dump(c, CMP / "context.json")
    M = {
        "ctxMinRet": f"{100 * c['min_return']:.1f}",
        "ctxSdFive": f"{100 * sd_name[500]:.1f}", "ctxSdThree": f"{100 * sd_name[3000]:.1f}",
        "ctxXsFive": f"{100 * xs_scale[500]:.1f}", "ctxXsThree": f"{100 * xs_scale[3000]:.1f}",
        "ctxNameMedFive": f"{100 * name_scale[500][1]:.1f}", "ctxNameMedThree": f"{100 * name_scale[3000][1]:.1f}",
        "ctxNameHiFive": f"{100 * name_scale[500][2]:.1f}", "ctxNameHiThree": f"{100 * name_scale[3000][2]:.1f}",
        "ctxPanelsExt": str(c["panels_with_ext_3000"]),
        "ctxStillTwo": str(c["clip_still_loc2"]), "ctxStillTwoCap": str(c["clip_still_loc2_at_cap"]),
        "ctxStillThree": str(c["clip_still_loc3"]), "ctxStillThreeCap": str(c["clip_still_loc3_at_cap"]),
        "ctxDropFixTwo": str(c["clip_loc2_pass_after_drop"]), "ctxDropAllTwo": str(c["clip_loc2_all"]),
        "ctxDropFixThree": str(c["clip_loc3_pass_after_drop"]), "ctxDropAllThree": str(c["clip_loc3_all"]),
        "ctxRevPairs": str(c["reversal_pairs_main"]), "ctxRevEntries": str(c["reversal_entries_main"]),
        "ctxRevPairsLow": str(c["reversal_pairs_up050"]), "ctxRevEntriesLow": f"{c['reversal_entries_up050']:,}".replace(",", "{,}"),
        "ctxXsShareGfc": f"{100 * c['share_03']['gfc']:.2f}", "ctxXsShareAll": f"{100 * c['share_03']['all']:.2f}",
        "ctxTsShareGfc": f"{100 * c['share_04']['gfc']:.3f}", "ctxTsShareCovid": f"{100 * c['share_04']['covid']:.2f}",
        "ctxTsShareAll": f"{100 * c['share_04']['all']:.2f}",
        "ctxLogShareFive": f"{100 * c['share_changed_05']['500']:.1f}", "ctxLogShareThree": f"{100 * c['share_changed_05']['3000']:.1f}",
        "ctxRankShareThree": f"{100 * c['share_changed_06']['3000']:.2f}",
        "ctxRankLower": f"{100 * c['rank_lower_share']:.0f}",
        "ctxWinLower": f"{100 * c['winsor_lower_share']:.0f}", "ctxWinLowerOne": f"{100 * c['winsor_lower_1deg']:.0f}",
        "ctxMechMonths": str(c["clip_mech"]["months"]), "ctxMechThTwo": f"{100 * (1 - c['clip_mech']['th2_loc']):.0f}",
        "ctxMechEll": f"{100 * (1 - c['clip_mech']['ell_loc']):.1f}", "ctxMechOther": f"{100 * abs(1 - c['clip_mech']['th2_other']):.1f}",
        "ctxClipLeadMax": f"{c['clip_lead_max_nonloc']:.1f}",
        "ctxRescTwo": str(c["clip_rescued2"][0]), "ctxRescTwoAll": str(c["clip_rescued2"][1]),
        "ctxRescThree": str(c["clip_rescued3"][0]), "ctxRescThreeAll": str(c["clip_rescued3"][1]),
        "ctxXsDecOne": f"{100 * c['xs_decile_share'][0]:.1f}", "ctxXsDecTen": f"{100 * c['xs_decile_share'][9]:.0f}",
        "ctxTsStill": str(c["ts_still"]["months"]), "ctxTsStillMax": f"{100 * c['ts_still']['max_abs_median']:.0f}",
        "ctxTsStillExt": f"{100 * c['ts_still']['share_with_ext']:.0f}",
        "ctxRevBig": str(c["rev_lead"]["months_big"]), "ctxRevMax": f"{c['rev_lead']['max']:.1f}",
        "ctxRevOther": f"{c['rev_lead']['other_touched_max']:.1f}",
    }
    for key, v in c["ratios"].items():
        M[f"ctxEll{key}"] = f"{100 * (1 - v['ell']):.0f}"
        M[f"ctxTh{key}"] = f"{100 * (1 - v['th1']):.1f}"
    lines = ["% generated by outlier_cleaning/code/context_stats.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    open(STUDY / "overleaf_reports" / "tables" / "context.tex", "w").write("\n".join(lines) + "\n")
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
