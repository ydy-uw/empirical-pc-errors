# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""Cross-treatment comparison: comparison/*.csv and overleaf_reports/tables/comparison/*.tex.

    python outlier_cleaning/code/compare.py
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from oc_common import EXP, CMP, STUDY, deg
from treatments import METHODS
from make_tables import S, f0, f1, f2, pct, fshare, num, describe

TAB = STUDY / "overleaf_reports" / "tables" / "comparison"
NAME = {"00_raw": "untreated", "01_fixed_clip": "fixed clip", "02_daily_winsorization": "daily winsorization",
        "03_cross_sectional_mad": "cross-sectional clip", "04_time_series_mad": "time-series clip",
        "05_log_returns": "log returns", "06_rank_transform": "rank transform",
        "07_drop_names": "drop names", "08_reversal_filter": "reversal filter"}


def share(m, T, P):
    fname = METHODS[m][0]
    t = T["touched"][str(P)]
    if fname in ("log_returns", "rank_normal", "drop_names"):
        return 100 * t["share_changed_mean"]
    return 100 * t["entries_changed"] / (239 * 63 * t["p_median"])


def series_corr(m):
    r = pd.read_csv(EXP / "00_raw" / "results" / "panels_main.csv", parse_dates=["formation_date"])
    t = pd.read_csv(EXP / m / "results" / "panels_main.csv", parse_dates=["formation_date"])
    x = r[r.n_assets == 3000].set_index("formation_date").A1
    y = t[t.n_assets == 3000].set_index("formation_date").A1
    return float(np.corrcoef(deg(x), deg(y.reindex(x.index)))[0, 1])


def main():
    TAB.mkdir(parents=True, exist_ok=True); CMP.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in METHODS:
        T = S(m)
        if T is None:
            continue
        r = dict(method=m, name=NAME[m], parameters=describe(METHODS[m][1]["main"]))
        for P in (500, 3000):
            r[f"share_{P}"] = share(m, T, P)
            r[f"k_{P}"] = T["counts"][str(P)]["skew"]["median"]
            r[f"k_equal_{P}"] = T["counts"][str(P)]["agree_with_raw"]
            r[f"peff_{P}"] = T["accuracy"][str(P)]["p_eff"]
            for j in (1, 2, 3):
                r[f"ang{j}_{P}"] = T["estimates"][str(P)][str(j)]["median_deg"]
                r[f"diff{j}_{P}"] = T["estimates"][str(P)][str(j)]["paired_diff_deg"]
                r[f"pass{j}_{P}"] = T["screen"][str(P)][str(j)]["pass_share"]
                r[f"loc{j}_{P}"] = T["screen"][str(P)][str(j)]["localized"]
            r[f"S_{P}"] = T["estimates"][str(P)]["S_over_k_pct"]
            r[f"kurt_{P}"] = T["accuracy"][str(P)]["kurt_pooled"]
            r[f"block_{P}"] = T["blocks"][str(P)]["own_k"]
            r[f"block1_{P}"] = T["blocks"][str(P)]["m1"]
            r[f"fa_nonconv_{P}"] = T["fa"][str(P)]["nonconverged"]
            r[f"afa3_{P}"] = T["fa"][str(P)]["3"]["Afa_deg"]
            r[f"oos_{P}"] = T["oos"][str(P)]["viol_simple_A"]
            r[f"sd1_{P}"] = T["estimates"][str(P)]["1"]["sd_deg"]
            r[f"bias3_{P}"] = T["estimates"][str(P)]["3"]["bias_deg"]
        r["bn_3000"] = T["counts"]["3000"]["bn"]["median"]
        r["kchange_3000"] = T["counts"]["3000"]["changes_month_to_month"]
        r["left_3000"] = T["touched"]["3000"]["above_100_left"]
        r["dropped_3000"] = T["touched"]["3000"]["names_dropped_total"]
        r["gfc_3000"] = T["crises"]["3000"]["gfc"]["A1_deg"]
        r["covid_3000"] = T["crises"]["3000"]["covid"]["A1_deg"]
        r["corr_A1_3000"] = series_corr(m)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(CMP / "main_settings.csv", index=False)

    # every setting, for the dose-response view
    allrows = []
    for m in METHODS:
        for f in sorted((EXP / m / "results").glob("summary_*.json")):
            s = f.stem.replace("summary_", ""); T = json.load(open(f))
            allrows.append(dict(method=m, setting=s, parameters=describe(METHODS[m][1][s]),
                                share_3000=share(m, T, 3000), k_500=T["counts"]["500"]["skew"]["median"],
                                k_3000=T["counts"]["3000"]["skew"]["median"],
                                k_equal_3000=T["counts"]["3000"]["agree_with_raw"],
                                **{f"ang{j}_3000": T["estimates"]["3000"][str(j)]["median_deg"] for j in (1, 2, 3)},
                                **{f"pass{j}_3000": T["screen"]["3000"][str(j)]["pass_share"] for j in (1, 2, 3)},
                                peff_3000=T["accuracy"]["3000"]["p_eff"], kurt_3000=T["accuracy"]["3000"]["kurt_pooled"]))
    pd.DataFrame(allrows).to_csv(CMP / "all_settings.csv", index=False)

    def lab(r):
        return NAME[r.method]

    L = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
         r"& & & \multicolumn{2}{c}{entries changed (\%)} & panels & $>{+}100\%$\\",
         r"\cmidrule(lr){4-5}",
         r"report & treatment & baseline setting & $P=500$ & $P=3{,}000$ & touched & left\\", r"\midrule"]
    for _, r in df.iterrows():
        T = S(r.method)
        L.append(f"{r.method[:2]} & {lab(r)} & {r.parameters} & {fshare(r.share_500)} & {fshare(r.share_3000)} & "
                 f"{T['touched']['3000']['panels_touched'] if r.method != '00_raw' else '--'} & {num(r.left_3000)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(TAB / "changes.tex", "w").write("\n".join(L) + "\n")

    L = [r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{skewness count} & \multicolumn{2}{c}{equal to untreated (\%)} & Bai--Ng & changes & \multicolumn{2}{c}{leading angle, crises}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){8-9}",
         r"treatment & $P=500$ & $P=3{,}000$ & $P=500$ & $P=3{,}000$ & $P=3{,}000$ & $P=3{,}000$ (\%) & 2008--09 & 2020\\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(f"{lab(r)} & {f0(r.k_500)} & {f0(r.k_3000)} & {pct(r.k_equal_500)} & {pct(r.k_equal_3000)} & {f0(r.bn_3000)} & "
                 f"{pct(r.kchange_3000)} & {f1(r.gfc_3000)} & {f1(r.covid_3000)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(TAB / "counts.tex", "w").write("\n".join(L) + "\n")

    L = [r"\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{median angle (deg.)} & \multicolumn{2}{c}{median change} & $S/k$ & \multicolumn{3}{c}{pass the screen (\%)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){8-10}",
         r"treatment & $j=1$ & $j=2$ & $j=3$ & $j=1$ & $j=3$ & (\%) & $j=1$ & $j=2$ & $j=3$\\", r"\midrule"]
    for _, r in df.iterrows():
        d1 = "--" if r.method == "00_raw" else f"${r.diff1_3000:+.2f}$"
        d3 = "--" if r.method == "00_raw" else f"${r.diff3_3000:+.2f}$"
        L.append(f"{lab(r)} & {f1(r.ang1_3000)} & {f1(r.ang2_3000)} & {f1(r.ang3_3000)} & {d1} & {d3} & {f1(r.S_3000)} & "
                 f"{pct(r.pass1_3000)} & {pct(r.pass2_3000)} & {pct(r.pass3_3000)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(TAB / "estimates.tex", "w").write("\n".join(L) + "\n")

    L = [r"\footnotesize\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{$\peff$} & kurtosis & sd, $j=1$ & bias, $j=3$ & block & fits not & $\Afa$, $j=3$\\",
         r"\cmidrule(lr){2-3}",
         r"treatment & $P=500$ & $P=3{,}000$ & pooled & (deg.) & (deg.) & (\%) & converged & (deg.)\\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(f"{lab(r)} & {f0(r.peff_500)} & {f0(r.peff_3000)} & {f1(r.kurt_3000)} & {f2(r.sd1_3000)} & ${r.bias3_3000:.2f}$ & "
                 f"{f2(r.block_3000)} & {r.fa_nonconv_3000} & {f1(r.afa3_3000)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(TAB / "accuracy.tex", "w").write("\n".join(L) + "\n")
    # candidate specifications: settings that pass the screen in most months at P = 3,000
    cand = [("00_raw", "main", "untreated"), ("01_fixed_clip", "main", r"fixed clip, $\pm100\%$"),
            ("01_fixed_clip", "c050", r"fixed clip, $\pm50\%$"), ("02_daily_winsorization", "main", r"daily winsorization, 0.1\%"),
            ("03_cross_sectional_mad", "c20", r"cross-sectional clip, $c=20$"),
            ("04_time_series_mad", "prior_c10", r"time-series clip, prior scale"),
            ("07_drop_names", "main", r"drop names, $+100\%$")]
    al = pd.read_csv(CMP / "all_settings.csv")
    L = [r"\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrcccccccc@{}}", r"\toprule",
         r"& entries & count & \multicolumn{3}{c}{median angle (deg.)} & pass & & fits & block\\",
         r"\cmidrule(lr){4-6}",
         r"treatment and setting & (\%) & equal (\%) & $j=1$ & $j=2$ & $j=3$ & $j=2/3$ (\%) & $\peff$ & failed & (\%)\\", r"\midrule"]
    for m, st, label in cand:
        r = al[(al.method == m) & (al.setting == st)].iloc[0]
        T = S(m, st)
        nc = str(T["fa"]["3000"]["nonconverged"]) if "fa" in T else "--"
        bl = f2(T["blocks"]["3000"]["own_k"]) if "blocks" in T else "--"
        L.append(f"{label} & {fshare(r.share_3000)} & {pct(r.k_equal_3000)} & "
                 f"{f1(r.ang1_3000)} & {f1(r.ang2_3000)} & {f1(r.ang3_3000)} & {pct(r.pass2_3000)}/{pct(r.pass3_3000)} & {f0(r.peff_3000)} & {nc} & {bl}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(TAB / "candidates.tex", "w").write("\n".join(L) + "\n")
    print(df[["method", "share_3000", "k_3000", "k_equal_3000", "ang1_3000", "ang3_3000", "pass2_3000", "pass3_3000",
              "peff_3000", "kurt_3000", "fa_nonconv_3000", "corr_A1_3000"]].round(3).to_string())


if __name__ == "__main__":
    main()
