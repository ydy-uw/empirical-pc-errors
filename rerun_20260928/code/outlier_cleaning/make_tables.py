# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""LaTeX tables and number macros for the technical reports.

    python outlier_cleaning/code/make_tables.py

Writes overleaf_reports/tables/<method>/{changes,counts,estimates,screen,accuracy,
dayvar,sweep}.tex and numbers.tex (macros quoted in the text), and
tables/comparison/*.tex.  Everything comes from experiments/*/results/summary_*.json.
In each table the untreated value is shown first (gray) and the treated value second.
"""
from __future__ import annotations
import json
import numpy as np
from oc_common import EXP, STUDY, P_GRID
from treatments import METHODS

TAB = STUDY / "overleaf_reports" / "tables"
PNAME = {500: "500", 1000: "1{,}000", 2000: "2{,}000", 3000: "3{,}000"}
WORD = {500: "Five", 1000: "Thousand", 2000: "TwoThousand", 3000: "Three"}   # macro stems
JW = {1: "One", 2: "Two", 3: "Three"}


def S(m, s="main"):
    f = EXP / m / "results" / f"summary_{s}.json"
    return json.load(open(f)) if f.is_file() else None


def g(x):   # untreated value, typeset grey
    return r"\textcolor{black!55}{" + x + "}"


def f0(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.0f}".replace(",", "{,}")
def f1(x): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}"
def f2(x): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
def sg(x, d=2):
    if x is None:
        return "--"
    if round(x, d) == 0:
        return f"${0:.{d}f}$"
    return f"${x:+.{d}f}$".replace("+", "{+}")
def pct(x, d=0): return "--" if x is None else f"{100*x:.{d}f}"
def num(x): return f"{int(x):,}".replace(",", "{,}")
def fshare(x):
    if x == 0:
        return "0"
    if x < 0.0001:
        return "$<$0.0001"
    v = round(x, 4)
    return f"{v:.4f}" if v < 0.01 else (f"{v:.3f}" if v < 0.995 else f"{v:.1f}")


def _q(v): return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
def kq(c): return f"{_q(c['median'])} [{_q(c['q25'])}--{_q(c['q75'])}]"


NARROW = {"changes": r"\setlength{\tabcolsep}{4pt}", "counts": r"\setlength{\tabcolsep}{4.5pt}",
          "estimates": r"\footnotesize\setlength{\tabcolsep}{4pt}", "dayvar": r"\footnotesize\setlength{\tabcolsep}{3pt}",
          "accuracy": r"\setlength{\tabcolsep}{5pt}", "sweep": r"\footnotesize\setlength{\tabcolsep}{2.5pt}"}


def write(m, name, lines):
    if name in NARROW and lines and lines[0].startswith(r"\begin{tabular}"):
        lines = [NARROW[name]] + lines
    d = TAB / m; d.mkdir(parents=True, exist_ok=True)
    open(d / f"{name}.tex", "w").write("\n".join(lines) + "\n")


def t_changes(m, T, R):
    fname = METHODS[m][0]
    lab = {"log_returns": "moved $>$1 point", "drop_names": "entries removed"}.get(fname, "entries changed")
    touched = "names removed" if fname == "drop_names" else "names touched"
    left = {"log_returns": "$>100$ log points", "rank_normal": "$>1$ left"}.get(fname, "$>{+}100\\%$ left")
    L = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         rf"$P$ & {lab} & share (\%) & panels touched & {touched} & names removed & {left}\\",
         r"& & & (of 239) & (median) & (total) & (\untreated)\\", r"\midrule"]
    for P in P_GRID:
        t, r = T["touched"][str(P)], R["touched"][str(P)]
        n_ch = t["entries_changed"]
        share = 100 * t["entries_changed"] / (239 * 63 * t["p_median"]) if fname != "drop_names" else 100 * t["share_changed_mean"]
        if fname == "log_returns":
            n_ch = t.get("entries_moved_1pt", n_ch); share = 100 * n_ch / (239 * 63 * P)
        L.append(f"{PNAME[P]} & {num(n_ch)} & {fshare(share)} & {t['panels_touched']} & {num(t['names_touched_median'])} & "
                 f"{num(t['names_dropped_total'])} & {num(t['above_100_left'])} ({g(num(r['above_100_left']))})\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "changes", L)


def t_counts(m, T, R):
    L = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{skewness rule} & \multicolumn{3}{c}{treated against \untreated\ (\% of months)} & \multicolumn{3}{c}{other rules, median}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-9}",
         r"$P$ & \untreated & treated & equal & higher & lower & Bai--Ng & ratio & Onatski\\", r"\midrule"]
    for P in P_GRID:
        t, r = T["counts"][str(P)], R["counts"][str(P)]
        L.append(f"{PNAME[P]} & {g(kq(r['skew']))} & {kq(t['skew'])} & {pct(t['agree_with_raw'])} & {pct(t['higher_than_raw'])} & "
                 f"{pct(t['lower_than_raw'])} & {g(f0(r['bn']['median']))}\\,/\\,{f0(t['bn']['median'])} & "
                 f"{g(f0(r['ah']['median']))}\\,/\\,{f0(t['ah']['median'])} & {g(f0(r['ed']['median']))}\\,/\\,{f0(t['ed']['median'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "counts", L)


def t_estimates(m, T, R):
    L = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"& & & \multicolumn{3}{c}{median angle} & \multicolumn{2}{c}{median change} &\\",
         r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"$P$ & $j$ & months & \untreated & treated & at \untreated\ count & all months & touched months & sd\,/\,bias\\", r"\midrule"]
    for P in P_GRID:
        for j in (1, 2, 3):
            t, r = T["estimates"][str(P)][str(j)], R["estimates"][str(P)][str(j)]
            L.append(f"{PNAME[P] if j == 1 else ''} & {j} & {t['months']} & {g(f1(r['median_deg']))} & {f1(t['median_deg'])} & "
                     f"{f1(t['median_deg_refcount'])} & {sg(t['paired_diff_deg'])} & {sg(t['paired_diff_touched_deg'])} ({t['touched_months']}) & "
                     f"{f2(t['sd_deg'])}\\,/\\,${t['bias_deg']:.2f}$\\\\")
        te, re_ = T["estimates"][str(P)], R["estimates"][str(P)]
        su = g(f1(re_['S_over_k_pct']) + "\\%")
        L.append(rf"& \multicolumn{{8}}{{l}}{{$S/k$: {su} \untreated, "
                 rf"{f1(te['S_over_k_pct'])}\% treated, {f1(te['S_over_k_refcount_pct'])}\% at the \untreated\ count}}\\")
        if P != 3000:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "estimates", L)


def t_screen(m, T, R):
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"& & & \multicolumn{2}{c}{pass (\%)} & \multicolumn{2}{c}{localized months} & participation\\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"$P$ & $j$ & months & \untreated & treated & \untreated & treated & (median, \%)\\", r"\midrule"]
    for P in P_GRID:
        for j in (1, 2, 3):
            t, r = T["screen"][str(P)][str(j)], R["screen"][str(P)][str(j)]
            L.append(f"{PNAME[P] if j == 1 else ''} & {j} & {t['months']} & {g(pct(r['pass_share'], 1))} & {pct(t['pass_share'], 1)} & "
                     f"{g(str(r['localized']))} & {t['localized']} & {g(pct(r['participation_median'], 1))}\\,/\\,{pct(t['participation_median'], 1)}\\\\")
        if P != 3000:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "screen", L)


def t_accuracy(m, T, R):
    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"$P$ & dispersion $c$ & $\peff$ & kurtosis, pooled & kurtosis, median name & bulk sd/mean\\", r"\midrule"]
    for P in P_GRID:
        t, r = T["accuracy"][str(P)], R["accuracy"][str(P)]
        L.append(f"{PNAME[P]} & {g(f2(r['cv_split']))}\\,/\\,{f2(t['cv_split'])} & {g(f0(r['p_eff']))}\\,/\\,{f0(t['p_eff'])} & "
                 f"{g(f1(r['kurt_pooled']))}\\,/\\,{f1(t['kurt_pooled'])} & {g(f1(r['kurt_median']))}\\,/\\,{f1(t['kurt_median'])} & "
                 f"{g(f2(R['bulk'][str(P)]))}\\,/\\,{f2(T['bulk'][str(P)])}\\\\")
    L += [r"\midrule", r"& \multicolumn{5}{l}{decile-1 residual block (\%), \untreated\,/\,treated}\\",
          r"$P$ & at own count & $m=1$ & $m=5$ & $m=12$ & $m=40$\\", r"\midrule"]
    for P in (500, 3000):
        t, r = T["blocks"][str(P)], R["blocks"][str(P)]
        L.append(f"{PNAME[P]} & {g(f2(r['own_k']))}\\,/\\,{f2(t['own_k'])} & {g(f2(r['m1']))}\\,/\\,{f2(t['m1'])} & "
                 f"{g(f2(r['m5']))}\\,/\\,{f2(t['m5'])} & {g(f2(r['m12']))}\\,/\\,{f2(t['m12'])} & {g(f2(r['m40']))}\\,/\\,{f2(t['m40'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "accuracy", L)


def t_dayvar(m, T, R):
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{median angle, $A_j$ / $\Afa$ (degrees)} & fits not & day-variance & co-movement\\",
         r"\cmidrule(lr){2-4}",
         r"$P$ & $j=1$ & $j=2$ & $j=3$ & converged & CV (median) & holds (\%)\\", r"\midrule"]
    for P in P_GRID:
        t, r = T["fa"][str(P)], R["fa"][str(P)]
        cells = []
        for j in (1, 2, 3):
            cells.append(f"{g(f1(r[str(j)]['A_deg']) + '/' + f1(r[str(j)]['Afa_deg']))}\;{f1(t[str(j)]['A_deg'])}/{f1(t[str(j)]['Afa_deg'])}")
        L.append(f"{PNAME[P]} & " + " & ".join(cells) +
                 f" & {g(str(r['nonconverged']))}\\,/\\,{t['nonconverged']} & {g(f2(r['day_var_cv']))}\\,/\\,{f2(t['day_var_cv'])} & "
                 f"{g(pct(r['comove_share']))}\\,/\\,{pct(t['comove_share'])}\\\\")
    L += [r"\midrule", r"& \multicolumn{6}{l}{disjoint windows: share of pairs below the simple bound (\%), \untreated\,/\,treated}\\",
          r"$P$ & pairs & with $A_j$ & with $\Afa$ & direction 1, $A_1$ & median realized & median slack\\", r"\midrule"]
    for P in (500, 3000):
        t, r = T["oos"][str(P)], R["oos"][str(P)]
        L.append(f"{PNAME[P]} & {t['pairs']} & {g(pct(r['viol_simple_A']))}\\,/\\,{pct(t['viol_simple_A'])} & "
                 f"{g(pct(r['viol_simple_FA']))}\\,/\\,{pct(t['viol_simple_FA'])} & {g(pct(r['viol_dir1_A']))}\\,/\\,{pct(t['viol_dir1_A'])} & "
                 f"{g(f2(r['realized_median']))}\\,/\\,{f2(t['realized_median'])} & {g(f2(r['slack_A_median']))}\\,/\\,{f2(t['slack_A_median'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "dayvar", L)


PARAM = {"c": "$c={}$", "q": "$q={}$", "mode": "{}", "up": "$u={}$", "rev": "$v={}$",
         "scale": "{}-window scale", "then_clip": "then clip"}
MODE = {"scaled": "scaled normal scores", "unit": "unit normal scores", "standard": "standardized, no ranks"}


def describe(params):
    if not params:
        return "--"
    return ", ".join(PARAM[k].format(MODE.get(v, v) if k == "mode" else v) for k, v in params.items()
                     if not (k == "then_clip" and v is None))


def t_sweep(m):
    fname, settings = METHODS[m]
    if len(settings) < 2:
        return
    R = S("00_raw")
    L = [r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
         r"& & entries & \multicolumn{2}{c}{count (median)} & \multicolumn{3}{c}{median angle, $P=3{,}000$} & pass, $j=2/3$\\".replace("entries", "entries" if fname != "drop_names" else "entries"),
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
         (r"& setting & changed (\%) & $P=500$ & $P=3{,}000$ & $j=1$ & $j=2$ & $j=3$ & $P=3{,}000$ (\%)\\" if fname != "drop_names"
          else r"& setting & removed (\%) & $P=500$ & $P=3{,}000$ & $j=1$ & $j=2$ & $j=3$ & $P=3{,}000$ (\%)\\"), r"\midrule"]
    rows = [("untreated", {}, R)] + [(s, p, S(m, s)) for s, p in settings.items()]
    for name, p, T in rows:
        if T is None:
            continue
        t3 = T["touched"]["3000"]
        share = 100 * t3["entries_changed"] / (239 * 63 * 3000)
        if fname in ("log_returns", "rank_normal", "drop_names") and name != "untreated":
            share = 100 * t3["share_changed_mean"]
        lab = r"\untreated" if name == "untreated" else (r"\textbf{baseline}" if name == "main" else "")
        L.append(f"{lab} & {describe(p)} & {fshare(share)} & {T['counts']['500']['skew']['median']:.0f} & {T['counts']['3000']['skew']['median']:.0f} & "
                 + " & ".join(f1(T['estimates']['3000'][str(j)]['median_deg']) for j in (1, 2, 3)) +
                 f" & {pct(T['screen']['3000']['2']['pass_share'])}/{pct(T['screen']['3000']['3']['pass_share'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "sweep", L)


def macros(m, T, R):
    """Macros for the numbers quoted in the text.  Suffix U = untreated, T = treated."""
    M = {}
    for P in (500, 3000):
        w = WORD[P]
        for tag, X in (("U", R), ("T", T)):
            M[f"k{w}{tag}"] = f"{X['counts'][str(P)]['skew']['median']:.0f}"
            M[f"peff{w}{tag}"] = f0(X['accuracy'][str(P)]['p_eff'])
            M[f"kurt{w}{tag}"] = f"{X['accuracy'][str(P)]['kurt_pooled']:.1f}"
            M[f"S{w}{tag}"] = f"{X['estimates'][str(P)]['S_over_k_pct']:.1f}"
            M[f"block{w}{tag}"] = f"{X['blocks'][str(P)]['own_k']:.2f}"
            M[f"blockFive{w}{tag}"] = f"{X['blocks'][str(P)]['m5']:.2f}"
            M[f"oos{w}{tag}"] = pct(X["oos"][str(P)]["viol_simple_A"])
            for j in (1, 2, 3):
                M[f"ang{JW[j]}{w}{tag}"] = f1(X["estimates"][str(P)][str(j)]["median_deg"])
                M[f"pass{JW[j]}{w}{tag}"] = pct(X["screen"][str(P)][str(j)]["pass_share"], 1)
                M[f"loc{JW[j]}{w}{tag}"] = str(X["screen"][str(P)][str(j)]["localized"])
                M[f"sd{JW[j]}{w}{tag}"] = f2(X["estimates"][str(P)][str(j)]["sd_deg"])
                M[f"bias{JW[j]}{w}{tag}"] = f2(X["estimates"][str(P)][str(j)]["bias_deg"])
                M[f"fa{JW[j]}{w}{tag}"] = f1(X["fa"][str(P)][str(j)]["Afa_deg"])
        t = T["estimates"][str(P)]
        for j in (1, 2, 3):
            M[f"diff{JW[j]}{w}"] = f"{t[str(j)]['paired_diff_deg']:+.2f}" if t[str(j)]["paired_diff_deg"] is not None else "--"
            M[f"below{JW[j]}{w}"] = pct(t[str(j)]["share_below_raw"])
            dt = t[str(j)].get("paired_diff_touched_deg")
            M[f"diffTouched{JW[j]}{w}"] = f"{dt:+.2f}" if dt is not None else "--"
            M[f"touchedMonths{JW[j]}{w}"] = str(t[str(j)].get("touched_months", 0))
            M[f"ref{JW[j]}{w}"] = f1(t[str(j)]["median_deg_refcount"])
        c = T["counts"][str(P)]
        M[f"kEqual{w}"] = pct(c["agree_with_raw"]); M[f"kHigher{w}"] = pct(c["higher_than_raw"])
        M[f"kLower{w}"] = pct(c["lower_than_raw"]); M[f"kChange{w}"] = pct(c["changes_month_to_month"])
        M[f"kChange{w}U"] = pct(R["counts"][str(P)]["changes_month_to_month"])
        tt = T["touched"][str(P)]
        M[f"changed{w}"] = num(tt["entries_changed"]); M[f"panelsTouched{w}"] = str(tt["panels_touched"])
        M[f"share{w}"] = fshare(100 * tt['entries_changed'] / (239 * 63 * tt['p_median']))
        M[f"dropped{w}"] = num(tt["names_dropped_total"]); M[f"left{w}"] = num(tt["above_100_left"])
        M[f"faNonconv{w}T"] = str(T["fa"][str(P)]["nonconverged"]); M[f"faNonconv{w}U"] = str(R["fa"][str(P)]["nonconverged"])
    for tag, X in (("U", R), ("T", T)):
        cr = X["crises"]["3000"]
        M[f"gfc{tag}"] = f1(cr["gfc"]["A1_deg"]); M[f"covid{tag}"] = f1(cr["covid"]["A1_deg"])
        M[f"gfcK{tag}"] = f"{cr['gfc']['k_median']:.0f}"; M[f"covidK{tag}"] = f"{cr['covid']['k_median']:.0f}"
        iv = X["interval"]["3000"]
        M[f"intSkew{tag}"] = f2(iv["skew"]); M[f"intBn{tag}"] = f2(iv["bn"])
    # a non-main setting with the slower diagnostics as well (01: c050, 03: c20): prefix alt
    for st in METHODS[m][1]:
        A = S(m, st)
        if st == "main" or A is None or "fa" not in A:
            continue
        for P in (500, 3000):
            w = WORD[P]
            M[f"altNonconv{w}"] = str(A["fa"][str(P)]["nonconverged"])
            M[f"altBlock{w}"] = f"{A['blocks'][str(P)]['own_k']:.2f}"
            M[f"altOos{w}"] = pct(A["oos"][str(P)]["viol_simple_A"])
            M[f"altPeff{w}"] = f0(A["accuracy"][str(P)]["p_eff"])
            M[f"altKurt{w}"] = f1(A["accuracy"][str(P)]["kurt_pooled"])
            M[f"altKEqual{w}"] = pct(A["counts"][str(P)]["agree_with_raw"])
            for j in (1, 2, 3):
                M[f"altAng{JW[j]}{w}"] = f1(A["estimates"][str(P)][str(j)]["median_deg"])
                M[f"altFa{JW[j]}{w}"] = f1(A["fa"][str(P)][str(j)]["Afa_deg"])
                M[f"altPass{JW[j]}{w}"] = pct(A["screen"][str(P)][str(j)]["pass_share"], 1)
        M["altGfc"] = f1(A["crises"]["3000"]["gfc"]["A1_deg"]); M["altCovid"] = f1(A["crises"]["3000"]["covid"]["A1_deg"])
        M["altShareThree"] = fshare(100 * A["touched"]["3000"]["entries_changed"] / (239 * 63 * 3000))
        M["altLeftThree"] = num(A["touched"]["3000"]["above_100_left"]); M["altLeftTwo"] = num(A["touched"]["2000"]["above_100_left"])
    lines = ["% generated by outlier_cleaning/code/make_tables.py; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    write(m, "numbers", lines)


def main():
    R = S("00_raw")
    raw_tables()
    for m in METHODS:
        T = S(m)
        if T is None or m == "00_raw":
            continue
        if True:
            t_changes(m, T, R)
        t_counts(m, T, R); t_estimates(m, T, R); t_screen(m, T, R)
        if "blocks" in T:
            t_accuracy(m, T, R); t_dayvar(m, T, R)
            macros(m, T, R)
        t_sweep(m)
        print("tables", m)



# ------------------------------------------------------------------ the untreated report
def raw_tables():
    R = S("00_raw"); m = "00_raw"
    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"$P$ & skewness rule & Bai--Ng & eigenvalue ratio & Onatski & changes month to month (\%)\\", r"\midrule"]
    for P in P_GRID:
        c = R["counts"][str(P)]
        L.append(f"{PNAME[P]} & {kq(c['skew'])} & {kq(c['bn'])} & {kq(c['ah'])} & {kq(c['ed'])} & {pct(c['changes_month_to_month'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "counts", L)
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"$P$ & $j$ & months & median & IQR & sd & bias\\", r"\midrule"]
    for P in P_GRID:
        for j in (1, 2, 3):
            e = R["estimates"][str(P)][str(j)]
            L.append(f"{PNAME[P] if j == 1 else ''} & {j} & {e['months']} & {f1(e['median_deg'])} & {f1(e['q25_deg'])}--{f1(e['q75_deg'])} & "
                     f"{f2(e['sd_deg'])} & ${e['bias_deg']:.2f}$\\\\")
        L.append(rf"& \multicolumn{{6}}{{l}}{{\footnotesize subspace number per direction {f1(R['estimates'][str(P)]['S_over_k_pct'])}\%; "
                 rf"$\peff\approx{R['accuracy'][str(P)]['p_eff']:.0f}$}}\\")
        if P != 3000:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "estimates", L)
    L = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"$P$ & $j$ & months & pass (\%) & localized & participation (median, \%) & peak share (median)\\", r"\midrule"]
    for P in P_GRID:
        for j in (1, 2, 3):
            s = R["screen"][str(P)][str(j)]
            L.append(f"{PNAME[P] if j == 1 else ''} & {j} & {s['months']} & {pct(s['pass_share'], 1)} & {s['localized']} & "
                     f"{pct(s['participation_median'], 1)} & {f2(s['peak_median'])}\\\\")
        if P != 3000:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "screen", L)
    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"$P$ & dispersion $c$ & $\peff$ & kurtosis, pooled & kurtosis, median name & bulk sd/mean\\", r"\midrule"]
    for P in P_GRID:
        a = R["accuracy"][str(P)]
        L.append(f"{PNAME[P]} & {f2(a['cv_split'])} & {f0(a['p_eff'])} & {f1(a['kurt_pooled'])} & {f1(a['kurt_median'])} & {f2(R['bulk'][str(P)])}\\\\")
    L += [r"\midrule", r"& \multicolumn{5}{l}{decile-1 residual block (\%)}\\",
          r"$P$ & at own count & $m=1$ & $m=5$ & $m=12$ & $m=40$\\", r"\midrule"]
    for P in (500, 3000):
        b = R["blocks"][str(P)]
        L.append(f"{PNAME[P]} & {f2(b['own_k'])} & {f2(b['m1'])} & {f2(b['m5'])} & {f2(b['m12'])} & {f2(b['m40'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "accuracy", L)
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"& \multicolumn{3}{c}{median angle, $A_j$ / $\Afa$ (degrees)} & fits not & day-variance & co-movement\\",
         r"\cmidrule(lr){2-4}", r"$P$ & $j=1$ & $j=2$ & $j=3$ & converged & CV (median) & holds (\%)\\", r"\midrule"]
    for P in P_GRID:
        f = R["fa"][str(P)]
        L.append(f"{PNAME[P]} & " + " & ".join(f"{f1(f[str(j)]['A_deg'])}/{f1(f[str(j)]['Afa_deg'])}" for j in (1, 2, 3)) +
                 f" & {f['nonconverged']} & {f2(f['day_var_cv'])} & {pct(f['comove_share'])}\\\\")
    L += [r"\midrule", r"$P$ & pairs & \multicolumn{2}{c}{below simple bound (\%), $A_j$ / $\Afa$} & direction 1 (\%) & median realized & median slack\\", r"\midrule"]
    for P in (500, 3000):
        o = R["oos"][str(P)]
        L.append(f"{PNAME[P]} & {o['pairs']} & \\multicolumn{{2}}{{c}}{{{pct(o['viol_simple_A'])} / {pct(o['viol_simple_FA'])}}} & {pct(o['viol_dir1_A'])} & "
                 f"{f2(o['realized_median'])} & {f2(o['slack_A_median'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(m, "dayvar", L)
    macros(m, R, R)


if __name__ == "__main__":
    main()
