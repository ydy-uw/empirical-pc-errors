# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""Summaries of each experiment folder: results/summary_<setting>.json.

    python outlier_cleaning/code/summarize.py            # every method and setting found

Conventions (the same as the outline's tables):
  * angles are arcsin(sqrt(.)) in degrees, medians over months;
  * A_j is reported over the months whose own count reaches j;
  * the screen's pass share for direction j is over the months whose untreated count
    reaches j, so every treatment is screened on the same months;
  * p_eff = P / (1 + c^2), c the median over months of the split-sample coefficient of
    variation of residual variances of the treated panels (Proposition A);
  * sd and bias of A_j from figure_code/theory.py at that p_eff, converted to degrees
    as in figure_code/paper/p11_headline_table.py.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from oc_common import EXP, P_GRID, CRISES, deg, dump
from theory import bias_second_order

N_DUAL = 62


def load(method, setting="main"):
    f = EXP / method / "results" / f"panels_{setting}.csv"
    return pd.read_csv(f, parse_dates=["formation_date"]) if f.is_file() else None


def q(x):
    x = pd.Series(x).dropna()
    return dict(median=float(x.median()), q25=float(x.quantile(0.25)), q75=float(x.quantile(0.75)),
                mean=float(x.mean())) if len(x) else None


def in_crisis(fd, tag):
    a, b = CRISES[tag]
    return (fd >= pd.Timestamp(a)) & (fd <= pd.Timestamp(b))


def spectra(method, setting):
    z = np.load(EXP / method / "results" / f"spectra_{setting}.npz")
    return {(pd.Timestamp(f), int(P)): t for f, P, t in zip(z["formation_date"], z["n_assets"], z["theta"])}


def summarize_core(df, raw, spec):
    if "A1_ref" not in df:            # the untreated run: its own count is the reference count
        df = df.assign(k_ref=df.k_skew, S_ref=df.S, **{f"A{j}_ref": df[f"A{j}"] for j in (1, 2, 3)})
    m = df.merge(raw[["formation_date", "n_assets", "A1", "A2", "A3", "k_skew", "S"]],
                 on=["formation_date", "n_assets"], suffixes=("", "_raw"))
    out = {"touched": {}, "counts": {}, "estimates": {}, "screen": {}, "accuracy": {},
           "bulk": {}, "interval": {}, "crises": {}, "time": {}}
    for P in P_GRID:
        g = m[m.n_assets == P].sort_values("formation_date")
        key = str(P)
        out["touched"][key] = dict(
            entries_changed=int(g.n_changed.sum()), panels_touched=int((g.n_changed > 0).sum()),
            share_changed_median=float(g.share_changed.median()), share_changed_mean=float(g.share_changed.mean()),
            names_touched_median=float(g.names_touched.median()), names_dropped_total=int(g.names_dropped.sum()),
            names_dropped_median=float(g.names_dropped.median()), p_median=float(g.p.median()),
            above_100_left=int(g.n_above_100.sum()),
            entries_moved_1pt=int(g.n_changed_1pt.sum()) if "n_changed_1pt" in g else None,
            zero_counts=int((g.k_skew_rule == 0).sum()) if "k_skew_rule" in g else 0)
        c = {r: q(g[f"k_{r}"]) for r in ("skew", "bn", "ah", "ed", "mp")}
        c["agree_with_raw"] = float((g.k_skew == g.k_skew_raw).mean())
        c["diff_vs_raw"] = q(g.k_skew - g.k_skew_raw)
        c["higher_than_raw"] = float((g.k_skew > g.k_skew_raw).mean())
        c["lower_than_raw"] = float((g.k_skew < g.k_skew_raw).mean())
        c["changes_month_to_month"] = float((g.k_skew.diff().dropna() != 0).mean())
        out["counts"][key] = c
        cv = float(g.cv_split.median())
        peff = P / (1 + cv ** 2)
        e = {}
        # bias per panel from the panel's own A_1..A_k (theory.bias_second_order, as in
        # figure_code/paper/p01_panel_stats.py), using the stored dual spectrum
        bias = np.full((len(g), 3), np.nan)
        for i, (_, r) in enumerate(g.iterrows()):
            k = int(r.k_skew)
            th = spec[(pd.Timestamp(r.formation_date), P)]
            b = bias_second_order(th[k:].mean() / th[:k], peff, N_DUAL)
            bias[i, :min(k, 3)] = b[:3]
        for j in (1, 2, 3):
            h = g[g.k_skew >= j]
            a = h[f"A{j}"]
            hr = g[g.k_skew_raw >= j]
            sd = np.degrees(a * np.sqrt(2 * (1 + 1 / (N_DUAL - h.k_skew)) / peff) / (2 * np.sqrt(a * (1 - a))))
            b = bias[(g.k_skew >= j).to_numpy(), j - 1]
            corr = np.clip(a.to_numpy() - b, 0, 1)
            both = g[(g.k_skew >= j) & (g.k_skew_raw >= j)]
            e[str(j)] = dict(
                months=int(len(h)), median_deg=float(deg(a.median())),
                q25_deg=float(deg(a.quantile(0.25))), q75_deg=float(deg(a.quantile(0.75))),
                median_deg_refcount=float(deg(g[f"A{j}_ref"].dropna().median())),
                median_deg_raw=float(deg(hr[f"A{j}_raw"].median())),
                paired_diff_deg=float(np.median(deg(both[f"A{j}"]) - deg(both[f"A{j}_raw"]))) if len(both) else None,
                share_below_raw=float((both[f"A{j}"] < both[f"A{j}_raw"]).mean()) if len(both) else None,
                touched_months=int((both.n_changed > 0).sum()),
                paired_diff_touched_deg=(float(np.median(deg(both[both.n_changed > 0][f"A{j}"])
                                                         - deg(both[both.n_changed > 0][f"A{j}_raw"])))
                                         if (both.n_changed > 0).any() else None),
                sd_deg=float(np.median(sd)), bias_deg=float(np.median(deg(a.to_numpy()) - deg(corr))))
        e["S_over_k_pct"] = 100 * float(g.S_over_k.median())
        e["S_median"] = float(g.S.median())
        e["S_over_k_refcount_pct"] = 100 * float((g.S_ref / g.k_ref).median()) if "S_ref" in g else None
        out["estimates"][key] = e
        s = {}
        for j in (1, 2, 3):
            h = g[g.k_skew_raw >= j]
            s[str(j)] = dict(months=int(len(h)), pass_share=float(1 - h[f"loc{j}"].mean()),
                             localized=int(h[f"loc{j}"].sum()),
                             participation_median=float(h[f"part{j}"].median()),
                             peak_median=float(h[f"peak{j}"].median()))
        out["screen"][key] = s
        out["accuracy"][key] = dict(cv_split=cv, p_eff=peff, cv_naive=float(g.cv_naive.median()),
                                    kurt_median=float(g.kurt_median.median()),
                                    kurt_pooled=float(g.kurt_pooled.median()))
        out["bulk"][key] = float(g.bulk_disp.median())
        out["interval"][key] = {r: float(deg(g[f"A1_at_{r}"].median())) for r in ("skew", "bn", "ah", "ed", "mp")}
        cr = {}
        for tag in CRISES:
            w = g[in_crisis(g.formation_date, tag)]
            cr[tag] = dict(months=int(len(w)), A1_deg=float(deg(w.A1.median())),
                           k_median=float(w.k_skew.median()), k_max=int(w.k_skew.max()))
        cr["all"] = dict(A1_deg=float(deg(g.A1.median())))
        out["crises"][key] = cr
        out["time"][key] = dict(A1_deg_by_year={str(y): float(deg(v.median()))
                                                for y, v in g.groupby(g.formation_date.dt.year).A1})
    return out


def summarize_heavy(method, setting="main"):
    r = EXP / method / "results"
    sfx = "" if setting == "main" else f"_{setting}"
    out = {}
    if (r / f"blocks{sfx}.csv").is_file():
        b = pd.read_csv(r / f"blocks{sfx}.csv")
        out["blocks"] = {str(P): dict(own_k=float(100 * b[(b.n_assets == P) & b.at_own_k].d1.median()),
                                      **{f"m{m}": float(100 * b[(b.n_assets == P) & (b.m == m)].d1.median())
                                         for m in (1, 3, 5, 8, 12, 20, 40)})
                         for P in (500, 3000)}
    if (r / f"fa{sfx}.csv").is_file():
        f = pd.read_csv(r / f"fa{sfx}.csv", parse_dates=["formation_date"])
        fa = {}
        for P in P_GRID:
            g = f[f.n_assets == P]
            d = dict(fits=int(len(g)), nonconverged=int((~g.converged).sum()),
                     day_var_cv=float(g.day_var_cv.median()), comove_share=float((g.comove_ratio >= 1).mean()),
                     max_leverage=float(g.max_leverage.median()))
            for j in (1, 2, 3):
                h = g.dropna(subset=[f"Afa{j}"])
                d[str(j)] = dict(A_deg=float(deg(h[f"A{j}"].median())), Afa_deg=float(deg(h[f"Afa{j}"].median())),
                                 diff_deg=float(np.median(deg(h[f"Afa{j}"]) - deg(h[f"A{j}"]))),
                                 below=float((h[f"Afa{j}"] < h[f"A{j}"]).mean()))
            for tag in CRISES:
                w = g[in_crisis(g.formation_date, tag)]
                d[f"crisis_{tag}"] = dict(A1_deg=float(deg(w.A1.median())), Afa1_deg=float(deg(w.Afa1.median())),
                                          nonconverged=int((~w.converged).sum()), months=int(len(w)))
            fa[str(P)] = d
        out["fa"] = fa
    if (r / f"oos{sfx}.csv").is_file():
        o = pd.read_csv(r / f"oos{sfx}.csv")
        out["oos"] = {str(P): dict(pairs=int(len(g)), viol_simple_A=float((g.realized_sub < g.simple_A).mean()),
                                   viol_simple_FA=float((g.realized_sub < g.simple_FA).mean()),
                                   slack_A_median=float((g.realized_sub - g.simple_A).median()),
                                   realized_median=float(g.realized_sub.median()),
                                   viol_dir1_A=float((g.realized_1 < g.bound_1_A).mean()))
                      for P, g in o.groupby("n_assets")}
    return out


def main():
    raw = load("00_raw")
    for d in sorted(EXP.iterdir()):
        if not (d / "results").is_dir():
            continue
        for f in sorted((d / "results").glob("panels_*.csv")):
            s = f.stem.replace("panels_", "")
            if not (d / "results" / f"spectra_{s}.npz").is_file():
                print(d.name, s, "skipped: run spectra.py first")
                continue
            df = load(d.name, s)
            summ = summarize_core(df, raw, spectra(d.name, s))
            summ.update(summarize_heavy(d.name, s))
            dump(summ, d / "results" / f"summary_{s}.json")
            print(d.name, s, "summarized")


if __name__ == "__main__":
    main()
