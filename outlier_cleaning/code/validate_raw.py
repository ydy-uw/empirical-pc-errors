"""Reproduction check of the untreated experiment against the outline's outputs.

    python outlier_cleaning/code/validate_raw.py

Compares experiments/00_raw/results with figure_code/out (panel_stats.csv,
daily_vol_data.csv, residual_blocks.csv, oos_bound.csv) and companion/out/idio_cv.json,
and writes experiments/00_raw/results/reproduction_check.json.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from oc_common import EXP, ROOT, dump

FO = ROOT / "figure_code" / "out"
R = EXP / "00_raw" / "results"


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    same_nan = bool((np.isfinite(a) == np.isfinite(b)).all())
    return dict(max_abs_diff=float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None,
                compared=int(ok.sum()), same_missing=same_nan)


def main():
    out = {}
    r = pd.read_csv(R / "panels_main.csv", parse_dates=["formation_date"])
    out["count_equals_pipeline"] = dict(panels=int(len(r)), equal=int((r.k_skew == r.k_pipeline).sum()))
    ps = pd.read_csv(FO / "panel_stats.csv", parse_dates=["formation_date"])
    m = r.merge(ps, on=["formation_date", "n_assets"], suffixes=("", "_ps"))
    out["A_vs_panel_stats"] = {f"A{j}": maxdiff(m[f"A{j}"], m[f"A{j}_ps"]) for j in (1, 2, 3)}
    out["participation_vs_panel_stats"] = {f"part{j}": maxdiff(m[f"part{j}"], m[f"part{j}_ps"]) for j in (1, 2, 3)}
    cv = json.load(open(ROOT / "companion" / "out" / "idio_cv.json"))
    out["cv_split_median_vs_idio_cv"] = {P: dict(here=float(r[r.n_assets == int(P)].cv_split.median()),
                                                 stored=cv[P]["split"]) for P in cv}
    fa = pd.read_csv(R / "fa.csv", parse_dates=["formation_date"])
    dv = pd.read_csv(FO / "daily_vol_data.csv", parse_dates=["formation_date"])
    dv = dv[dv.arm == "raw"]
    m = fa.merge(dv, on=["formation_date", "n_assets"], suffixes=("", "_dv"))
    out["fa_vs_daily_vol_data"] = {c: maxdiff(m[c], m[f"{c}_dv"]) for c in
                                   ("A1", "Afa1", "A3", "Afa3", "iterations", "comove_ratio")}
    b = pd.read_csv(R / "blocks.csv", parse_dates=["formation_date"])
    rb = pd.read_csv(FO / "residual_blocks.csv", parse_dates=["formation_date"])
    rb = rb[rb.arm == "data"]
    m = b.merge(rb, on=["formation_date", "n_assets", "m"], suffixes=("", "_ref"))
    out["blocks_vs_residual_blocks"] = maxdiff(m.d1, m.d1_ref)
    o = pd.read_csv(R / "oos.csv", parse_dates=["t", "t2"])
    ob = pd.read_csv(FO / "oos_bound.csv", parse_dates=["t", "t2"])
    ob = ob[ob.arm == "raw"]
    m = o.merge(ob, on=["t", "t2", "n_assets"], suffixes=("", "_ref"))
    out["oos_vs_oos_bound"] = dict(pairs_here=int(len(o)), pairs_ref=int(len(ob)), matched=int(len(m)),
                                   realized_sub=maxdiff(m.realized_sub, m.realized_sub_ref),
                                   simple_A=maxdiff(m.simple_A, m.simple_A_ref))
    dump(out, R / "reproduction_check.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
