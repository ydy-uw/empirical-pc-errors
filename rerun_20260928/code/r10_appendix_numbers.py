"""Numbers for the data-cleaning appendix and for the effect of the cleaning on counts and angles."""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import spac_flags as sf

EMP = ROOT / "emp"
el = pd.concat([pd.read_parquet(f) for f in glob.glob(str(EMP / "data/processed/clean_20260928/candidates/year=*/eligible.parquet"))])
base = el[el.baseline_rank <= 3000].copy()
out = {}

# ---- SPAC rule: which branch flags each SPAC stock-month (original top-3,000)
h = sf.security_history()
m = base.merge(h[["permno", "start", "end", "spac_A", "spac_B", "link_sdc", "link_sic"]], on="permno")
m = m[(m.start <= m.formation_date) & (m.end >= m.formation_date)]
s = m[m.spac]
out["spac_stock_months"] = int(len(s)); out["spac_securities"] = int(s.permno.nunique())
out["spac_by_name"] = int(s.spac_A.sum()); out["spac_link_only"] = int((~s.spac_A).sum())
out["spac_link_only_secs"] = int(s.loc[~s.spac_A, "permno"].nunique())
out["spac_sdc_linked"] = int(s.link_sdc.sum()); out["spac_sic_linked"] = int(s.link_sic.sum())
y = s.formation_date.dt.year
out["spac_by_period"] = {k: int(v) for k, v in pd.cut(y, [2005, 2009, 2019, 2023, 2025], labels=["2006-09", "2010-19", "2020-23", "2024-25"]).value_counts().sort_index().items()}
out["spac_sic6770_last_year"] = int(m[m.spac & m.link_sic & (pd.to_numeric(h.set_index('permno').siccd, errors='coerce').reindex(m.permno).to_numpy() == 6770)].formation_date.dt.year.max()) if False else None
sd = pd.read_parquet(EMP / "data/raw/wrds/crsp_names_20260928/sdc_spac_ipos.parquet")
sd = sd[sd.businesss.str.contains(sf.SDC_BUSINESS, case=False, na=False, regex=True)]
out["sdc_records"] = int(len(sd)); out["sdc_us"] = int((sd.nation == "United States").sum())
out["spac_median_volw"] = float(s.vol_w.median()); out["spac_share_below8"] = float((s.vol_w < 0.08).mean())

# ---- low-volatility screen and the winsorization
ns = base[~base.spac]
w, r = ns.vol_w < 0.08, ns.vol < 0.08
out["lowvol_w"] = int(w.sum()); out["lowvol_raw"] = int(r.sum()); out["lowvol_only_w"] = int((w & ~r).sum()); out["lowvol_only_raw"] = int((~w & r).sum())
out["lowvol_secs"] = int(ns.loc[w, "permno"].nunique())

# ---- effect on counts and angles: 0927 (original) vs 0928 (cleaned) panels, clipped and untreated
def deg(a): return np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))
eff = {}
for tag, old, new in (("clipped", ROOT / "outlier_cleaning/experiments/01_fixed_clip/results/panels_c050.csv",
                       ROOT / "rerun_20260928/work/outlier_cleaning/experiments/01_fixed_clip/results/panels_c050.csv"),
                      ("untreated", ROOT / "outlier_cleaning/experiments/00_raw/results/panels_main.csv",
                       ROOT / "rerun_20260928/work/outlier_cleaning/experiments/00_raw/results/panels_main.csv")):
    o = pd.read_csv(old, parse_dates=["formation_date"]); n = pd.read_csv(new, parse_dates=["formation_date"])
    j = o.merge(n, on=["formation_date", "n_assets"], suffixes=("_o", "_n"))
    for P, g in j.groupby("n_assets"):
        d = dict(months=len(g),
                 k_skew_median=(float(g.k_skew_o.median()), float(g.k_skew_n.median())),
                 k_skew_same=float((g.k_skew_o == g.k_skew_n).mean()),
                 k_skew_absdiff_le1=float(((g.k_skew_o - g.k_skew_n).abs() <= 1).mean()),
                 k_bn_same=float((g.k_bn_o == g.k_bn_n).mean()), k_ah_same=float((g.k_ah_o == g.k_ah_n).mean()),
                 k_ed_same=float((g.k_ed_o == g.k_ed_n).mean()))
        for jj in (1, 2, 3):
            a, b = deg(g[f"A{jj}_o"]), deg(g[f"A{jj}_n"])
            ok = np.isfinite(a) & np.isfinite(b)
            dd = (b - a)[ok]
            d[f"ang{jj}_median"] = (float(np.nanmedian(a)), float(np.nanmedian(b)))
            d[f"ang{jj}_absdiff_median"] = float(np.median(np.abs(dd))) if len(dd) else None
            d[f"ang{jj}_absdiff_p90"] = float(np.percentile(np.abs(dd), 90)) if len(dd) else None
        same_k = g.k_skew_o == g.k_skew_n
        a, b = deg(g.A1_o[same_k]), deg(g.A1_n[same_k])
        d["ang1_absdiff_median_same_k"] = float(np.median(np.abs(b - a)))
        eff[f"{tag}_{P}"] = d
out["effect"] = eff
p = ROOT / "rerun_20260928/appendix_numbers.json"
p.write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str))
