"""Cleaned top-P universes for the 28 September 2026 rerun.

At every formation date, from the securities that the pipeline finds eligible
(r02_candidates.py; unchanged eligibility rules), remove

  * SPACs on the formation date (spac_flags.py), and
  * securities whose winsorized 63-day volatility is below VOL_CUTOFF (annualized),

then redo the company-level step of the pipeline (choose_permco_representatives: company
capitalization summed over all common share classes, the largest eligible class represents
the company) and take the P largest companies (select_top_cap), P = 500, 1000, 2000, 3000.
The next-largest eligible company therefore takes the place of each excluded one, and the
universes stay nested.

Output: emp/data/processed/rolling/rolling_topcap_clean_200601_202511_20260928/
  year=YYYY/selected_universes.parquet  (formation_date, n_assets, rank, permno, permco,
                                          company_cap, vol_w, baseline_rank)
  collection_manifest.json, exclusions.parquet (per month and size: what was removed)
This directory is what companion/code/rawpanel.py reads when HLHH_ROLLING_DIR points to it.
Run from emp/:  .venv/Scripts/python.exe ../rerun_20260928/code/r03_build_universes.py
"""
from __future__ import annotations
import sys, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parents[1] / "emp"
sys.path.insert(0, str(EMP))
from src.empirical_jse.crsp import choose_permco_representatives, select_top_cap  # noqa: E402

VOL_CUTOFF = 0.08
SIZES = (500, 1000, 2000, 3000)
CAND = EMP / "data/processed/clean_20260928/candidates"
RUN_ID = "rolling_topcap_clean_200601_202511_20260928"
OUT = EMP / "data/processed/rolling" / RUN_ID


def main() -> None:
    if (OUT / "collection_manifest.json").exists():
        raise FileExistsError(f"{OUT} already complete; move it to legacy/ before rebuilding")
    OUT.mkdir(parents=True, exist_ok=True)
    years, excl = [], []
    for yd in sorted(CAND.glob("year=*")):
        el = pd.read_parquet(yd / "eligible.parquet")
        cr = pd.read_parquet(yd / "common_rows.parquet")
        sel_all = []
        for fd, e in el.groupby("formation_date"):
            common = cr[cr.formation_date == fd].drop(columns="formation_date")
            lowvol = e.vol_w < VOL_CUTOFF
            keep = e[~e.spac & ~lowvol]
            reps = choose_permco_representatives(common, keep.permno.tolist())
            info = e.set_index("permno")
            for P in SIZES:
                permnos = select_top_cap(reps, P)
                s = reps.set_index("permno").loc[list(permnos)].reset_index()[["permno", "permco", "company_cap"]]
                s.insert(0, "rank", np.arange(1, P + 1)); s.insert(0, "n_assets", P); s.insert(0, "formation_date", fd)
                s["vol_w"] = s.permno.map(info.vol_w).to_numpy()
                s["baseline_rank"] = s.permno.map(info.baseline_rank).to_numpy()
                sel_all.append(s)
                base = e[e.baseline_rank <= P]                     # the original top-P companies
                excl.append(dict(formation_date=fd, n_assets=P,
                                 spac_removed=int(base.spac.sum()),
                                 lowvol_removed=int((~base.spac & (base.vol_w < VOL_CUTOFF)).sum()),
                                 kept_from_original=int(np.isin(permnos, base.permno).sum()),
                                 deepest_baseline_rank=int(np.nanmax(s.baseline_rank))))
        sel = pd.concat(sel_all, ignore_index=True)
        d = OUT / yd.name; d.mkdir(exist_ok=True)
        f = d / "selected_universes.parquet"
        sel.to_parquet(f, index=False)
        years.append(dict(year=int(yd.name[5:]), relative_path=yd.name,
                          sha256=hashlib.sha256(f.read_bytes()).hexdigest(), rows=len(sel)))
        print(yd.name, len(sel), flush=True)
    pd.DataFrame(excl).to_parquet(OUT / "exclusions.parquet", index=False)
    (OUT / "collection_manifest.json").write_text(json.dumps(dict(
        run_id=RUN_ID, created_at_utc=datetime.now(timezone.utc).isoformat(),
        derived_from="rolling_topcap_200601_202511_remote_v2_20260724 eligibility (reproduced exactly)",
        exclusions=dict(spac="rerun_20260928/code/spac_flags.py",
                        low_volatility=f"winsorized (1%/99%, per stock, own 63 days) annualized vol < {VOL_CUTOFF}"),
        sizes=list(SIZES), years=years), indent=1))


if __name__ == "__main__":
    main()
