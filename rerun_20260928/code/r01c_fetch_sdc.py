"""Third WRDS pull: SDC New Issues SPAC / blank-check IPO records (28 Sep 2026 rerun).
Discovers the SDC tables and flag columns, then saves every IPO flagged as SPAC or blank check.
Writes emp/data/raw/wrds/crsp_names_20260928/sdc_spac.parquet and sdc_schema.txt."""
import sys, hashlib, json, re
from pathlib import Path
EMP = Path(__file__).resolve().parents[2] / "emp"
sys.path.insert(0, str(EMP))
import wrds_client
OUT = EMP / "data" / "raw" / "wrds" / "crsp_names_20260928"
db = wrds_client.connect()
log = []
try:
    for lib in ("tr_sdc_ni", "sdc"):
        try:
            tabs = db.list_tables(lib)
        except Exception as e:
            log.append(f"{lib}: list_tables failed {e}"); continue
        log.append(f"{lib}: {tabs}")
        for t in tabs:
            try:
                cols = list(db.describe_table(lib, t)["name"])
            except Exception as e:
                log.append(f"  {t}: describe failed {e}"); continue
            hit = [c for c in cols if re.search(r"spac|blank|shell", c, re.I)]
            log.append(f"  {t} ({len(cols)} cols) flag-like: {hit}")
            log.append("     " + ", ".join(cols))
            if hit:
                keep = [c for c in cols if re.search(r"cusip|ticker|tic|name|nm|date|dt|spac|blank|shell|exch|master|deal|issue", c, re.I)]
                cond = " or ".join(f"cast({c} as text) in ('Y','Yes','YES','1','true','t')" for c in hit)
                try:
                    df = db.raw_sql(f"select {', '.join(keep)} from {lib}.{t} where {cond}")
                    df.to_parquet(OUT / f"sdc_spac_{lib}_{t}.parquet")
                    log.append(f"  -> saved {len(df)} rows from {lib}.{t}")
                except Exception as e:
                    log.append(f"  -> fetch failed {e}")
finally:
    db.close()
(OUT / "sdc_schema.txt").write_text("\n".join(log))
print("\n".join(l for l in log if not l.startswith("     ")))
