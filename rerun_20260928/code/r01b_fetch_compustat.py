"""Second WRDS pull for SPAC identification (28 Sep 2026 rerun).

Writes to emp/data/raw/wrds/crsp_names_20260928/:
  libraries.txt           the WRDS libraries this account can read
  ccm_lnkhist.parquet     CRSP-Compustat link history
  comp_company.parquet    Compustat company header (current SIC, names)
  comp_funda_sich.parquet Compustat annual historical SIC (sich) by gvkey x fiscal year
"""
import sys, hashlib, json
from pathlib import Path
EMP = Path(__file__).resolve().parents[2] / "emp"
sys.path.insert(0, str(EMP))
import wrds_client
OUT = EMP / "data" / "raw" / "wrds" / "crsp_names_20260928"
db = wrds_client.connect()
try:
    libs = db.list_libraries()
    (OUT / "libraries.txt").write_text("\n".join(sorted(libs)))
    print([l for l in libs if any(k in l.lower() for k in ("spac", "sdc", "ipo", "sec", "comp", "ccm", "audit"))])
    q = {
        "ccm_lnkhist": "select gvkey, lpermno, lpermco, linktype, linkprim, linkdt, linkenddt from crsp_a_ccm.ccmxpf_lnkhist where linktype in ('LU','LC','LS','LN','LX','LD') and lpermno is not null",
        "comp_company": "select gvkey, conm, sic, naics, ipodate, dldte, dlrsn, cik from comp.company",
        "comp_funda_sich": "select gvkey, datadate, fyear, sich, naicsh, conm from comp.funda where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C' and datadate >= '2004-01-01'",
    }
    for name, sql in q.items():
        df = db.raw_sql(sql)
        df.to_parquet(OUT / f"{name}.parquet"); print(name, len(df))
finally:
    db.close()
man = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob("*.parquet")}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
