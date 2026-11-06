"""SDC New Issues: IPOs of blank-check companies (SIC 6770 or SPAC business description), 2000-2025.
Writes emp/data/raw/wrds/crsp_names_20260928/sdc_spac_ipos.parquet (28 Sep 2026 rerun)."""
import sys, hashlib, json
from pathlib import Path
EMP = Path(__file__).resolve().parents[2] / "emp"
sys.path.insert(0, str(EMP))
import wrds_client
OUT = EMP / "data" / "raw" / "wrds" / "crsp_names_20260928"
db = wrds_client.connect()
try:
    sql = """select master_deal_no, ninames, cusip, cusip9, ticker, upticker, ipo, ipodate, master_deal_date,
                    firstradedate, sicp, upsicp, sic_print, sicdesc, businesss, description, security, exchange,
                    nation, offerpric, totdolamt, withdrawn_date, registration_status_long
             from tr_sdc_ni.wrds_ni_details
             where master_deal_date >= '2000-01-01'
               and (sicp = '6770' or upsicp = '6770' or sic_print ilike '%%6770%%'
                    or businesss ilike '%%blank check%%' or businesss ilike '%%special purpose acq%%'
                    or businesss ilike '%%business combination%%' or sicdesc ilike '%%blank check%%')"""
    df = db.raw_sql(sql)
    df.to_parquet(OUT / "sdc_spac_ipos.parquet")
    print(len(df)); print(df.ipo.value_counts(dropna=False)); print(df.sicp.value_counts().head())
    print(df.sample(8, random_state=0)[["ninames","cusip","ipodate","sicp","businesss","description"]].to_string())
finally:
    db.close()
man = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob("*.parquet")}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
