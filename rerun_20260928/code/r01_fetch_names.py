"""Fetch CRSP name/SIC history from WRDS for SPAC identification (28 Sep 2026 rerun).

Writes (licensed data, stays in emp/data):
  emp/data/raw/wrds/crsp_names_20260928/stksecurityinfohist.parquet  (CIZ security info history)
  emp/data/raw/wrds/crsp_names_20260928/stocknames.parquet           (legacy SIZ name history, cross-check)
Run from emp/:  .venv/Scripts/python.exe ../rerun_20260928/code/r01_fetch_names.py
"""
import sys, hashlib, json
from pathlib import Path
EMP = Path(__file__).resolve().parents[2] / "emp"
sys.path.insert(0, str(EMP))
import wrds_client

OUT = EMP / "data" / "raw" / "wrds" / "crsp_names_20260928"
OUT.mkdir(parents=True, exist_ok=True)
db = wrds_client.connect()
try:
    for lib, tab in (("crsp_a_stock", "stksecurityinfohist"), ("crsp_a_stock", "stocknames")):
        try:
            cols = db.describe_table(lib, tab)
            print(lib, tab, list(cols["name"]))
        except Exception as e:
            print("describe failed", lib, tab, e)
    ciz = db.raw_sql("select * from crsp_a_stock.stksecurityinfohist where secinfoenddt >= '2005-01-01'")
    ciz.to_parquet(OUT / "stksecurityinfohist.parquet")
    print("ciz rows", len(ciz))
    siz = db.raw_sql("select * from crsp_a_stock.stocknames where nameenddt >= '2005-01-01'")
    siz.to_parquet(OUT / "stocknames.parquet")
    print("siz rows", len(siz))
finally:
    db.close()
man = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.glob("*.parquet")}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("done")
