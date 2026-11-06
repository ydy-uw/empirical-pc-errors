# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""Check that the inputs of the study are the untouched originals.

Every input file (the CRSP daily Parquet files and the per-year files of the frozen
rolling run that the study reads) is hashed with SHA-256 and compared with the hash
recorded in the pipeline's own manifests (where the manifest is present in this copy of the data).  The study opens these files read-only; the
check is run before and after each experiment and the result is written to the
experiment's logs/ folder.

    python outlier_cleaning/code/integrity.py <output.json>
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
from oc_common import sha256, dump
import paths

READ_FROM_ROLLING = ("selected_universes.parquet", "factor_selections.parquet")


def check():
    rows, ok = [], True
    cman = json.load(open(Path(paths.CRSP_PARQUET) / "collection_manifest.json"))
    for y in cman["years"]:
        f = Path(paths.CRSP_PARQUET) / y["relative_path"]
        h = sha256(f)
        rows.append(dict(file=str(Path(paths.CRSP_PARQUET).name + "/" + y["relative_path"]),
                         expected=y["parquet_sha256"], found=h, match=h == y["parquet_sha256"]))
    for ydir in sorted(Path(paths.ROLLING).glob("year=*")):
        mf = ydir / "manifest.json"
        man = ({e["path"]: e["sha256"] for e in json.load(open(mf))["files"]}
               if mf.is_file() else {})
        for name in READ_FROM_ROLLING:
            if not (ydir / name).is_file():
                continue
            h = sha256(ydir / name)
            exp = man.get(name)
            rows.append(dict(file=f"{Path(paths.ROLLING).name}/{ydir.name}/{name}",
                             expected=exp, found=h, match=(h == exp) if exp else None))
    checked = [r["match"] for r in rows if r["match"] is not None]
    return dict(checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                files=len(rows), with_manifest_hash=len(checked), all_match=all(checked),
                detail=rows)


def unchanged(before, after):
    """True when every input file has the same hash after the run as before it."""
    a = {r["file"]: r["found"] for r in after["detail"]}
    return all(a.get(r["file"]) == r["found"] for r in before["detail"])


if __name__ == "__main__":
    res = check()
    if len(sys.argv) > 1:
        dump(res, sys.argv[1])
    print(f"{res['files']} input files; {res['with_manifest_hash']} have a manifest hash; "
          f"all of those match: {res['all_match']}")
    sys.exit(0 if res["all_match"] else 1)
