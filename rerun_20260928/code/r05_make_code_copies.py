"""Copy the analysis code of the 27 September package into rerun_20260928/code/ with redirected paths.

The originals are not modified.  In the copies
  * every output that the originals write under outlier_cleaning/ or figure_code/out/ goes to
    $HLHH_RERUN_OUT (default rerun_20260928/work/) instead;
  * inputs are unchanged: the CRSP extract and the universes are read through
    companion/code/rawpanel.py, whose universe directory is chosen with HLHH_ROLLING_DIR;
  * run_method.py records k_pipeline (the pipeline's own factor count, used only by
    validate_raw.py, which checks 00_raw against the old outputs) only when the universe run
    has factor_selections.parquet files; the cleaned universes do not.
Every substitution is asserted, so a missed path stops the script.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DST = ROOT / "rerun_20260928" / "code"
HEADER = "# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.\n"


def patch(src: Path, dst: Path, subs: list[tuple[str, str, int]]) -> None:
    s = src.read_text(encoding="utf-8")
    for old, new, count in subs:
        n = s.count(old)
        assert n == count, f"{src.name}: expected {count} x {old!r}, found {n}"
        s = s.replace(old, new)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(HEADER + s, encoding="utf-8")


# ---------------------------------------------------------------- outlier_cleaning/code
oc_src = ROOT / "outlier_cleaning" / "code"
oc_dst = DST / "outlier_cleaning"
for f in sorted(oc_src.glob("*.py")):
    subs = []
    if f.name == "oc_common.py":
        subs = [("STUDY = HERE.parent                                  # outlier_cleaning/\n",
                 "STUDY = Path(os.environ.get(\"HLHH_RERUN_OUT\", HERE.parents[1] / \"work\")) / \"outlier_cleaning\"\n", 1),
                ("ROOT = STUDY.parent                                  # project root\n",
                 "ROOT = HERE.parents[2]                               # project root\n", 1)]
    if f.name == "run_method.py":
        subs = [("            from p01_panel_stats import jung_counts\n"
                 "            kp = jung_counts()\n"
                 "            core[\"k_pipeline\"] = [int(kp.loc[(f, P)]) for f, P in zip(core.formation_date, core.n_assets)]\n",
                 "            from p01_panel_stats import jung_counts\n"
                 "            try:\n"
                 "                kp = jung_counts()\n"
                 "                core[\"k_pipeline\"] = [int(kp.loc[(f, P)]) for f, P in zip(core.formation_date, core.n_assets)]\n"
                 "            except FileNotFoundError:   # universes without the pipeline's factor selections\n"
                 "                core[\"k_pipeline\"] = -1\n", 1)]
    if f.name == "integrity.py":   # the cleaned universe run has no factor_selections.parquet
        subs = [("        for name in READ_FROM_ROLLING:\n            h = sha256(ydir / name)\n",
                 "        for name in READ_FROM_ROLLING:\n            if not (ydir / name).is_file():\n                continue\n            h = sha256(ydir / name)\n", 1)]
    patch(f, oc_dst / f.name, subs)

# ---------------------------------------------------------------- figure_code/outline_2026092{6,7}
ROOTLINE = "ROOT = Path(__file__).resolve().parents[2]\n"
NEWROOT = ("ROOT = Path(__file__).resolve().parents[4]\n"
           "import os  # noqa: E402\n"
           "WORK = Path(os.environ.get(\"HLHH_RERUN_OUT\", ROOT / \"rerun_20260928\" / \"work\"))\n")
for sub in ("outline_20260926", "outline_20260927"):
    for f in sorted((ROOT / "figure_code" / sub).glob("*.py")):
        s = f.read_text(encoding="utf-8")
        subs = [(ROOTLINE, NEWROOT, 1)]
        for old, new in (('ROOT / "figure_code" / "out"', 'WORK / "figure_code" / "out"'),
                         ('ROOT / "outlier_cleaning', 'WORK / "outlier_cleaning'),
                         # never default to writing into an earlier package
                         ('ROOT / "paper" / "overleaf_upload_20260926"', 'ROOT / "paper" / "overleaf_upload_20260928"'),
                         ('ROOT / "paper" / "overleaf_upload_20260927"', 'ROOT / "paper" / "overleaf_upload_20260928"')):
            if s.count(old):
                subs.append((old, new, s.count(old)))
        patch(f, DST / "figure_code" / sub / f.name, subs)
        t = (DST / "figure_code" / sub / f.name).read_text(encoding="utf-8")
        assert 'ROOT / "figure_code" / "out"' not in t and 'ROOT / "outlier_cleaning' not in t

print("copies written under", DST)
for p in sorted(DST.rglob("*.py")):
    if "outlier_cleaning" in p.parts or "figure_code" in p.parts:
        print("  ", p.relative_to(DST))
