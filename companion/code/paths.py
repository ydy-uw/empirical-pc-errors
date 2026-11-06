"""Where the companion scripts find the pipeline's data.

Every data path used by ``companion/code`` (and, through it, by ``wpca/code``)
is defined here and nowhere else.  The defaults assume the layout of this
package, i.e. that the WRDS pipeline in ``emp/`` has written its data under
``emp/data`` with the run identifiers used in the paper.  Override any of them
with an environment variable, for example

    HLHH_EMP_DATA=/big/disk/emp/data python code/emp_main.py
    HLHH_ROLLING_ID=my_rolling_run_v1 python code/emp_main.py

Variables (all optional):

    HLHH_EMP_DATA            root of the pipeline's data tree   (default: <package>/emp/data)
    HLHH_DATASET_ID          CRSP Parquet collection id          (default: DATASET_ID below)
    HLHH_ROLLING_ID          top-cap rolling run id              (default: ROLLING_ID below)
    HLHH_NESTED_ROLLING_ID   nested-random rolling run id        (default: NESTED_ROLLING_ID below)
    HLHH_CRSP_PARQUET        full path, overrides EMP_DATA + DATASET_ID
    HLHH_ROLLING_DIR         full path, overrides EMP_DATA + ROLLING_ID
    HLHH_NESTED_ROLLING_DIR  full path, overrides EMP_DATA + NESTED_ROLLING_ID

What each directory must contain (all written by ``emp/src``):

    CRSP_PARQUET/year=YYYY/daily.parquet
        written by ``python -m src.prepare_crsp_parquet``
    ROLLING/year=YYYY/{observable_error_floors, factor_residual_lengths,
                       factor_selections, selected_universes}.parquet
        written by ``python -m src.run_rolling_experiments`` (top-cap design)
    NESTED_ROLLING/year=YYYY/{factor_residual_lengths, factor_selections}.parquet
        written by ``python -m src.run_rolling_experiments
                    --selection-design top3000_nested_random``

Provenance note.  The original September build read the floor and
factor-path tables from ``emp/data/processed/observable_floors_theorem2_*``
and ``factor_path_theorem2_*``.  Those directories are byte-identical copies
of the per-year files of the rolling run ``ROLLING_ID`` (they were downloaded
from WRDS Cloud by ``src.sync_wrds_cloud_observable_floors``), so reading the
rolling run directly gives the same inputs.  The original build also read the
security universes from the earlier run ``rolling_topcap_..._remote_v1_20260723``;
its ``selected_universes.parquet`` files have the same SHA-256 as those of
``ROLLING_ID`` in all twenty years, so a single run serves both.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


EMP_DATA = _env_path("HLHH_EMP_DATA", PACKAGE_ROOT / "emp" / "data")

DATASET_ID = os.environ.get(
    "HLHH_DATASET_ID", "crsp_ciz_2005_2025_remote_v1_20260723")
ROLLING_ID = os.environ.get(
    "HLHH_ROLLING_ID", "rolling_topcap_200601_202511_remote_v2_20260724")
NESTED_ROLLING_ID = os.environ.get(
    "HLHH_NESTED_ROLLING_ID",
    "rolling_nested_random_top3000_200601_202511_v1_20260811")

CRSP_PARQUET = _env_path(
    "HLHH_CRSP_PARQUET",
    EMP_DATA / "interim" / "crsp_ciz_parquet" / DATASET_ID)
ROLLING = _env_path(
    "HLHH_ROLLING_DIR",
    EMP_DATA / "processed" / "rolling" / ROLLING_ID)
NESTED_ROLLING = _env_path(
    "HLHH_NESTED_ROLLING_DIR",
    EMP_DATA / "processed" / "rolling" / NESTED_ROLLING_ID)


def year_files(root: Path | str, filename: str) -> list[str]:
    """Sorted ``root/year=*/filename`` paths; fail loudly if there are none.

    ``pd.concat([])`` would otherwise stop the script with the unhelpful
    "No objects to concatenate".  Also warns (once per directory) when the
    directory is not a completed pipeline output, i.e. has no
    ``collection_manifest.json`` -- an interrupted run would otherwise be
    analysed silently with some years missing.
    """
    files = sorted(str(p) for p in Path(root).glob(f"year=*/{filename}"))
    if not files:
        raise FileNotFoundError(
            f"no {root}/year=*/{filename} files.\n"
            "Run the emp/ pipeline first (docs/03_data_pipeline.md), or point "
            "the HLHH_* environment variables at your data. To see what is "
            "found: `python code/paths.py` from companion/.")
    _warn_if_incomplete(Path(root))
    return files


_WARNED: set[str] = set()


def _warn_if_incomplete(root: Path) -> None:
    key = str(root)
    if key in _WARNED:
        return
    _WARNED.add(key)
    if not (root / "collection_manifest.json").is_file():
        import warnings
        warnings.warn(
            f"{root} has no collection_manifest.json: the run that wrote it "
            "did not finish, so some years may be missing.",
            stacklevel=3)


def _describe(path: Path) -> tuple[bool, str]:
    """(complete?, one-line description) for one input directory."""
    if not path.is_dir():
        return False, "MISSING   "
    years = sorted(p.name for p in path.glob("year=*") if p.is_dir())
    complete = (path / "collection_manifest.json").is_file()
    span = f"{years[0][5:]}-{years[-1][5:]}" if years else "none"
    state = "ok        " if complete and years else "INCOMPLETE"
    return complete and bool(years), f"{state} {len(years):2d} year dirs ({span})"


if __name__ == "__main__":
    import sys
    print(f"{'EMP_DATA':15s} {'exists    ' if EMP_DATA.is_dir() else 'MISSING   '} {EMP_DATA}")
    all_ok = True
    for name in ("CRSP_PARQUET", "ROLLING", "NESTED_ROLLING"):
        path = globals()[name]
        ok, text = _describe(path)
        all_ok &= ok
        print(f"{name:15s} {text}  {path}")
    if not all_ok:
        print("\nSome inputs are missing or incomplete (no collection_manifest.json).\n"
              "Build them with emp/ (docs/03_data_pipeline.md) or set the HLHH_* "
              "variables described at the top of companion/code/paths.py.")
    sys.exit(0 if all_ok else 1)
