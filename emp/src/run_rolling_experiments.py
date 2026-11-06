"""Resumable full-history rolling CRSP variance experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import re
from uuid import uuid4

import pandas as pd
import yaml

from src.empirical_jse.crsp import (
    MissingDelistingReturnError,
    build_eligible_universe,
    build_estimation_panel,
    build_return_panel,
    formation_and_oos_dates,
    select_nested_random_top_cap,
    select_top_cap,
)
from src.empirical_jse.dataset import (
    ParquetCIZCollection,
    monthly_formation_dates,
    sha256_file,
    trading_calendar_from_reference,
)
from src.empirical_jse.experiment import (
    compute_formation_jung_diagnostics,
    evaluate_panel,
)
from src.empirical_jse.qa import audit_estimation_panel


ROOT = Path(__file__).resolve().parents[1]
LOOKBACK = 63
FIXED_FACTOR_COUNTS = tuple(range(1, 16)) + (20, 30)
JUNG_PRIMARY_ALPHA = 0.10
JUNG_MAX_FACTORS = 30
JUNG_RETRY_MAX_FACTORS = 50
JUNG_ALPHA_SENSITIVITY = (0.05, 0.10, 0.20)
MIN_PARTICIPATION_FRACTION = 0.01
MAX_ABSOLUTE_LOADING = 0.50
MP_ADJUST = True
NESTED_RANDOM_OUTER_SIZE = 3000
DEFAULT_SELECTION_SEED = 20260811
_SAFE_PUBLIC_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PUBLIC_SKIP_FIELDS = frozenset(
    {
        "formation_date",
        "n_assets",
        "reason",
        "error_class",
        "error_code",
    }
)
_IDENTIFIER_TEXT = re.compile(
    r"(?i)\b(?:permno|permco|cusip|ncusip|ticker|security[_ -]?id)\b"
)
YEAR_OUTPUT_FILES = frozenset(
    {
        "variance_comparisons.parquet",
        "factor_diagnostics.parquet",
        "factor_selections.parquet",
        "factor_residual_lengths.parquet",
        "error_decomposition_diagnostics.parquet",
        "observable_error_floors.parquet",
        "selected_universes.parquet",
        "extreme_events.parquet",
        "qa_records.json",
        "skipped_cells.json",
    }
)


def public_skip_record(
    *,
    formation_date: pd.Timestamp,
    n_assets: int | None,
    reason: str,
    error_code: str,
    exception: BaseException | None = None,
) -> dict[str, object]:
    """Return a bounded aggregate skip record with no exception message."""

    if not _SAFE_PUBLIC_TOKEN.fullmatch(reason):
        raise ValueError("unsafe public skip reason")
    if not _SAFE_PUBLIC_TOKEN.fullmatch(error_code):
        raise ValueError("unsafe public skip error code")
    record: dict[str, object] = {
        "formation_date": pd.Timestamp(formation_date).normalize(),
        "n_assets": int(n_assets) if n_assets is not None else None,
        "reason": reason,
        "error_code": error_code,
    }
    if exception is not None:
        error_class = type(exception).__name__
        record["error_class"] = (
            error_class
            if _SAFE_PUBLIC_TOKEN.fullmatch(error_class)
            else "Exception"
        )
    assert_public_skip_records([record])
    return record


def assert_public_skip_records(records: list[dict[str, object]]) -> None:
    """Reject exception messages, security identifiers, and unknown fields."""

    for record in records:
        required = {"formation_date", "n_assets", "reason", "error_code"}
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"public skip record lacks required fields: {sorted(missing)}"
            )
        unexpected = set(record) - _PUBLIC_SKIP_FIELDS
        if unexpected:
            raise ValueError(
                f"public skip record contains prohibited fields: "
                f"{sorted(unexpected)}"
            )
        for field in ("reason", "error_class", "error_code"):
            value = record.get(field)
            if field == "error_class" and (
                value is None or bool(pd.isna(value))
            ):
                continue
            if value is not None and (
                not isinstance(value, str)
                or not _SAFE_PUBLIC_TOKEN.fullmatch(value)
            ):
                raise ValueError(f"unsafe public skip {field}")
        encoded = json.dumps(record, ensure_ascii=False, default=str)
        if _IDENTIFIER_TEXT.search(encoded):
            raise ValueError("public skip record contains a security identifier")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def research_code_fingerprint() -> dict[str, str]:
    paths = [Path(__file__).resolve()]
    paths.extend(sorted((ROOT / "src" / "empirical_jse").glob("*.py")))
    paths.extend((ROOT / "config" / "research.yml", ROOT / "requirements.txt"))
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in paths
        if path.is_file()
    }


def runtime_fingerprint() -> dict[str, object]:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "pyarrow"):
        packages[package] = package_version(package)
    return {
        "python": platform.python_version(),
        "packages": packages,
    }


def write_new_json_atomic(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable JSON output already exists: {path}")
    staging = path.with_name(f".{path.name}.staging-{uuid4().hex}")
    staging.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(path)


def validate_reference_calendar_coverage(
    reference_snapshot: Path,
    calendar: pd.DatetimeIndex,
    *,
    requested_end: str,
) -> dict[str, object]:
    """Require a manifest-backed, complete final out-of-sample month."""
    manifest_path = reference_snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_period = pd.Timestamp(requested_end).to_period("M")
    oos_period = requested_period + 1
    required_calendar_end = oos_period.end_time.normalize()
    metadata_end = pd.Timestamp(
        manifest["metadata"]["date_range_inclusive"]["end"]
    ).normalize()
    calendar_records = [
        record
        for record in manifest.get("files", [])
        if record.get("path") == "calendar/calendar.csv.gz"
    ]
    if len(calendar_records) != 1:
        raise ValueError(
            "reference manifest must contain exactly one calendar artifact"
        )
    artifact_end = pd.Timestamp(calendar_records[0]["max_date"]).normalize()
    if metadata_end < required_calendar_end or artifact_end < required_calendar_end:
        raise ValueError(
            "reference snapshot does not certify the complete final OOS "
            f"calendar month {oos_period}; required through "
            f"{required_calendar_end.date()}, metadata_end="
            f"{metadata_end.date()}, calendar_artifact_end="
            f"{artifact_end.date()}"
        )
    oos_dates = calendar[calendar.to_period("M") == oos_period]
    if len(oos_dates) < 15:
        raise ValueError(
            "reference calendar has implausibly few trading dates in the "
            f"final OOS month {oos_period}: {len(oos_dates)}"
        )
    return {
        "requested_last_formation_month": str(requested_period),
        "required_final_oos_month": str(oos_period),
        "required_calendar_end": str(required_calendar_end.date()),
        "manifest_metadata_end": str(metadata_end.date()),
        "calendar_artifact_end": str(artifact_end.date()),
        "final_oos_trading_days": int(len(oos_dates)),
        "final_oos_first_trading_date": str(oos_dates.min().date()),
        "final_oos_last_trading_date": str(oos_dates.max().date()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--reference-snapshot", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--selection-design",
        choices=("top_cap", "top3000_nested_random"),
        default="top_cap",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=DEFAULT_SELECTION_SEED,
    )
    parser.add_argument("--allow-config-override", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "processed" / "rolling",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "research.yml",
    )
    return parser.parse_args()


def verified_year(
    year_dir: Path,
    *,
    expected_year: int | None = None,
    expected_run_signature: str | None = None,
    expected_formation_dates_sha256: str | None = None,
) -> bool:
    manifest_path = year_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if manifest.get("format_version") != 2:
        return False
    if expected_year is not None and manifest.get("year") != expected_year:
        return False
    if (
        expected_run_signature is not None
        and manifest.get("run_signature_sha256") != expected_run_signature
    ):
        return False
    if (
        expected_formation_dates_sha256 is not None
        and manifest.get("formation_dates_sha256")
        != expected_formation_dates_sha256
    ):
        return False
    formation_dates = manifest.get("formation_dates")
    if not isinstance(formation_dates, list):
        return False
    if canonical_sha256(formation_dates) != manifest.get(
        "formation_dates_sha256"
    ):
        return False
    records = manifest.get("files")
    if not isinstance(records, list):
        return False
    record_names = [record.get("path") for record in records]
    if (
        len(record_names) != len(set(record_names))
        or set(record_names) != YEAR_OUTPUT_FILES
    ):
        return False
    for record in records:
        relative = Path(str(record["path"]))
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
        ):
            return False
        path = year_dir / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256(path) != record["sha256"]
        ):
            return False
    return True


def write_year(
    run_root: Path,
    year: int,
    *,
    comparisons: list[pd.DataFrame],
    factor_diagnostics: list[pd.DataFrame],
    factor_selections: list[pd.DataFrame],
    factor_residuals: list[pd.DataFrame],
    error_decompositions: list[pd.DataFrame],
    observable_error_floors: list[pd.DataFrame],
    universes: list[pd.DataFrame],
    extreme_events: list[pd.DataFrame],
    qa_records: list[dict[str, object]],
    skip_records: list[dict[str, object]],
    run_signature_sha256: str,
    formation_dates: list[str],
) -> Path:
    assert_public_skip_records(skip_records)
    final = run_root / f"year={year}"
    if final.exists():
        raise FileExistsError(f"year output already exists: {final}")
    staging = run_root / f".year={year}.staging-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)

    tables = {
        "variance_comparisons.parquet": comparisons,
        "factor_diagnostics.parquet": factor_diagnostics,
        "factor_selections.parquet": factor_selections,
        "factor_residual_lengths.parquet": factor_residuals,
        "error_decomposition_diagnostics.parquet": error_decompositions,
        "observable_error_floors.parquet": observable_error_floors,
        "selected_universes.parquet": universes,
        "extreme_events.parquet": extreme_events,
    }
    for filename, frames in tables.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.shape[1] == 0:
            frame = pd.DataFrame({"_empty": pd.Series(dtype="boolean")})
        frame.to_parquet(staging / filename, index=False, compression="zstd")
    (staging / "qa_records.json").write_text(
        json.dumps(qa_records, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (staging / "skipped_cells.json").write_text(
        json.dumps(skip_records, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted(staging.iterdir()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "format_version": 2,
        "year": year,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_signature_sha256": run_signature_sha256,
        "formation_dates": formation_dates,
        "formation_dates_sha256": canonical_sha256(formation_dates),
        "files": files,
        "qa_records": len(qa_records),
        "skipped_cells": len(skip_records),
        "formation_jung_cells": len(factor_selections),
        "formation_jung_selection_rows": int(
            sum(len(frame) for frame in factor_selections)
        ),
        "formation_jung_selection_scope": "formation_63_day_only",
        "public_skip_schema_version": 1,
        "skip_exception_messages_written": False,
        "security_identifiers_in_skip_records": False,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(final)
    if not verified_year(
        final,
        expected_year=year,
        expected_run_signature=run_signature_sha256,
        expected_formation_dates_sha256=canonical_sha256(formation_dates),
    ):
        raise RuntimeError(f"published year failed verification: {final}")
    return final


def main() -> None:
    args = parse_args()
    dataset_root = (
        args.dataset_root
        if args.dataset_root.is_absolute()
        else ROOT / args.dataset_root
    )
    reference = (
        args.reference_snapshot
        if args.reference_snapshot.is_absolute()
        else ROOT / args.reference_snapshot
    )
    output_root = (
        args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    )
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"missing research config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_start = str(config["sample"]["first_formation_date"])
    configured_end = str(config["sample"]["last_formation_date"])
    configured_sizes = sorted(
        {int(value) for value in config["rolling"]["sizes"]}
    )
    requested_start = args.start or configured_start
    requested_end = args.end or configured_end
    requested_sizes = sorted(
        {int(value) for value in (args.sizes or configured_sizes)}
    )
    overrides = {
        "start": (requested_start, configured_start),
        "end": (requested_end, configured_end),
        "sizes": (requested_sizes, configured_sizes),
    }
    changed = {
        key: {"requested": requested, "configured": configured}
        for key, (requested, configured) in overrides.items()
        if requested != configured
    }
    if (changed or args.start_year is not None or args.end_year is not None) and (
        not args.allow_config_override
    ):
        raise ValueError(
            "requested sample differs from config/research.yml; pass "
            "--allow-config-override only for a labelled robustness run. "
            f"differences={changed}, start_year={args.start_year}, "
            f"end_year={args.end_year}"
        )
    protocol_checks = {
        "lookback": (
            int(config["rolling"]["estimation_trading_days"]),
            LOOKBACK,
        ),
        "jung_primary_alpha": (
            float(config["factors"]["jung_primary_alpha"]),
            JUNG_PRIMARY_ALPHA,
        ),
        "jung_initial_max": (
            int(config["factors"]["jung_initial_max"]),
            JUNG_MAX_FACTORS,
        ),
        "jung_retry_max": (
            int(config["factors"]["jung_retry_max"]),
            JUNG_RETRY_MAX_FACTORS,
        ),
        "fixed_factor_counts": (
            tuple(int(value) for value in config["factors"]["fixed_q_grid"]),
            FIXED_FACTOR_COUNTS,
        ),
        "min_participation_fraction": (
            float(config["qa_gates"]["min_factor_participation_fraction"]),
            MIN_PARTICIPATION_FRACTION,
        ),
        "max_absolute_loading": (
            float(config["qa_gates"]["max_factor_absolute_loading"]),
            MAX_ABSOLUTE_LOADING,
        ),
    }
    drift = {
        key: {"configured": configured, "implemented": implemented}
        for key, (configured, implemented) in protocol_checks.items()
        if configured != implemented
    }
    if drift:
        raise RuntimeError(
            "implementation constants drift from config/research.yml: "
            f"{drift}"
        )
    run_root = output_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    collection_manifest = run_root / "collection_manifest.json"
    if collection_manifest.exists():
        raise FileExistsError(
            f"completed run already exists and will not be overwritten: "
            f"{collection_manifest}"
        )

    collection = ParquetCIZCollection(dataset_root)
    calendar = trading_calendar_from_reference(reference)
    reference_calendar_coverage = validate_reference_calendar_coverage(
        reference,
        calendar,
        requested_end=requested_end,
    )
    formations = monthly_formation_dates(
        calendar,
        start=requested_start,
        end=requested_end,
        lookback=LOOKBACK,
    )
    expected_periods = pd.period_range(
        pd.Timestamp(requested_start).to_period("M"),
        pd.Timestamp(requested_end).to_period("M"),
        freq="M",
    )
    observed_periods = pd.PeriodIndex(
        [date.to_period("M") for date in formations],
        freq="M",
    )
    if not observed_periods.equals(expected_periods):
        missing_periods = expected_periods.difference(observed_periods)
        extra_periods = observed_periods.difference(expected_periods)
        raise ValueError(
            "reference calendar does not provide complete requested monthly "
            "coverage, including the 63-day first lookback and final OOS "
            f"month; missing={missing_periods.astype(str).tolist()}, "
            f"extra={extra_periods.astype(str).tolist()}"
        )
    final_window = formation_and_oos_dates(
        calendar,
        formations[-1],
        lookback=LOOKBACK,
    )
    final_oos_period = pd.Timestamp(requested_end).to_period("M") + 1
    expected_final_oos_dates = tuple(
        calendar[calendar.to_period("M") == final_oos_period]
    )
    if final_window.out_of_sample != expected_final_oos_dates:
        raise ValueError(
            "final formation window does not contain every certified trading "
            f"date in OOS month {final_oos_period}"
        )
    if args.start_year is not None:
        formations = tuple(date for date in formations if date.year >= args.start_year)
    if args.end_year is not None:
        formations = tuple(date for date in formations if date.year <= args.end_year)
    years = sorted({date.year for date in formations})
    sizes = requested_sizes
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("--sizes must contain positive integers")
    if args.selection_design == "top3000_nested_random" and (
        sizes != [500, 1000, 2000, 3000]
    ):
        raise ValueError(
            "top3000_nested_random requires --sizes 500 1000 2000 3000"
        )
    if not formations:
        raise ValueError(
            "requested sample has no valid formation dates with a complete "
            "63-day lookback and next-month OOS window"
        )

    formation_strings = [date.strftime("%Y-%m-%d") for date in formations]
    code_fingerprint = research_code_fingerprint()
    reference_manifest = reference / "manifest.json"
    run_spec = {
        "format_version": 1,
        "run_id": args.run_id,
        "inputs": {
            "dataset_root": str(dataset_root.resolve()),
            "dataset_collection_manifest_sha256": collection.manifest_sha256,
            "reference_snapshot": str(reference.resolve()),
            "reference_manifest_sha256": sha256_file(reference_manifest),
            "research_config": str(config_path.resolve()),
            "research_config_sha256": sha256_file(config_path),
            "reference_calendar_coverage": reference_calendar_coverage,
        },
        "requested_sample": {
            "start": requested_start,
            "end": requested_end,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "formation_dates": formation_strings,
            "formation_dates_sha256": canonical_sha256(formation_strings),
            "sizes": sizes,
            "config_override": bool(args.allow_config_override),
        },
        "protocol": {
            "selection_design": (
                "top_cap_permco_representative"
                if args.selection_design == "top_cap"
                else "top3000_cap_then_monthly_nested_random"
            ),
            "selection_seed": (
                None
                if args.selection_design == "top_cap"
                else int(args.selection_seed)
            ),
            "selection_seed_schedule": (
                None
                if args.selection_design == "top_cap"
                else "numpy_seedsequence_master_year_month"
            ),
            "selection_outer_universe_size": (
                None
                if args.selection_design == "top_cap"
                else NESTED_RANDOM_OUTER_SIZE
            ),
            "lookback_trading_days": LOOKBACK,
            "oos_horizon": "next_calendar_month",
            "fixed_factor_counts": list(FIXED_FACTOR_COUNTS),
            "jung_primary_alpha": JUNG_PRIMARY_ALPHA,
            "jung_max_factors": JUNG_MAX_FACTORS,
            "jung_retry_max_factors": JUNG_RETRY_MAX_FACTORS,
            "jung_alpha_sensitivity": list(JUNG_ALPHA_SENSITIVITY),
            "jung_selection_scope": "formation_63_day_only",
            "public_skip_schema_version": 1,
            "portfolio_weight_diagnostics_version": 1,
            "mp_adjust": MP_ADJUST,
            "min_participation_fraction": MIN_PARTICIPATION_FRACTION,
            "max_absolute_loading": MAX_ABSOLUTE_LOADING,
        },
        "code": {
            "files": code_fingerprint,
            "combined_sha256": canonical_sha256(code_fingerprint),
        },
        "runtime": runtime_fingerprint(),
    }
    run_signature_sha256 = canonical_sha256(run_spec)
    run_spec_path = run_root / "run_spec.json"
    if run_spec_path.exists():
        existing_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if canonical_sha256(existing_spec) != run_signature_sha256:
            raise RuntimeError(
                "run-id is already bound to different inputs, parameters, or "
                f"code; preserve it under legacy/ or choose a new run-id: "
                f"{run_spec_path}"
            )
    else:
        write_new_json_atomic(run_spec_path, run_spec)

    year_records = []
    for year in years:
        year_formation_strings = [
            value for value in formation_strings if pd.Timestamp(value).year == year
        ]
        year_formation_sha256 = canonical_sha256(year_formation_strings)
        final = run_root / f"year={year}"
        if final.exists():
            if not verified_year(
                final,
                expected_year=year,
                expected_run_signature=run_signature_sha256,
                expected_formation_dates_sha256=year_formation_sha256,
            ):
                raise RuntimeError(
                    "existing year failed content/provenance verification; "
                    f"preserve it under legacy/ or choose a new run-id: {final}"
                )
            year_records.append(
                {
                    "year": year,
                    "relative_path": final.relative_to(run_root).as_posix(),
                    "status": "reused_verified",
                    "manifest_sha256": sha256(final / "manifest.json"),
                }
            )
            print(f"{year}: reused verified output", flush=True)
            continue

        comparisons: list[pd.DataFrame] = []
        factor_diagnostics: list[pd.DataFrame] = []
        factor_selections: list[pd.DataFrame] = []
        factor_residuals: list[pd.DataFrame] = []
        error_decompositions: list[pd.DataFrame] = []
        observable_error_floors: list[pd.DataFrame] = []
        universes: list[pd.DataFrame] = []
        extreme_events: list[pd.DataFrame] = []
        qa_records: list[dict[str, object]] = []
        skip_records: list[dict[str, object]] = []

        for formation in (date for date in formations if date.year == year):
            window = formation_and_oos_dates(
                calendar,
                formation,
                lookback=LOOKBACK,
            )
            daily = collection.load_range(
                window.estimation[0],
                window.out_of_sample[-1],
            )
            try:
                eligibility = build_eligible_universe(daily, calendar, formation)
            except Exception as exc:
                skip_records.append(
                    public_skip_record(
                        formation_date=formation,
                        n_assets=None,
                        reason="eligibility_failure",
                        error_code="eligibility_exception",
                        exception=exc,
                    )
                )
                continue

            nested_samples = None
            if args.selection_design == "top3000_nested_random":
                if len(eligibility.representatives) < NESTED_RANDOM_OUTER_SIZE:
                    for size in sizes:
                        skip_records.append(
                            public_skip_record(
                                formation_date=formation,
                                n_assets=size,
                                reason="insufficient_outer_universe",
                                error_code="top3000_universe_unavailable",
                            )
                        )
                    print(
                        f"{formation.date()}: available="
                        f"{len(eligibility.representatives)}; "
                        "top3000_nested_random_skipped",
                        flush=True,
                    )
                    continue
                nested_samples = select_nested_random_top_cap(
                    eligibility.representatives,
                    sizes,
                    outer_n=NESTED_RANDOM_OUTER_SIZE,
                    seed=(
                        int(args.selection_seed),
                        int(formation.year),
                        int(formation.month),
                    ),
                )

            for size in sizes:
                if len(eligibility.representatives) < size:
                    skip_records.append(
                        public_skip_record(
                            formation_date=formation,
                            n_assets=size,
                            reason="insufficient_representatives",
                            error_code="universe_below_requested_size",
                        )
                    )
                    continue
                permnos = (
                    select_top_cap(eligibility.representatives, size)
                    if nested_samples is None
                    else nested_samples[size]
                )
                selected = (
                    eligibility.representatives.set_index("permno")
                    .loc[list(permnos)]
                    .reset_index()
                )
                selected.insert(0, "rank", range(1, len(selected) + 1))
                selected.insert(
                    0,
                    "selection_seed",
                    (
                        pd.NA
                        if nested_samples is None
                        else int(args.selection_seed)
                    ),
                )
                selected.insert(0, "selection_design", args.selection_design)
                selected.insert(0, "n_assets", size)
                selected.insert(0, "formation_date", formation)
                universes.append(selected)

                estimation_rows = daily.loc[
                    daily["permno"].isin(permnos)
                    & daily["dlycaldt"].isin(window.estimation)
                ].copy()
                numeric_returns = pd.to_numeric(
                    estimation_rows["dlyret"], errors="coerce"
                )
                events = estimation_rows.loc[numeric_returns.abs().gt(1.0)].copy()
                if not events.empty:
                    events.insert(0, "n_assets", size)
                    events.insert(0, "formation_date", formation)
                    extreme_events.append(events)
                try:
                    raw_estimation, centered_estimation = (
                        build_estimation_panel(daily, window, permnos)
                    )
                except Exception as exc:
                    skip_records.append(
                        public_skip_record(
                            formation_date=formation,
                            n_assets=size,
                            reason="estimation_panel_failure",
                            error_code="formation_matrix_invalid",
                            exception=exc,
                        )
                    )
                    continue
                try:
                    formation_jung = compute_formation_jung_diagnostics(
                        centered_estimation,
                        formation_date=formation,
                        primary_alpha=JUNG_PRIMARY_ALPHA,
                        initial_max_factors=JUNG_MAX_FACTORS,
                        retry_max_factors=JUNG_RETRY_MAX_FACTORS,
                        sensitivity_alphas=JUNG_ALPHA_SENSITIVITY,
                    )
                except Exception as exc:
                    skip_records.append(
                        public_skip_record(
                            formation_date=formation,
                            n_assets=size,
                            reason="jung_selection_failure",
                            error_code="formation_jung_exception",
                            exception=exc,
                        )
                    )
                    continue
                factor_diagnostics.append(
                    formation_jung.factor_diagnostics
                )
                factor_selections.append(formation_jung.factor_selections)
                factor_residuals.append(formation_jung.factor_residuals)
                try:
                    panel = build_return_panel(daily, window, permnos)
                except MissingDelistingReturnError as exc:
                    skip_records.append(
                        public_skip_record(
                            formation_date=formation,
                            n_assets=size,
                            reason="incomplete_terminal_return",
                            error_code="terminal_return_incomplete",
                            exception=exc,
                        )
                    )
                    continue
                except Exception as exc:
                    skip_records.append(
                        public_skip_record(
                            formation_date=formation,
                            n_assets=size,
                            reason="panel_failure",
                            error_code="oos_panel_exception",
                            exception=exc,
                        )
                    )
                    continue

                estimation_qa = audit_estimation_panel(raw_estimation)
                try:
                    result = evaluate_panel(
                        panel,
                        fixed_factor_counts=FIXED_FACTOR_COUNTS,
                        jung_alpha=JUNG_PRIMARY_ALPHA,
                        jung_max_factors=JUNG_MAX_FACTORS,
                        jung_retry_max_factors=JUNG_RETRY_MAX_FACTORS,
                        jung_alpha_sensitivity=JUNG_ALPHA_SENSITIVITY,
                        mp_adjust=MP_ADJUST,
                        min_participation_fraction=MIN_PARTICIPATION_FRACTION,
                        max_absolute_loading=MAX_ABSOLUTE_LOADING,
                        formation_jung=formation_jung,
                    )
                except Exception as exc:
                    skip_records.append(
                        public_skip_record(
                            formation_date=formation,
                            n_assets=size,
                            reason="experiment_failure",
                            error_code="variance_experiment_exception",
                            exception=exc,
                        )
                    )
                    continue
                comparisons.append(result.comparisons)
                error_decompositions.append(
                    result.error_decomposition_diagnostics
                )
                observable_error_floors.append(
                    result.observable_error_floors
                )
                qa_records.append(
                    {
                        "formation_date": formation,
                        "n_assets": size,
                        "eligibility_counts": dict(eligibility.counts),
                        "estimation": asdict(estimation_qa),
                        "oos": asdict(result.oos_qa),
                    }
                )
            print(
                f"{formation.date()}: available="
                f"{len(eligibility.representatives)}; "
                f"completed_cells={sum(1 for item in qa_records if pd.Timestamp(item['formation_date']) == formation)}",
                flush=True,
            )

        final = write_year(
            run_root,
            year,
            comparisons=comparisons,
            factor_diagnostics=factor_diagnostics,
            factor_selections=factor_selections,
            factor_residuals=factor_residuals,
            error_decompositions=error_decompositions,
            observable_error_floors=observable_error_floors,
            universes=universes,
            extreme_events=extreme_events,
            qa_records=qa_records,
            skip_records=skip_records,
            run_signature_sha256=run_signature_sha256,
            formation_dates=year_formation_strings,
        )
        year_records.append(
            {
                "year": year,
                "relative_path": final.relative_to(run_root).as_posix(),
                "status": "computed_verified",
                "manifest_sha256": sha256(final / "manifest.json"),
            }
        )
        print(f"{year}: published verified output", flush=True)

    write_new_json_atomic(
        collection_manifest,
        {
            "format_version": 2,
            "run_id": args.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_signature_sha256": run_signature_sha256,
            "run_spec_sha256": sha256(run_spec_path),
            "dataset_root": str(dataset_root),
            "reference_snapshot": str(reference),
            "start": requested_start,
            "end": requested_end,
            "sizes": sizes,
            "selection_design": run_spec["protocol"]["selection_design"],
            "selection_seed": run_spec["protocol"]["selection_seed"],
            "selection_outer_universe_size": run_spec["protocol"][
                "selection_outer_universe_size"
            ],
            "years": year_records,
        },
    )
    print(f"Wrote run collection: {collection_manifest}", flush=True)


if __name__ == "__main__":
    main()
