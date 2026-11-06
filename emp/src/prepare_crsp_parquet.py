"""Convert verified annual CIZ gzip CSV snapshots to immutable Parquet years."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
from uuid import uuid4

from src.empirical_jse.crsp import verify_snapshot
from src.empirical_jse.dataset import load_raw_annual_csv
from src.empirical_jse.qa import audit_extract


ROOT = Path(__file__).resolve().parents[1]


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


def transform_code_fingerprint() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "empirical_jse" / "crsp.py",
        ROOT / "src" / "empirical_jse" / "dataset.py",
        ROOT / "src" / "empirical_jse" / "qa.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in paths
    }


def transform_runtime_fingerprint() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "packages": {
            package: package_version(package)
            for package in ("numpy", "pandas", "pyarrow")
        },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--snapshot-prefix", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "raw" / "wrds" / "crsp_ciz" / "annual",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "interim" / "crsp_ciz_parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must not precede --start-year")
    raw_root = args.raw_root if args.raw_root.is_absolute() else ROOT / args.raw_root
    output_base = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )
    dataset_root = output_base / args.dataset_id
    collection_manifest = dataset_root / "collection_manifest.json"
    if collection_manifest.exists():
        raise FileExistsError(
            f"dataset collection already exists and will not be overwritten: "
            f"{collection_manifest}"
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    code_fingerprint = transform_code_fingerprint()
    dataset_spec = {
        "format_version": 1,
        "dataset_id": args.dataset_id,
        "source_snapshot_prefix": args.snapshot_prefix,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "raw_root": str(raw_root.resolve()),
        "transform_code": code_fingerprint,
        "transform_code_sha256": canonical_sha256(code_fingerprint),
        "runtime": transform_runtime_fingerprint(),
    }
    dataset_spec_sha256 = canonical_sha256(dataset_spec)
    spec_path = dataset_root / "dataset_spec.json"
    if spec_path.exists():
        existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if canonical_sha256(existing_spec) != dataset_spec_sha256:
            raise RuntimeError(
                "dataset-id is already bound to a different conversion spec: "
                f"{spec_path}"
            )
    else:
        write_new_json_atomic(spec_path, dataset_spec)

    year_records: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        raw_snapshot = raw_root / f"{args.snapshot_prefix}_{year}"
        verification = verify_snapshot(raw_snapshot)
        if not verification.ok:
            raise RuntimeError(
                f"raw year {year} failed verification: "
                + "; ".join(verification.errors)
            )
        raw_manifest = raw_snapshot / "manifest.json"
        raw_payload = json.loads(raw_manifest.read_text(encoding="utf-8"))
        csv_records = [
            item
            for item in raw_payload["files"]
            if item["path"].endswith("dsf_v2.csv.gz")
        ]
        if len(csv_records) != 1:
            raise RuntimeError(f"year {year} does not contain exactly one daily file")
        csv_path = raw_snapshot / csv_records[0]["path"]

        final_year = dataset_root / f"year={year}"
        final_parquet = final_year / "daily.parquet"
        final_manifest = final_year / "manifest.json"
        if final_year.exists():
            if not final_parquet.is_file() or not final_manifest.is_file():
                raise RuntimeError(f"incomplete existing Parquet year: {final_year}")
            existing = json.loads(final_manifest.read_text(encoding="utf-8"))
            if (
                existing.get("parquet_sha256") != sha256(final_parquet)
                or existing.get("source_manifest_sha256") != sha256(raw_manifest)
                or existing.get("dataset_spec_sha256") != dataset_spec_sha256
            ):
                raise RuntimeError(f"existing Parquet year failed hashes: {year}")
            status = "reused_verified"
            rows = int(existing["rows"])
            parquet_hash = existing["parquet_sha256"]
        else:
            staging = dataset_root / f".year={year}.staging-{uuid4().hex}"
            staging.mkdir(parents=False, exist_ok=False)
            frame = load_raw_annual_csv(csv_path)
            qa = audit_extract(frame)
            if qa.duplicate_permno_dates:
                raise RuntimeError(
                    f"year {year} has {qa.duplicate_permno_dates} duplicate keys"
                )
            staging_parquet = staging / "daily.parquet"
            frame.to_parquet(
                staging_parquet,
                index=False,
                compression="zstd",
                engine="pyarrow",
            )
            parquet_hash = sha256(staging_parquet)
            year_manifest = {
                "format_version": 1,
                "year": year,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_snapshot": raw_snapshot.name,
                "source_manifest_sha256": sha256(raw_manifest),
                "source_csv_sha256": csv_records[0]["sha256"],
                "dataset_spec_sha256": dataset_spec_sha256,
                "rows": len(frame),
                "columns": list(frame.columns),
                "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                "qa": asdict(qa),
                "parquet_sha256": parquet_hash,
            }
            (staging / "manifest.json").write_text(
                json.dumps(
                    year_manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            staging.rename(final_year)
            status = "converted_verified"
            rows = len(frame)

        year_records.append(
            {
                "year": year,
                "relative_path": f"year={year}/daily.parquet",
                "rows": rows,
                "parquet_sha256": parquet_hash,
                "bytes": final_parquet.stat().st_size,
                "year_manifest_sha256": sha256(final_manifest),
                "source_manifest_sha256": sha256(raw_manifest),
                "status": status,
            }
        )
        print(f"{year}: {status}; rows={rows}", flush=True)

    collection = {
        "format_version": 2,
        "dataset_id": args.dataset_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_prefix": args.snapshot_prefix,
        "dataset_spec_sha256": dataset_spec_sha256,
        "dataset_spec_file_sha256": sha256(spec_path),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "years": year_records,
    }
    write_new_json_atomic(collection_manifest, collection)
    print(f"Wrote Parquet collection manifest: {collection_manifest}", flush=True)


if __name__ == "__main__":
    main()
