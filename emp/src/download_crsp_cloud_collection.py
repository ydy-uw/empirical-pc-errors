"""Resumable year-by-year CRSP CIZ downloader through WRDS Cloud."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

from src.empirical_jse.crsp import verify_snapshot
from src.wrds_cloud import connect_wrds_cloud
from src.wrds_cloud_export import extract_ciz_cloud_snapshot


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--snapshot-prefix", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "raw" / "wrds" / "crsp_ciz" / "annual",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must not precede --start-year")
    raw_root = args.raw_root if args.raw_root.is_absolute() else ROOT / args.raw_root
    collection_root = raw_root / "collections" / args.collection_id
    collection_manifest = collection_root / "collection_manifest.json"
    if collection_manifest.exists():
        raise FileExistsError(
            "collection manifest already exists and will not be overwritten: "
            f"{collection_manifest}"
        )

    records: list[dict[str, object]] = []
    transport, _first_use, _fingerprint = connect_wrds_cloud()
    transport.set_keepalive(30)
    try:
        for year in range(args.start_year, args.end_year + 1):
            snapshot_id = f"{args.snapshot_prefix}_{year}"
            snapshot = raw_root / snapshot_id
            if snapshot.exists():
                verification = verify_snapshot(snapshot)
                if not verification.ok:
                    raise RuntimeError(
                        f"existing snapshot {snapshot} failed verification: "
                        + "; ".join(verification.errors)
                    )
                status = "reused_verified"
            else:
                snapshot = extract_ciz_cloud_snapshot(
                    transport,
                    raw_root=raw_root,
                    snapshot_id=snapshot_id,
                    start_date=date(year, 1, 1),
                    end_date=date(year, 12, 31),
                    metadata={
                        "entry_point": "src.download_crsp_cloud_collection",
                        "collection_id": args.collection_id,
                    },
                )
                verification = verify_snapshot(snapshot)
                if not verification.ok:
                    raise RuntimeError(
                        f"new snapshot {snapshot} failed verification: "
                        + "; ".join(verification.errors)
                    )
                status = "downloaded_verified"

            manifest = snapshot / "manifest.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            records.append(
                {
                    "year": year,
                    "snapshot_id": snapshot_id,
                    "relative_path": snapshot.relative_to(raw_root).as_posix(),
                    "manifest_sha256": _sha256(manifest),
                    "rows": sum(
                        int(item.get("rows", 0))
                        for item in payload.get("files", [])
                    ),
                    "status": status,
                }
            )
            print(
                f"{year}: {status}; rows={records[-1]['rows']}; "
                f"snapshot={snapshot_id}",
                flush=True,
            )
    finally:
        transport.close()

    collection_root.mkdir(parents=True, exist_ok=False)
    collection_payload = {
        "format_version": 1,
        "collection_id": args.collection_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "snapshot_prefix": args.snapshot_prefix,
        "transport": "wrds_cloud_psql_copy_gzip_sftp",
        "snapshots": records,
    }
    collection_manifest.write_text(
        json.dumps(
            collection_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote collection manifest: {collection_manifest}", flush=True)


if __name__ == "__main__":
    main()
