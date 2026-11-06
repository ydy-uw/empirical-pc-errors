"""Download one bounded immutable CRSP CIZ snapshot through WRDS Cloud."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.empirical_jse.crsp import verify_snapshot
from src.wrds_cloud import connect_wrds_cloud
from src.wrds_cloud_export import extract_ciz_cloud_snapshot


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "raw" / "wrds" / "crsp_ciz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transport, _first_use, _fingerprint = connect_wrds_cloud()
    transport.set_keepalive(30)
    try:
        snapshot = extract_ciz_cloud_snapshot(
            transport,
            raw_root=args.raw_root,
            snapshot_id=args.snapshot_id,
            start_date=args.start,
            end_date=args.end,
            metadata={"entry_point": "src.download_crsp_cloud_ciz"},
        )
    finally:
        transport.close()
    verification = verify_snapshot(snapshot)
    if not verification.ok:
        raise RuntimeError("; ".join(verification.errors))
    print(f"Verified immutable snapshot: {snapshot}")


if __name__ == "__main__":
    main()
