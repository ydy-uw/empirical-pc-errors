"""Read-only WRDS Cloud SSH login probe."""

from __future__ import annotations

from dataclasses import asdict
import json

from src.wrds_cloud import run_probe


def main() -> None:
    result = run_probe()
    print(json.dumps(asdict(result), indent=2))
    if result.exit_status:
        raise SystemExit(result.exit_status)


if __name__ == "__main__":
    main()
