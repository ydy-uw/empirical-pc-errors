"""Reusable WRDS connection for this research project."""

from __future__ import annotations

from pathlib import Path

import wrds


PROJECT_ROOT = Path(__file__).resolve().parent
USERNAME_FILE = PROJECT_ROOT / ".wrds" / "username"


def get_username() -> str:
    """Return the locally configured WRDS username."""
    if not USERNAME_FILE.is_file():
        raise RuntimeError(
            "WRDS is not configured. Run configure_wrds.py in the project virtual environment."
        )

    username = USERNAME_FILE.read_text(encoding="utf-8").strip()
    if not username:
        raise RuntimeError("The configured WRDS username is empty; rerun configure_wrds.py.")
    return username


def connect(*, verbose: bool = False) -> wrds.Connection:
    """Connect using the username here and PostgreSQL's standard password file."""
    return wrds.Connection(wrds_username=get_username(), verbose=verbose)


def check_connection() -> None:
    """Run a read-only smoke test and print the accessible WRDS libraries."""
    username = get_username()
    db = connect()
    try:
        libraries = db.list_libraries()
        preview = ", ".join(libraries[:12])
        print(f"Connected to WRDS as {username}.")
        print(f"Accessible libraries: {len(libraries)}")
        print(f"First libraries: {preview}")
    finally:
        db.close()


if __name__ == "__main__":
    check_connection()

