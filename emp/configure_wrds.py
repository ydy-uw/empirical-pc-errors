"""One-time interactive setup for WRDS on this Windows machine.

The WRDS password is entered with hidden input and saved only in PostgreSQL's
standard per-user password file. It is never stored in this project.
"""

from __future__ import annotations

import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import wrds


HOST = "wrds-pgdata.wharton.upenn.edu"
PORT = 9737
DATABASE = "wrds"
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_DIR = PROJECT_ROOT / ".wrds"
USERNAME_FILE = LOCAL_CONFIG_DIR / "username"
STATUS_FILE = LOCAL_CONFIG_DIR / "setup_status.json"


def write_status(status: str, **details: object) -> None:
    LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    STATUS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    print("WRDS one-time setup")
    print("Your password will be hidden and will not be saved in this project.")
    username = input("WRDS username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return 2

    password = getpass.getpass("WRDS password: ")
    if not password:
        print("Password cannot be empty.")
        return 2

    print("\nChecking the encrypted connection to WRDS...")
    try:
        raw_connection = psycopg2.connect(
            host=HOST,
            port=PORT,
            dbname=DATABASE,
            user=username,
            password=password,
            sslmode="require",
            connect_timeout=30,
            application_name="empirical_jse/setup",
        )
        with raw_connection.cursor() as cursor:
            cursor.execute("select current_user")
            authenticated_user = cursor.fetchone()[0]
        raw_connection.close()

        # Let the official client create/update PostgreSQL's standard password file:
        # %APPDATA%\postgresql\pgpass.conf on Windows.
        credential_writer = wrds.Connection(
            autoconnect=False,
            wrds_username=username,
            wrds_password=password,
        )
        credential_writer.create_pgpass_file()
        del credential_writer
        del password

        LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        USERNAME_FILE.write_text(username + "\n", encoding="utf-8")

        print("Verifying unattended access through the official WRDS client...")
        db = wrds.Connection(wrds_username=username)
        try:
            libraries = db.list_libraries()
        finally:
            db.close()

        write_status(
            "success",
            username=authenticated_user,
            accessible_library_count=len(libraries),
        )
        print(f"\nSuccess: connected as {authenticated_user}.")
        print(f"Accessible libraries: {len(libraries)}")
        return 0
    except Exception as exc:
        write_status("failed", username=username, error_type=type(exc).__name__)
        print(f"\nSetup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No password was written to this project.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if sys.stdin.isatty():
            input("\nPress Enter to close this window...")
