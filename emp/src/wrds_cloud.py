"""Secure WRDS Cloud SSH helpers.

Credentials are read from PostgreSQL's per-user password file and never
printed or copied into this project.  The first observed SSH host key is
recorded in the ignored ``.wrds`` directory; every later connection requires
an exact match.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import os
from pathlib import Path
import socket
from typing import Callable

import paramiko

from wrds_client import get_username


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRDS_CONFIG = PROJECT_ROOT / ".wrds"
KNOWN_HOSTS = WRDS_CONFIG / "known_hosts"
CLOUD_HOST = "wrds-cloud.wharton.upenn.edu"
CLOUD_PORT = 22
POSTGRES_HOST = "wrds-pgdata.wharton.upenn.edu"
POSTGRES_PORT = "9737"
POSTGRES_DATABASE = "wrds"


@dataclass(frozen=True)
class CloudProbe:
    """Sanitized result of a WRDS Cloud login probe."""

    host: str
    host_key_algorithm: str
    host_key_sha256: str
    first_use_key_recorded: bool
    command_stdout: str
    command_stderr: str
    exit_status: int


def _split_pgpass_line(line: str) -> tuple[str, ...]:
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.rstrip("\r\n"):
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    fields.append("".join(current))
    return tuple(fields)


def _pgpass_path() -> Path:
    override = os.environ.get("PGPASSFILE")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is unavailable; cannot locate pgpass.conf")
    return Path(appdata) / "postgresql" / "pgpass.conf"


def read_wrds_password(username: str) -> str:
    """Return the matching WRDS password without exposing it in output."""

    path = _pgpass_path()
    if not path.is_file():
        raise RuntimeError(f"PostgreSQL password file is missing: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.lstrip().startswith("#"):
            continue
        fields = _split_pgpass_line(raw_line)
        if len(fields) != 5:
            continue
        host, port, database, user, password = fields
        if (
            host in {POSTGRES_HOST, "*"}
            and port in {POSTGRES_PORT, "*"}
            and database in {POSTGRES_DATABASE, "*"}
            and user in {username, "*"}
        ):
            if not password:
                raise RuntimeError("The matching WRDS pgpass entry is empty")
            return password
    raise RuntimeError("No matching WRDS entry was found in pgpass.conf")


def _sha256_fingerprint(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _verify_or_record_host_key(
    host: str,
    key: paramiko.PKey,
) -> bool:
    WRDS_CONFIG.mkdir(parents=True, exist_ok=True)
    host_keys = paramiko.HostKeys()
    if KNOWN_HOSTS.is_file():
        host_keys.load(str(KNOWN_HOSTS))
    known = host_keys.lookup(host)
    if known:
        expected = known.get(key.get_name())
        if expected is None or expected.asbytes() != key.asbytes():
            raise paramiko.BadHostKeyException(host, key, expected)
        return False
    host_keys.add(host, key.get_name(), key)
    host_keys.save(str(KNOWN_HOSTS))
    return True


def _interactive_handler(
    password: str,
) -> Callable[[str, str, list[tuple[str, bool]]], list[str]]:
    def respond(
        title: str,
        instructions: str,
        prompts: list[tuple[str, bool]],
    ) -> list[str]:
        del title, instructions
        responses: list[str] = []
        for prompt, _echo in prompts:
            normalized = " ".join(prompt.lower().split())
            if "password" in normalized:
                responses.append(password)
            elif (
                "passcode" in normalized
                or "option" in normalized
                or "duo push" in normalized
            ):
                responses.append("1")
            else:
                raise paramiko.AuthenticationException(
                    "WRDS returned an unrecognized keyboard-interactive prompt"
                )
        return responses

    return respond


def connect_wrds_cloud(
    *,
    timeout_seconds: float = 45.0,
) -> tuple[paramiko.Transport, bool, str]:
    """Authenticate to WRDS Cloud and return a host-key-pinned transport."""

    username = get_username()
    password = read_wrds_password(username)
    connection = socket.create_connection(
        (CLOUD_HOST, CLOUD_PORT),
        timeout=timeout_seconds,
    )
    transport = paramiko.Transport(connection)
    transport.banner_timeout = timeout_seconds
    transport.auth_timeout = timeout_seconds * 3
    try:
        transport.start_client(timeout=timeout_seconds)
        server_key = transport.get_remote_server_key()
        first_use = _verify_or_record_host_key(CLOUD_HOST, server_key)
        transport.auth_interactive(
            username,
            _interactive_handler(password),
        )
        if not transport.is_authenticated():
            raise paramiko.AuthenticationException(
                "WRDS Cloud did not authenticate the SSH session"
            )
        fingerprint = _sha256_fingerprint(server_key)
        return transport, first_use, fingerprint
    except Exception:
        transport.close()
        raise
    finally:
        del password


def run_probe() -> CloudProbe:
    """Run a fixed, read-only command on WRDS Cloud."""

    transport, first_use, fingerprint = connect_wrds_cloud()
    try:
        channel = transport.open_session(timeout=30)
        channel.exec_command(
            "hostname -f; id -un; pwd; "
            "command -v psql; command -v gzip; command -v sha256sum; "
            "psql -X -qAt -d wrds "
            "-c 'select current_user, current_database()'"
        )
        stdout = channel.makefile("rb").read().decode("utf-8", errors="replace")
        stderr = (
            channel.makefile_stderr("rb")
            .read()
            .decode("utf-8", errors="replace")
        )
        exit_status = channel.recv_exit_status()
        server_key = transport.get_remote_server_key()
    finally:
        transport.close()
    return CloudProbe(
        host=CLOUD_HOST,
        host_key_algorithm=server_key.get_name(),
        host_key_sha256=fingerprint,
        first_use_key_recorded=first_use,
        command_stdout=stdout,
        command_stderr=stderr,
        exit_status=exit_status,
    )


__all__ = [
    "CLOUD_HOST",
    "CloudProbe",
    "connect_wrds_cloud",
    "read_wrds_password",
    "run_probe",
]
