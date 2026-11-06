"""Immutable CRSP CIZ extraction through WRDS Cloud server-side gzip."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
from typing import Any
from uuid import uuid4

import paramiko

from src.empirical_jse.crsp import (
    CIZ_DAILY_COLUMNS,
    ImmutableSnapshotBuilder,
    build_dsf_v2_query,
    iter_annual_slices,
)
from src.wrds_cloud import CLOUD_HOST


ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RemoteCommandResult:
    """Captured result of one fixed WRDS Cloud shell command."""

    stdout: str
    stderr: str
    exit_status: int


@dataclass(frozen=True)
class RemoteExport:
    """Verified remote gzip export ready for SFTP streaming."""

    remote_directory: str
    remote_query_path: str
    remote_artifact_path: str
    compressed_bytes: int
    compressed_sha256: str
    csv_lines: int
    command_stderr: str

    @property
    def rows(self) -> int:
        return max(self.csv_lines - 1, 0)


def _run_remote_command(
    transport: paramiko.Transport,
    command: str,
) -> RemoteCommandResult:
    channel = transport.open_session()
    channel.exec_command(command)
    stdout = channel.makefile("rb").read().decode("utf-8", errors="replace")
    stderr = (
        channel.makefile_stderr("rb")
        .read()
        .decode("utf-8", errors="replace")
    )
    exit_status = channel.recv_exit_status()
    channel.close()
    return RemoteCommandResult(stdout, stderr, exit_status)


def _mkdir_if_missing(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    try:
        attributes = sftp.stat(path.as_posix())
    except OSError:
        sftp.mkdir(path.as_posix(), mode=0o700)
        return
    if not stat.S_ISDIR(attributes.st_mode):
        raise RuntimeError(f"Remote path is not a directory: {path}")


def _write_new_remote_text(
    sftp: paramiko.SFTPClient,
    path: PurePosixPath,
    content: str,
) -> None:
    try:
        sftp.stat(path.as_posix())
    except OSError:
        pass
    else:
        raise FileExistsError(f"Remote artifact already exists: {path}")
    with sftp.file(path.as_posix(), mode="wx") as output:
        output.write(content.encode("utf-8"))
        output.flush()


def _archive_remote_run(
    sftp: paramiko.SFTPClient,
    *,
    run_root: PurePosixPath,
    legacy_root: PurePosixPath,
    error_type: str,
) -> PurePosixPath:
    archived = legacy_root / f"{run_root.name}-failed"
    sftp.rename(run_root.as_posix(), archived.as_posix())
    _write_new_remote_text(
        sftp,
        archived / "README.txt",
        "Failed WRDS Cloud export preserved without deletion.\n"
        f"Error type: {error_type}\n",
    )
    return archived


def _server_key_fingerprint(transport: paramiko.Transport) -> str:
    key = transport.get_remote_server_key()
    digest = hashlib.sha256(key.asbytes()).digest()
    import base64

    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _render_date_parameters(
    sql: str,
    parameters: Mapping[str, Any],
) -> str:
    """Render internally generated date parameters for a remote psql file."""

    rendered = sql
    for name, value in parameters.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Unsafe query parameter name: {name!r}")
        if not isinstance(value, date):
            raise TypeError(
                f"Remote query parameter {name!r} must be a date"
            )
        marker = f"%({name})s"
        if marker not in rendered:
            raise ValueError(f"Query does not contain parameter {marker}")
        rendered = rendered.replace(marker, f"DATE '{value.isoformat()}'")
    if "%(" in rendered:
        raise ValueError("Remote SQL still contains unresolved parameters")
    return rendered


def export_query_gzip(
    transport: paramiko.Transport,
    *,
    snapshot_id: str,
    query_sql: str,
) -> RemoteExport:
    """Run ``COPY TO STDOUT | gzip`` on WRDS Cloud and verify the result."""

    if not _SNAPSHOT_RE.fullmatch(snapshot_id):
        raise ValueError("snapshot_id contains unsupported characters")
    if "\x00" in query_sql:
        raise ValueError("query_sql cannot contain NUL bytes")

    sftp = paramiko.SFTPClient.from_transport(transport)
    run_root: PurePosixPath | None = None
    try:
        remote_home = PurePosixPath(sftp.normalize("."))
        project_root = remote_home / "empirical_jse_exports"
        runs_root = project_root / "runs"
        legacy_root = project_root / "legacy"
        _mkdir_if_missing(sftp, project_root)
        _mkdir_if_missing(sftp, runs_root)
        _mkdir_if_missing(sftp, legacy_root)

        run_name = f"{snapshot_id}-{uuid4().hex}"
        run_root = runs_root / run_name
        sftp.mkdir(run_root.as_posix(), mode=0o700)
        query_path = run_root / "query.sql"
        partial_path = run_root / "dsf_v2.csv.gz.partial"
        artifact_path = run_root / "dsf_v2.csv.gz"
        _write_new_remote_text(sftp, query_path, query_sql)
    except Exception as exc:
        if run_root is not None:
            try:
                _archive_remote_run(
                    sftp,
                    run_root=run_root,
                    legacy_root=legacy_root,
                    error_type=type(exc).__name__,
                )
            except Exception as recovery_exc:
                exc.add_note(
                    "Remote failed-run archival also failed: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                )
        raise
    finally:
        sftp.close()

    try:
        quoted_query = shlex.quote(query_path.as_posix())
        quoted_partial = shlex.quote(partial_path.as_posix())
        quoted_artifact = shlex.quote(artifact_path.as_posix())
        script = (
            "set -euo pipefail\n"
            "umask 077\n"
            f"test ! -e {quoted_partial}\n"
            f"test ! -e {quoted_artifact}\n"
            "psql -X -q -v ON_ERROR_STOP=1 -d wrds "
            f"-f {quoted_query} | gzip -n -9 > {quoted_partial}\n"
            f"gzip -t {quoted_partial}\n"
            f"mv -- {quoted_partial} {quoted_artifact}\n"
            f"sha256sum {quoted_artifact}\n"
            f"wc -c < {quoted_artifact}\n"
            f"gzip -cd {quoted_artifact} | wc -l\n"
        )
        command = "bash -lc " + shlex.quote(script)
        result = _run_remote_command(transport, command)
        if result.exit_status:
            raise RuntimeError(
                "WRDS Cloud export failed with status "
                f"{result.exit_status}: {result.stderr.strip()}"
            )

        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        if len(lines) != 3:
            raise RuntimeError(
                "Unexpected WRDS Cloud verification output: "
                + json.dumps(lines)
            )
        digest_fields = lines[0].split()
        if len(digest_fields) < 2 or len(digest_fields[0]) != 64:
            raise RuntimeError("Malformed remote sha256sum output")
        try:
            compressed_bytes = int(lines[1])
            csv_lines = int(lines[2])
        except ValueError as exc:
            raise RuntimeError("Malformed remote byte/line counts") from exc
        if compressed_bytes <= 0 or csv_lines < 1:
            raise RuntimeError(
                "Remote export is empty or missing its CSV header"
            )
    except Exception as exc:
        archive_error: Exception | None = None
        try:
            recovery_sftp = paramiko.SFTPClient.from_transport(transport)
            try:
                _archive_remote_run(
                    recovery_sftp,
                    run_root=run_root,
                    legacy_root=legacy_root,
                    error_type=type(exc).__name__,
                )
            finally:
                recovery_sftp.close()
        except Exception as recovery_exc:
            archive_error = recovery_exc
        if archive_error is not None:
            exc.add_note(
                "Remote failed-run archival also failed: "
                f"{type(archive_error).__name__}: {archive_error}"
            )
        raise

    return RemoteExport(
        remote_directory=run_root.as_posix(),
        remote_query_path=query_path.as_posix(),
        remote_artifact_path=artifact_path.as_posix(),
        compressed_bytes=compressed_bytes,
        compressed_sha256=digest_fields[0],
        csv_lines=csv_lines,
        command_stderr=result.stderr.strip(),
    )


def _remote_chunks(
    transport: paramiko.Transport,
    remote_path: str,
    *,
    block_size: int = 1024 * 1024,
) -> Iterator[bytes]:
    """Stream an immutable remote file over one continuous SSH channel.

    A fixed ``cat`` command avoids SFTP's request/response latency on
    high-round-trip links while preserving the exact remote bytes.
    """

    channel = transport.open_session(
        window_size=64 * 1024 * 1024,
        max_packet_size=32 * 1024,
    )
    try:
        channel.exec_command("cat -- " + shlex.quote(remote_path))
        while True:
            chunk = channel.recv(block_size)
            if not chunk:
                break
            yield bytes(chunk)
        stderr = (
            channel.makefile_stderr("rb")
            .read()
            .decode("utf-8", errors="replace")
        )
        exit_status = channel.recv_exit_status()
        if exit_status:
            raise RuntimeError(
                "Remote byte stream failed with status "
                f"{exit_status}: {stderr.strip()}"
            )
    finally:
        channel.close()


def _inspect_ciz_gzip(
    path: Path,
    *,
    expected_rows: int,
    expected_bytes: int,
) -> Mapping[str, Any]:
    if path.stat().st_size != expected_bytes:
        raise RuntimeError("SFTP artifact byte count differs from WRDS Cloud")
    row_count = 0
    minimum_date: date | None = None
    maximum_date: date | None = None
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise RuntimeError("Downloaded gzip has no CSV header") from exc
        if header != CIZ_DAILY_COLUMNS:
            raise RuntimeError(
                f"Unexpected CIZ header: expected {CIZ_DAILY_COLUMNS}, got {header}"
            )
        date_position = header.index("dlycaldt")
        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise RuntimeError(
                    f"CSV row {row_count + 1} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            parsed_date = date.fromisoformat(row[date_position])
            minimum_date = (
                parsed_date
                if minimum_date is None
                else min(minimum_date, parsed_date)
            )
            maximum_date = (
                parsed_date
                if maximum_date is None
                else max(maximum_date, parsed_date)
            )
    if row_count != expected_rows:
        raise RuntimeError(
            f"CSV row count mismatch: local={row_count}, remote={expected_rows}"
        )
    return {
        "format": "csv.gz",
        "rows": row_count,
        "min_date": None if minimum_date is None else minimum_date.isoformat(),
        "max_date": None if maximum_date is None else maximum_date.isoformat(),
        "remote_compressed_bytes": expected_bytes,
    }


def _archive_failed_staging(
    builder: ImmutableSnapshotBuilder,
    *,
    legacy_root: Path,
    reason: str,
) -> Path | None:
    if not builder.staging_path.exists():
        return None
    legacy_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = legacy_root / (
        f"{timestamp}_failed_cloud_snapshot_{builder.snapshot_id}_{uuid4().hex[:8]}"
    )
    builder.staging_path.rename(archive)
    (archive / "README.md").write_text(
        "# Failed WRDS Cloud snapshot staging\n\n"
        "This directory is preserved rather than deleted.\n\n"
        f"Reason: `{reason}`\n",
        encoding="utf-8",
    )
    return archive


def extract_ciz_cloud_snapshot(
    transport: paramiko.Transport,
    *,
    raw_root: str | Path,
    snapshot_id: str,
    start_date: date | str,
    end_date: date | str,
    metadata: Mapping[str, Any] | None = None,
    legacy_root: str | Path = ROOT / "legacy",
    created_at_utc: str | None = None,
) -> Path:
    """Create a verified immutable CIZ snapshot via WRDS Cloud."""

    builder = ImmutableSnapshotBuilder(
        raw_root,
        snapshot_id,
        created_at_utc=created_at_utc,
    )
    queries: list[dict[str, Any]] = []
    try:
        for period in iter_annual_slices(start_date, end_date):
            query = build_dsf_v2_query(period)
            rendered_select = _render_date_parameters(
                query.sql,
                query.params,
            )
            ordered_select = rendered_select.rstrip() + (
                "\nORDER BY d.dlycaldt, d.permno"
            )
            copy_sql = (
                "COPY (\n"
                + ordered_select
                + "\n) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, NULL '');\n"
            )
            remote = export_query_gzip(
                transport,
                snapshot_id=f"{snapshot_id}-{period.year}",
                query_sql=copy_sql,
            )
            relative = f"daily/year={period.year}/dsf_v2.csv.gz"
            record = builder.add_binary_chunks(
                relative,
                _remote_chunks(transport, remote.remote_artifact_path),
                inspect=lambda path, export=remote: _inspect_ciz_gzip(
                    path,
                    expected_rows=export.rows,
                    expected_bytes=export.compressed_bytes,
                ),
            )
            if record["sha256"] != remote.compressed_sha256:
                raise RuntimeError(
                    "Local SFTP artifact hash differs from WRDS Cloud sha256"
                )
            queries.append(
                {
                    "year": period.year,
                    "sql": ordered_select,
                    "params": {
                        key: (
                            value.isoformat()
                            if isinstance(value, date)
                            else value
                        )
                        for key, value in query.params.items()
                    },
                    "artifact": relative,
                    "rows": record["rows"],
                    "min_date": record["min_date"],
                    "max_date": record["max_date"],
                    "remote_directory": remote.remote_directory,
                    "remote_query_path": remote.remote_query_path,
                    "remote_artifact_path": remote.remote_artifact_path,
                    "remote_sha256": remote.compressed_sha256,
                    "remote_bytes": remote.compressed_bytes,
                    "remote_command_stderr": remote.command_stderr,
                }
            )

        manifest_metadata = {
            "source": "WRDS CRSP CIZ",
            "schema": "crsp_a_stock",
            "table": "dsf_v2",
            "transport": "wrds_cloud_psql_copy_gzip_sftp",
            "wrds_cloud_host": CLOUD_HOST,
            "wrds_cloud_host_key_sha256": _server_key_fingerprint(transport),
            "date_range_inclusive": {
                "start": date.fromisoformat(str(start_date)),
                "end": date.fromisoformat(str(end_date)),
            },
            "columns": CIZ_DAILY_COLUMNS,
            "raw_query_security_restriction": (
                "current_or_prior_year_materialized_security_history_join"
            ),
            "ordering": ["dlycaldt", "permno"],
            "compression": "gzip -n -9 on WRDS Cloud",
            "queries": queries,
            "user_metadata": dict(metadata or {}),
        }
        return builder.finalize(manifest_metadata)
    except Exception as exc:
        archive = _archive_failed_staging(
            builder,
            legacy_root=Path(legacy_root),
            reason=f"{type(exc).__name__}: {exc}",
        )
        if archive is not None:
            exc.add_note(f"Local staging preserved at {archive}")
        raise


__all__ = [
    "RemoteCommandResult",
    "RemoteExport",
    "export_query_gzip",
    "extract_ciz_cloud_snapshot",
]
