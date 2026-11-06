"""Mirror a complete WRDS Cloud empirical run into the local workspace.

Unlike the publication synchronizers, this tool is allowed to transfer the
licensed raw rows, prepared panels, selected universes, experiment outputs,
and aggregate reports needed for a genuinely local reproduction.  It remains
deliberately conservative: only ``data/`` and ``output/`` below one fixed run
are visible, links are rejected, existing files are never overwritten, and
every transferred byte stream is recorded with SHA-256 provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from queue import Empty, Queue
from threading import Event, Lock
from typing import Any, BinaryIO
from uuid import uuid4

import paramiko

from src import wrds_cloud

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "empirical_topcap_2005_2025_v2_20260723"
DEFAULT_RECEIPT = (
    ROOT
    / "notes"
    / "wrds_cloud_runs"
    / DEFAULT_RUN_ID
    / "local_workspace_mirror_v1.json"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.=+@ -]*$")
_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_FILE_BYTES = 2 * 1024**4


@dataclass(frozen=True)
class RemoteFile:
    relative_path: str
    bytes: int
    mtime: int


@dataclass(frozen=True)
class SyncSummary:
    run_id: str
    remote_root: str
    files: int
    bytes: int
    downloaded_files: int
    downloaded_bytes: int
    verified_existing_files: int
    receipt: str
    wrds_cloud_host_key_sha256: str


def _safe_remote_home(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not isinstance(value, str)
        or "\\" in value
        or "\0" in value
        or not path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("WRDS returned an unsafe remote home")
    return path


def _safe_relative(value: PurePosixPath) -> str:
    text = value.as_posix()
    if (
        value.is_absolute()
        or not value.parts
        or value.parts[0] not in {"data", "output"}
        or any(part in {"", ".", ".."} for part in value.parts)
        or any(not _SAFE_COMPONENT.fullmatch(part) for part in value.parts)
    ):
        raise ValueError(f"unsafe remote workspace path: {text!r}")
    return text


def _remote_run_root(remote_home: str, run_id: str) -> PurePosixPath:
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    return _safe_remote_home(remote_home) / "empirical_jse_runs" / run_id


def _require_directory(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    attributes = sftp.lstat(path.as_posix())
    mode = int(attributes.st_mode)
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"remote path is not a real directory: {path}")


def inventory_remote(
    sftp: paramiko.SFTPClient,
    *,
    remote_home: str,
    run_id: str,
) -> tuple[PurePosixPath, list[RemoteFile]]:
    """Return a stable, link-free inventory of the run's data and outputs."""

    run_root = _remote_run_root(remote_home, run_id)
    records: list[RemoteFile] = []
    for top in ("data", "output"):
        top_path = run_root / top
        _require_directory(sftp, top_path)
        pending = [top_path]
        while pending:
            directory = pending.pop()
            entries = sorted(sftp.listdir_attr(directory.as_posix()), key=lambda x: x.filename)
            for attributes in entries:
                name = attributes.filename
                if name in {"", ".", ".."} or "/" in name or "\\" in name:
                    raise RuntimeError(f"unsafe remote member name: {name!r}")
                absolute = directory / name
                mode = int(attributes.st_mode)
                if stat.S_ISLNK(mode):
                    raise RuntimeError(f"remote workspace contains a link: {absolute}")
                if stat.S_ISDIR(mode):
                    pending.append(absolute)
                    continue
                if not stat.S_ISREG(mode):
                    raise RuntimeError(f"remote workspace has a special file: {absolute}")
                size = int(attributes.st_size)
                if size < 0 or size > _MAX_FILE_BYTES:
                    raise RuntimeError(f"unsafe remote file size ({size}): {absolute}")
                relative = absolute.relative_to(run_root)
                records.append(
                    RemoteFile(
                        relative_path=_safe_relative(relative),
                        bytes=size,
                        mtime=int(attributes.st_mtime),
                    )
                )
    records.sort(key=lambda item: item.relative_path)
    if len({item.relative_path for item in records}) != len(records):
        raise RuntimeError("remote inventory contains duplicate paths")
    return run_root, records


def inventory_remote_fast(
    transport: paramiko.Transport,
    *,
    remote_home: str,
    run_id: str,
) -> tuple[PurePosixPath, list[RemoteFile]]:
    """Inventory with one fixed remote ``find`` to avoid SFTP RTT scaling."""

    run_root = _remote_run_root(remote_home, run_id)
    channel = transport.open_session(timeout=30)
    command = (
        "cd -- "
        + shlex.quote(run_root.as_posix())
        + " && find -P data output -mindepth 1 "
        + r"-printf '%y\t%p\t%s\t%T@\0'"
    )
    channel.exec_command(command)
    stdout = channel.makefile("rb").read()
    stderr = channel.makefile_stderr("rb").read().decode(
        "utf-8", errors="replace"
    )
    exit_status = channel.recv_exit_status()
    if exit_status != 0:
        raise RuntimeError(
            f"remote workspace inventory failed with status {exit_status}: {stderr}"
        )
    records: list[RemoteFile] = []
    for raw in stdout.split(b"\0"):
        if not raw:
            continue
        fields = raw.decode("utf-8", errors="strict").split("\t")
        if len(fields) != 4:
            raise RuntimeError("malformed remote workspace inventory record")
        kind, relative_text, size_text, mtime_text = fields
        relative = PurePosixPath(relative_text)
        _safe_relative(relative)
        if kind == "d":
            continue
        if kind == "l":
            raise RuntimeError(
                f"remote workspace contains a link: {run_root / relative}"
            )
        if kind != "f":
            raise RuntimeError(
                f"remote workspace has a special file: {run_root / relative}"
            )
        size = int(size_text)
        if size < 0 or size > _MAX_FILE_BYTES:
            raise RuntimeError(f"unsafe remote file size ({size}): {relative}")
        records.append(
            RemoteFile(
                relative_path=relative.as_posix(),
                bytes=size,
                mtime=int(float(mtime_text)),
            )
        )
    records.sort(key=lambda item: item.relative_path)
    if len({item.relative_path for item in records}) != len(records):
        raise RuntimeError("remote inventory contains duplicate paths")
    return run_root, records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_remote_via_ssh(
    transport: paramiko.Transport,
    remote_path: PurePosixPath,
    destination: BinaryIO | None,
    *,
    offset: int = 0,
) -> tuple[int, str]:
    """Read one fixed regular file, optionally resuming at a byte offset."""

    if offset < 0:
        raise ValueError("remote byte offset cannot be negative")

    channel = transport.open_session(
        window_size=64 * 1024 * 1024,
        max_packet_size=32 * 1024,
    )
    digest = hashlib.sha256()
    total = 0
    try:
        quoted = shlex.quote(remote_path.as_posix())
        if offset:
            channel.exec_command(
                f"dd if={quoted} iflag=skip_bytes bs=1M skip={offset} status=none"
            )
        else:
            channel.exec_command("cat -- " + quoted)
        while True:
            block = channel.recv(1024 * 1024)
            if not block:
                break
            content = bytes(block)
            total += len(content)
            digest.update(content)
            if destination is not None:
                destination.write(content)
        stderr = channel.makefile_stderr("rb").read().decode(
            "utf-8", errors="replace"
        )
        status = channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(
                f"remote byte stream failed with status {status} for "
                f"{remote_path}: {stderr}"
            )
    finally:
        channel.close()
    return total, digest.hexdigest()


def _remote_sha256_via_ssh(
    transport: paramiko.Transport, remote_path: PurePosixPath
) -> str:
    """Hash a remote file without retransmitting its contents."""

    channel = transport.open_session()
    try:
        channel.exec_command("sha256sum -- " + shlex.quote(remote_path.as_posix()))
        stdout = channel.makefile("rb").read().decode("ascii", errors="strict")
        stderr = channel.makefile_stderr("rb").read().decode(
            "utf-8", errors="replace"
        )
        status = channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(
                f"remote SHA-256 failed with status {status} for "
                f"{remote_path}: {stderr}"
            )
        value = stdout.split(maxsplit=1)[0].lower()
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuntimeError(f"invalid remote SHA-256 output for {remote_path}")
        return value
    finally:
        channel.close()


def _unchanged_remote(
    before: RemoteFile,
    after: Any,
    *,
    remote_path: PurePosixPath,
) -> None:
    mode = int(after.st_mode)
    if (
        stat.S_ISLNK(mode)
        or not stat.S_ISREG(mode)
        or int(after.st_size) != before.bytes
        or int(after.st_mtime) != before.mtime
    ):
        raise RuntimeError(f"remote file changed during synchronization: {remote_path}")


def _prepare_resumable_partial(
    partial: Path,
    partial_metadata: Path,
    *,
    expected_metadata: dict[str, Any],
    expected_bytes: int,
) -> int:
    """Create or validate tool-owned resume state and return its byte offset."""

    if partial.exists() and not partial_metadata.exists():
        raise FileExistsError(f"unidentified resumable-download state: {partial}")
    if partial_metadata.exists():
        observed_metadata = json.loads(partial_metadata.read_text(encoding="utf-8"))
        if observed_metadata != expected_metadata:
            raise FileExistsError(f"resumable-download metadata conflict: {partial}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_bytes:
            raise RuntimeError(f"resumable download is larger than remote file: {partial}")
        return offset
    with partial_metadata.open("x", encoding="utf-8") as metadata_output:
        metadata_output.write(json.dumps(expected_metadata, sort_keys=True) + "\n")
        metadata_output.flush()
        os.fsync(metadata_output.fileno())
    return 0


def _sync_one(
    sftp: paramiko.SFTPClient,
    *,
    transport: paramiko.Transport,
    run_root: PurePosixPath,
    record: RemoteFile,
    local_root: Path,
) -> tuple[dict[str, Any], bool]:
    relative = PurePosixPath(record.relative_path)
    remote_path = run_root / relative
    destination = local_root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = sftp.lstat(remote_path.as_posix())
    _unchanged_remote(record, before, remote_path=remote_path)

    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise FileExistsError(f"local destination is not a regular file: {destination}")
        local_size = destination.stat().st_size
        local_sha256 = _sha256_file(destination)
        remote_sha256 = _remote_sha256_via_ssh(transport, remote_path)
        after = sftp.lstat(remote_path.as_posix())
        _unchanged_remote(record, after, remote_path=remote_path)
        if (
            local_size != record.bytes
            or local_sha256 != remote_sha256
        ):
            raise FileExistsError(
                f"local file conflicts with WRDS Cloud and was not overwritten: {destination}"
            )
        return {
            **asdict(record),
            "sha256": remote_sha256,
            "local_path": destination.relative_to(local_root).as_posix(),
        }, False

    partial = destination.with_name(f".{destination.name}.partial-wrds-sync-v1")
    partial_metadata = destination.with_name(
        f".{destination.name}.partial-wrds-sync-v1.json"
    )
    expected_metadata = {
        "remote_path": remote_path.as_posix(),
        "bytes": record.bytes,
        "mtime": record.mtime,
    }
    offset = _prepare_resumable_partial(
        partial,
        partial_metadata,
        expected_metadata=expected_metadata,
        expected_bytes=record.bytes,
    )
    with partial.open("ab") as output:
        downloaded, _suffix_sha256 = _read_remote_via_ssh(
            transport, remote_path, output, offset=offset
        )
        output.flush()
        os.fsync(output.fileno())
    after = sftp.lstat(remote_path.as_posix())
    _unchanged_remote(record, after, remote_path=remote_path)
    if offset + downloaded != record.bytes or partial.stat().st_size != record.bytes:
        raise RuntimeError(f"downloaded byte count mismatch: {remote_path}")
    remote_sha256 = _remote_sha256_via_ssh(transport, remote_path)
    if _sha256_file(partial) != remote_sha256:
        raise RuntimeError(f"downloaded SHA-256 verification failed: {remote_path}")
    partial.rename(destination)
    partial_metadata.unlink()
    return {
        **asdict(record),
        "sha256": remote_sha256,
        "local_path": destination.relative_to(local_root).as_posix(),
    }, True


def _available_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def synchronize(
    sftp: paramiko.SFTPClient,
    *,
    transport: paramiko.Transport,
    remote_home: str,
    run_id: str,
    local_root: Path,
    receipt_path: Path,
    host_key_sha256: str,
    run_root: PurePosixPath | None = None,
    inventory: list[RemoteFile] | None = None,
) -> SyncSummary:
    if receipt_path.exists():
        raise FileExistsError(f"synchronization receipt already exists: {receipt_path}")
    if run_root is None or inventory is None:
        run_root, inventory = inventory_remote(
            sftp, remote_home=remote_home, run_id=run_id
        )
    missing_bytes = sum(
        record.bytes
        for record in inventory
        if not local_root.joinpath(*PurePosixPath(record.relative_path).parts).exists()
    )
    if missing_bytes > _available_bytes(local_root):
        raise RuntimeError(
            f"insufficient local space: need {missing_bytes} bytes for missing files"
        )

    files: list[dict[str, Any]] = []
    downloaded_files = 0
    downloaded_bytes = 0
    for index, record in enumerate(inventory, start=1):
        verified, downloaded = _sync_one(
            sftp,
            transport=transport,
            run_root=run_root,
            record=record,
            local_root=local_root,
        )
        files.append(verified)
        if downloaded:
            downloaded_files += 1
            downloaded_bytes += record.bytes
        if index % 25 == 0 or index == len(inventory):
            print(
                f"workspace-sync {index}/{len(inventory)} files; "
                f"downloaded={downloaded_files}; verified_existing="
                f"{index - downloaded_files}",
                file=sys.stderr,
                flush=True,
            )

    payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "remote_root": run_root.as_posix(),
        "local_root": str(local_root.resolve()),
        "wrds_cloud_host_key_sha256": host_key_sha256,
        "policy": {
            "remote_scopes": ["data", "output"],
            "remote_links_followed": False,
            "existing_local_files_overwritten": False,
            "licensed_raw_rows_downloaded": True,
            "selected_security_files_downloaded": True,
            "atomic_partial_files": True,
            "sha256_verified": True,
        },
        "files": files,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_path.with_name(
        f".{receipt_path.name}.partial-{uuid4().hex}"
    )
    temporary_receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_receipt.rename(receipt_path)
    return SyncSummary(
        run_id=run_id,
        remote_root=run_root.as_posix(),
        files=len(files),
        bytes=sum(item.bytes for item in inventory),
        downloaded_files=downloaded_files,
        downloaded_bytes=downloaded_bytes,
        verified_existing_files=len(files) - downloaded_files,
        receipt=str(receipt_path.resolve()),
        wrds_cloud_host_key_sha256=host_key_sha256,
    )


def synchronize_parallel(
    *,
    remote_home: str,
    run_id: str,
    local_root: Path,
    receipt_path: Path,
    run_root: PurePosixPath,
    inventory: list[RemoteFile],
    workers: int = 8,
) -> SyncSummary:
    """Synchronize independent files over persistent parallel SSH transports."""

    if workers < 1 or workers > 16:
        raise ValueError("workers must lie in [1, 16]")
    if run_root != _remote_run_root(remote_home, run_id):
        raise RuntimeError("remote run root disagrees with the authenticated home")
    if receipt_path.exists():
        raise FileExistsError(f"synchronization receipt already exists: {receipt_path}")
    missing_bytes = sum(
        record.bytes
        for record in inventory
        if not local_root.joinpath(*PurePosixPath(record.relative_path).parts).exists()
    )
    if missing_bytes > _available_bytes(local_root):
        raise RuntimeError(
            f"insufficient local space: need {missing_bytes} bytes for missing files"
        )

    tasks: Queue[RemoteFile] = Queue()
    for record in inventory:
        tasks.put(record)
    stop = Event()
    lock = Lock()
    verified_by_path: dict[str, dict[str, Any]] = {}
    downloaded_files = 0
    downloaded_bytes = 0
    host_fingerprints: set[str] = set()

    def work() -> None:
        nonlocal downloaded_files, downloaded_bytes
        transport, _first_use, fingerprint = wrds_cloud.connect_wrds_cloud()
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            normalized_home = sftp.normalize(".")
            if _safe_remote_home(normalized_home) != _safe_remote_home(remote_home):
                raise RuntimeError("parallel WRDS session returned a different home")
            with lock:
                host_fingerprints.add(fingerprint)
            while not stop.is_set():
                try:
                    record = tasks.get_nowait()
                except Empty:
                    return
                try:
                    verified, downloaded = _sync_one(
                        sftp,
                        transport=transport,
                        run_root=run_root,
                        record=record,
                        local_root=local_root,
                    )
                    with lock:
                        verified_by_path[record.relative_path] = verified
                        if downloaded:
                            downloaded_files += 1
                            downloaded_bytes += record.bytes
                        completed = len(verified_by_path)
                        if completed % 25 == 0 or completed == len(inventory):
                            print(
                                f"workspace-sync {completed}/{len(inventory)} files; "
                                f"downloaded={downloaded_files}; verified_existing="
                                f"{completed - downloaded_files}",
                                file=sys.stderr,
                                flush=True,
                            )
                except Exception:
                    stop.set()
                    raise
                finally:
                    tasks.task_done()
        finally:
            sftp.close()
            transport.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work) for _ in range(workers)]
        for future in futures:
            future.result()
    if len(verified_by_path) != len(inventory):
        raise RuntimeError(
            f"parallel synchronization stopped after {len(verified_by_path)} "
            f"of {len(inventory)} files"
        )
    if len(host_fingerprints) != 1:
        raise RuntimeError("parallel WRDS sessions disagreed on the host key")
    host_key_sha256 = next(iter(host_fingerprints))
    files = [verified_by_path[record.relative_path] for record in inventory]
    payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "remote_root": run_root.as_posix(),
        "local_root": str(local_root.resolve()),
        "wrds_cloud_host_key_sha256": host_key_sha256,
        "policy": {
            "remote_scopes": ["data", "output"],
            "remote_links_followed": False,
            "existing_local_files_overwritten": False,
            "licensed_raw_rows_downloaded": True,
            "selected_security_files_downloaded": True,
            "atomic_partial_files": True,
            "sha256_verified": True,
            "parallel_persistent_transports": workers,
        },
        "files": files,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_path.with_name(
        f".{receipt_path.name}.partial-{uuid4().hex}"
    )
    temporary_receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_receipt.rename(receipt_path)
    return SyncSummary(
        run_id=run_id,
        remote_root=run_root.as_posix(),
        files=len(files),
        bytes=sum(item.bytes for item in inventory),
        downloaded_files=downloaded_files,
        downloaded_bytes=downloaded_bytes,
        verified_existing_files=len(files) - downloaded_files,
        receipt=str(receipt_path.resolve()),
        wrds_cloud_host_key_sha256=host_key_sha256,
    )


def verify_local(receipt_path: Path, *, local_root: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format_version") != 1 or not isinstance(receipt.get("files"), list):
        raise RuntimeError(f"invalid workspace synchronization receipt: {receipt_path}")
    failures: list[str] = []
    total_bytes = 0
    for record in receipt["files"]:
        relative = _safe_relative(PurePosixPath(str(record["local_path"])))
        path = local_root.joinpath(*PurePosixPath(relative).parts)
        expected_bytes = int(record["bytes"])
        total_bytes += expected_bytes
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != record["sha256"]
        ):
            failures.append(relative)
    return {
        "receipt": str(receipt_path.resolve()),
        "files_checked": len(receipt["files"]),
        "bytes_checked": total_bytes,
        "failures": failures,
        "all_checks_pass": not failures,
    }


def _connect_inventory(run_id: str) -> dict[str, Any]:
    transport, first_use, fingerprint = wrds_cloud.connect_wrds_cloud()
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        remote_home = sftp.normalize(".")
        root, records = inventory_remote_fast(
            transport, remote_home=remote_home, run_id=run_id
        )
    finally:
        sftp.close()
        transport.close()
    by_scope: dict[str, dict[str, int]] = {}
    for record in records:
        scope = PurePosixPath(record.relative_path).parts[0]
        aggregate = by_scope.setdefault(scope, {"files": 0, "bytes": 0})
        aggregate["files"] += 1
        aggregate["bytes"] += record.bytes
    return {
        "run_id": run_id,
        "remote_root": root.as_posix(),
        "files": len(records),
        "bytes": sum(record.bytes for record in records),
        "by_scope": by_scope,
        "first_use_key_recorded": first_use,
        "wrds_cloud_host_key_sha256": fingerprint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("inventory", "sync", "verify"),
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--local-root", type=Path, default=ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "inventory":
        result: Any = _connect_inventory(args.run_id)
    elif args.action == "verify":
        result = verify_local(args.receipt, local_root=args.local_root)
        if not result["all_checks_pass"]:
            raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
    else:
        transport, _first_use, fingerprint = wrds_cloud.connect_wrds_cloud()
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            remote_home = sftp.normalize(".")
            run_root, inventory = inventory_remote_fast(
                transport, remote_home=remote_home, run_id=args.run_id
            )
        finally:
            sftp.close()
            transport.close()
        del fingerprint
        result = synchronize_parallel(
            remote_home=remote_home,
            run_id=args.run_id,
            local_root=args.local_root,
            receipt_path=args.receipt,
            run_root=run_root,
            inventory=inventory,
        )
    print(json.dumps(asdict(result) if hasattr(result, "__dataclass_fields__") else result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
