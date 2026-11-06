"""Read-only status monitor for an immutable WRDS Cloud grid run."""

from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath
import re

import paramiko

from src.wrds_cloud import connect_wrds_cloud
from src.wrds_cloud_export import _run_remote_command


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_PHASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_JOB_ID = re.compile(r"^[0-9]+$")
_QSTAT = "/gridware/sge/bin/lx-amd64/qstat"
_QSTAT_SUMMARY_FIELDS = {
    "job_number",
    "job_name",
    "start_time",
    "job_state",
    "exec_host_list",
    "usage",
    "scheduling info",
}


def _tail_text(
    sftp: paramiko.SFTPClient,
    path: PurePosixPath,
    *,
    max_bytes: int,
) -> str | None:
    try:
        size = sftp.stat(path.as_posix()).st_size
    except OSError:
        return None
    with sftp.file(path.as_posix(), mode="rb") as handle:
        handle.seek(max(size - max_bytes, 0))
        content = handle.read()
    return content.decode("utf-8", errors="replace")


def _summarize_qstat(text: str) -> list[str]:
    summary = []
    for line in text.splitlines():
        field = line.split(":", maxsplit=1)[0].rstrip(" 0123456789,")
        if field in _QSTAT_SUMMARY_FIELDS:
            summary.append(line)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-tail-bytes", type=int, default=32_768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError("unsafe --run-id")
    if not _SAFE_PHASE.fullmatch(args.phase):
        raise ValueError("unsafe --phase")
    if not _SAFE_JOB_ID.fullmatch(args.job_id):
        raise ValueError("unsafe --job-id")
    if not 1 <= args.log_tail_bytes <= 1_000_000:
        raise ValueError("--log-tail-bytes must be in [1, 1000000]")

    transport, _first_use, fingerprint = connect_wrds_cloud()
    try:
        qstat = _run_remote_command(
            transport,
            f"{_QSTAT} -j {args.job_id}",
        )
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            home = PurePosixPath(sftp.normalize("."))
            run_root = home / "empirical_jse_runs" / args.run_id
            status_root = run_root / "status"
            try:
                status_files = sorted(item.filename for item in sftp.listdir_attr(
                    status_root.as_posix()
                ))
            except OSError:
                status_files = []
            stdout_tail = _tail_text(
                sftp,
                run_root / f"{args.phase}.out",
                max_bytes=args.log_tail_bytes,
            )
            stderr_tail = _tail_text(
                sftp,
                run_root / f"{args.phase}.err",
                max_bytes=args.log_tail_bytes,
            )
        finally:
            sftp.close()
    finally:
        transport.close()

    payload = {
        "format_version": 1,
        "run_id": args.run_id,
        "phase": args.phase,
        "job_id": args.job_id,
        "host_key_sha256": fingerprint,
        "qstat_exit_status": qstat.exit_status,
        "qstat_summary": _summarize_qstat(qstat.stdout),
        "qstat_stderr": qstat.stderr,
        "status_files": status_files,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
