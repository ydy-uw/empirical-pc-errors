"""Deploy one immutable Git source snapshot to WRDS Cloud."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import paramiko

from src.wrds_cloud import connect_wrds_cloud
from src.wrds_cloud_export import _run_remote_command
from wrds_client import get_username


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _mkdir_if_missing(
    sftp: paramiko.SFTPClient,
    path: PurePosixPath,
) -> None:
    try:
        sftp.stat(path.as_posix())
    except OSError:
        sftp.mkdir(path.as_posix(), mode=0o700)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-ref", default="HEAD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = _git_output("rev-parse", args.git_ref)
    if _git_output("status", "--short", "--untracked-files=no"):
        raise RuntimeError("Refusing to deploy tracked worktree changes")

    archive_root = ROOT / "tmp" / "wrds_cloud_deployments"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"{commit}.tar.gz"
    if not archive.exists():
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                f"--output={archive}",
                commit,
                "src",
                "tests",
                "config",
                "wrds_client.py",
                "requirements.txt",
                "README.md",
            ],
            cwd=ROOT,
            check=True,
        )
    archive_hash = _sha256(archive)

    transport, first_use, host_fingerprint = connect_wrds_cloud()
    transport.set_keepalive(30)
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            remote_home = PurePosixPath(sftp.normalize("."))
            projects = remote_home / "empirical_jse_projects"
            _mkdir_if_missing(sftp, projects)
            project = projects / commit
            try:
                sftp.stat(project.as_posix())
            except OSError:
                sftp.mkdir(project.as_posix(), mode=0o700)
            else:
                raise FileExistsError(
                    f"Immutable remote project already exists: {project}"
                )

            remote_archive = project / "source.tar.gz"
            sftp.put(archive.as_posix(), remote_archive.as_posix())
            remote_archive_hash = _sha256(archive)
            if remote_archive_hash != archive_hash:
                raise RuntimeError("Local deployment archive changed during upload")

            wrds_config = project / ".wrds"
            sftp.mkdir(wrds_config.as_posix(), mode=0o700)
            username_path = wrds_config / "username"
            with sftp.file(username_path.as_posix(), mode="wx") as output:
                output.write((get_username() + "\n").encode("utf-8"))
                output.flush()
        finally:
            sftp.close()

        quoted_project = shlex.quote(project.as_posix())
        quoted_archive = shlex.quote(remote_archive.as_posix())
        command = (
            "set -euo pipefail\n"
            f"tar -xzf {quoted_archive} -C {quoted_project}\n"
            f"cd {quoted_project}\n"
            "python3 -m compileall -q src\n"
            "python3 -c 'import numpy, pandas, scipy, pyarrow, matplotlib, wrds'\n"
            f"sha256sum {quoted_archive}\n"
        )
        result = _run_remote_command(
            transport,
            "bash -lc " + shlex.quote(command),
        )
        if result.exit_status:
            raise RuntimeError(
                f"Remote deployment validation failed: {result.stderr.strip()}"
            )
        remote_digest = result.stdout.split()[0]
        if remote_digest != archive_hash:
            raise RuntimeError("Remote source archive SHA-256 mismatch")
    finally:
        transport.close()

    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "local_archive": archive.relative_to(ROOT).as_posix(),
        "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "remote_project": project.as_posix(),
        "remote_archive": remote_archive.as_posix(),
        "wrds_cloud_host_key_sha256": host_fingerprint,
        "first_use_host_key_recorded": first_use,
    }
    manifest_root = ROOT / "notes" / "wrds_cloud_deployments"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{commit}.json"
    if manifest_path.exists():
        raise FileExistsError(f"Deployment manifest exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
