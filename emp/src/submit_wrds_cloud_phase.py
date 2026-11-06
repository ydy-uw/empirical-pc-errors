"""Submit one fixed empirical-pipeline phase to the WRDS Cloud grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shlex

import paramiko

from src.wrds_cloud import connect_wrds_cloud
from src.wrds_cloud_export import _run_remote_command


ROOT = Path(__file__).resolve().parents[1]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RAW_PREFIX = "crsp_ciz_remote_v1_20260723"
RAW_COLLECTION = "crsp_ciz_2005_2025_remote_v1_20260723"
REFERENCE_ID = "reference_1972_2025_remote_v1_20260723"
DATASET_ID = "crsp_ciz_2005_2025_remote_v1_20260723"
ROLLING_ID = "rolling_topcap_200601_202511_remote_v1_20260723"
ANALYSIS_ID = "analysis_topcap_200601_202511_remote_v1_20260723"


def _write_new_remote(
    sftp: paramiko.SFTPClient,
    path: PurePosixPath,
    content: str,
) -> None:
    try:
        sftp.stat(path.as_posix())
    except OSError:
        pass
    else:
        raise FileExistsError(f"Remote path already exists: {path}")
    with sftp.file(path.as_posix(), mode="wx") as output:
        output.write(content.encode("utf-8"))
        output.flush()


def _phase_script(
    phase: str,
    *,
    project: PurePosixPath,
    run_root: PurePosixPath,
    analysis_id: str = ANALYSIS_ID,
    rolling_id: str = ROLLING_ID,
    status_key: str | None = None,
    selection_design: str = "top_cap",
    selection_seed: int = 20260811,
    allow_incomplete_primary_inference: bool = False,
) -> str:
    status_name = status_key or phase
    project_q = shlex.quote(project.as_posix())
    run_q = shlex.quote(run_root.as_posix())
    rolling_id_q = shlex.quote(rolling_id)
    preamble = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "umask 077\n"
        f"project_path={project_q}\n"
        f"run_path={run_q}\n"
        'cd "$project_path"\n'
        'export PYTHONPATH="$project_path"\n'
        'mkdir -p "$run_path/status"\n'
        f"date -u +%FT%TZ > \"$run_path/status/{status_name}.started\"\n"
    )
    if phase == "extract_prepare":
        body = f"""
python3 -m src.download_crsp_reference \
  --start 1972-01-01 \
  --end 2025-12-31 \
  --snapshot-id {REFERENCE_ID} \
  --raw-root "$run_path/data/raw/wrds/crsp_ciz_reference"

python3 -m src.download_crsp_collection \
  --start-year 2005 \
  --end-year 2025 \
  --snapshot-prefix {RAW_PREFIX} \
  --collection-id {RAW_COLLECTION} \
  --raw-root "$run_path/data/raw/wrds/crsp_ciz/annual" \
  --chunksize 250000

python3 -m src.prepare_crsp_parquet \
  --start-year 2005 \
  --end-year 2025 \
  --snapshot-prefix {RAW_PREFIX} \
  --dataset-id {DATASET_ID} \
  --raw-root "$run_path/data/raw/wrds/crsp_ciz/annual" \
  --output-root "$run_path/data/interim/crsp_ciz_parquet"
""".lstrip()
    elif phase == "rolling":
        selection_args = ""
        if selection_design == "top3000_nested_random":
            selection_args = (
                " \\\n  --selection-design top3000_nested_random"
                f" \\\n  --selection-seed {int(selection_seed)}"
            )
        elif selection_design != "top_cap":
            raise ValueError("unsupported selection design")
        body = f"""
test -f "$run_path/status/extract_prepare.complete"
python3 -m src.run_rolling_experiments \
  --dataset-root "$run_path/data/interim/crsp_ciz_parquet/{DATASET_ID}" \
  --reference-snapshot \
    "$run_path/data/raw/wrds/crsp_ciz_reference/{REFERENCE_ID}" \
  --run-id {rolling_id_q} \
  --output-root "$run_path/data/processed/rolling"{selection_args}
""".lstrip()
    elif phase == "analyze":
        inference_args = (
            " \\\n  --allow-incomplete-primary-inference"
            if allow_incomplete_primary_inference
            else ""
        )
        rolling_dependency = (
            'test -f "$run_path/status/rolling.complete"\n'
            if rolling_id == ROLLING_ID
            else (
                "test -f "
                f'"$run_path/data/processed/rolling/{rolling_id_q}/'
                'collection_manifest.json"\n'
            )
        )
        body = f"""
{rolling_dependency.rstrip()}
python3 -m src.analyze_rolling_results \
  --run-root "$run_path/data/processed/rolling/{rolling_id_q}" \
  --analysis-id {analysis_id} \
  --output-root "$run_path/output/analysis" \
  --bootstrap-draws 10000 \
  --block-length 12 \
  --seed 20260723{inference_args}
""".lstrip()
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    return (
        preamble
        + body
        + f'date -u +%FT%TZ > "$run_path/status/{status_name}.complete"\n'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("extract_prepare", "rolling", "analyze"),
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--analysis-id", default=ANALYSIS_ID)
    parser.add_argument("--rolling-id", default=ROLLING_ID)
    parser.add_argument(
        "--selection-design",
        choices=("top_cap", "top3000_nested_random"),
        default="top_cap",
    )
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument(
        "--allow-incomplete-primary-inference",
        action="store_true",
        help=(
            "For analysis only, retain descriptive results when a primary "
            "cell lacks enough months for block-bootstrap inference."
        ),
    )
    parser.add_argument(
        "--submission-tag",
        help=(
            "Safe suffix for an additional immutable rolling or analysis "
            "submission within the same empirical run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not _SAFE_ID.fullmatch(args.git_commit):
        raise ValueError("unsafe --git-commit")
    if not _SAFE_ID.fullmatch(args.run_id):
        raise ValueError("unsafe --run-id")
    if not _SAFE_ID.fullmatch(args.analysis_id):
        raise ValueError("unsafe --analysis-id")
    if not _SAFE_ID.fullmatch(args.rolling_id):
        raise ValueError("unsafe --rolling-id")
    if args.submission_tag is not None and not _SAFE_ID.fullmatch(
        args.submission_tag
    ):
        raise ValueError("unsafe --submission-tag")
    if args.phase != "analyze" and args.analysis_id != ANALYSIS_ID:
        raise ValueError(
            "--analysis-id is only valid for analyze"
        )
    if args.phase != "rolling" and (
        args.selection_design != "top_cap"
        or args.selection_seed != 20260811
    ):
        raise ValueError(
            "selection arguments are only valid for the rolling phase"
        )
    if args.phase == "extract_prepare" and (
        args.rolling_id != ROLLING_ID or args.submission_tag is not None
    ):
        raise ValueError(
            "--rolling-id and --submission-tag are not valid for "
            "extract_prepare"
        )

    transport, _first_use, fingerprint = connect_wrds_cloud()
    transport.set_keepalive(30)
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            home = PurePosixPath(sftp.normalize("."))
            project = home / "empirical_jse_projects" / args.git_commit
            sftp.stat(project.as_posix())
            runs_root = home / "empirical_jse_runs"
            try:
                sftp.stat(runs_root.as_posix())
            except OSError:
                sftp.mkdir(runs_root.as_posix(), mode=0o700)
            run_root = runs_root / args.run_id
            if args.phase == "extract_prepare":
                try:
                    sftp.stat(run_root.as_posix())
                except OSError:
                    sftp.mkdir(run_root.as_posix(), mode=0o700)
                else:
                    raise FileExistsError(
                        f"Immutable remote run already exists: {run_root}"
                    )
            else:
                sftp.stat(run_root.as_posix())

            suffix = (
                f"_{args.submission_tag}"
                if args.submission_tag is not None
                else ""
            )
            status_key = f"{args.phase}{suffix}"
            job_script = run_root / f"job_{status_key}.sh"
            _write_new_remote(
                sftp,
                job_script,
                _phase_script(
                    args.phase,
                    project=project,
                    run_root=run_root,
                    analysis_id=args.analysis_id,
                    rolling_id=args.rolling_id,
                    status_key=status_key,
                    selection_design=args.selection_design,
        selection_seed=args.selection_seed,
        allow_incomplete_primary_inference=(
            args.allow_incomplete_primary_inference
        ),
                ),
            )
        finally:
            sftp.close()

        job_name = {
            "extract_prepare": "ejse_ep",
            "rolling": "ejse_roll",
            "analyze": "ejse_an",
        }[args.phase]
        if args.submission_tag is not None:
            job_name = f"ejse_{args.submission_tag}"[:15]
        stdout_path = run_root / f"{status_key}.out"
        stderr_path = run_root / f"{status_key}.err"
        submit = (
            "cd "
            + shlex.quote(run_root.as_posix())
            + " && /gridware/sge/bin/lx-amd64/qsub"
            + " -terse -V -cwd"
            + " -N "
            + shlex.quote(job_name)
            + " -o "
            + shlex.quote(stdout_path.as_posix())
            + " -e "
            + shlex.quote(stderr_path.as_posix())
            + " "
            + shlex.quote(job_script.name)
        )
        result = _run_remote_command(transport, submit)
        if result.exit_status:
            raise RuntimeError(f"qsub failed: {result.stderr.strip()}")
        job_id = result.stdout.strip()
        if not job_id:
            raise RuntimeError("qsub returned an empty job id")
    finally:
        transport.close()

    submission = {
        "format_version": 1,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "analysis_id": args.analysis_id if args.phase == "analyze" else None,
        "rolling_id": (
            args.rolling_id
            if args.phase in {"rolling", "analyze"}
            else None
        ),
        "selection_design": (
            args.selection_design if args.phase == "rolling" else None
        ),
        "selection_seed": (
            args.selection_seed
            if args.phase == "rolling"
            and args.selection_design == "top3000_nested_random"
            else None
        ),
        "submission_tag": args.submission_tag,
        "git_commit": args.git_commit,
        "run_id": args.run_id,
        "job_id": job_id,
        "remote_project": project.as_posix(),
        "remote_run_root": run_root.as_posix(),
        "remote_job_script": job_script.as_posix(),
        "remote_stdout": stdout_path.as_posix(),
        "remote_stderr": stderr_path.as_posix(),
        "wrds_cloud_host_key_sha256": fingerprint,
    }
    local_root = ROOT / "notes" / "wrds_cloud_runs" / args.run_id
    local_root.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"_{args.submission_tag}"
        if args.submission_tag is not None
        else ""
    )
    local_manifest = local_root / f"{args.phase}{suffix}_submission.json"
    if local_manifest.exists():
        raise FileExistsError(f"Local submission exists: {local_manifest}")
    local_manifest.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
