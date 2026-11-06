# 03 — Building the data: WRDS → raw snapshots → Parquet → rolling runs

Everything downstream (companion, wpca, research_tools, the July and August
analyses) reads the output of **two rolling runs**: one with top-cap universes
and one with nested-random universes. This document shows how to produce them.

All commands are run from `emp/` with the virtual environment active. Relative
paths are relative to `emp/`.

## 1. Data flow at a glance

```
WRDS (crsp_a_stock)
  │  download_crsp_reference        calendar, delistings, flag dictionary, market index (1972–2025)
  │  download_crsp_collection       dsf_v2 daily rows, one immutable snapshot per year (2005–2025)
  ▼
data/raw/wrds/crsp_ciz_reference/<reference-id>/        manifest.json + calendar/ delists/ flags/ market/ (csv.gz)
data/raw/wrds/crsp_ciz/annual/<prefix>_<year>/          manifest.json + daily/year=<year>/dsf_v2.csv.gz
data/raw/wrds/crsp_ciz/annual/collections/<collection-id>/collection_manifest.json
  │  prepare_crsp_parquet           type normalization, QA, one Parquet file per year
  ▼
data/interim/crsp_ciz_parquet/<dataset-id>/year=<year>/daily.parquet  (+ manifests, dataset_spec.json)
  │  run_rolling_experiments        239 month-ends × 4 sizes = 956 panels
  ▼
data/processed/rolling/<rolling-id>/year=<year>/        10 files per year (docs/04)
data/processed/rolling/<rolling-id>/{run_spec.json, collection_manifest.json}
```

## 2. The identifiers the paper used

Use the same IDs and every default path in `companion/`, `wpca/` and
`research_tools/` will work without flags.

| object | ID |
|---|---|
| reference snapshot | `reference_1972_2025_remote_v1_20260723` |
| raw snapshot prefix (one snapshot per year) | `crsp_ciz_remote_v1_20260723` → `crsp_ciz_remote_v1_20260723_2005` … `_2025` |
| raw collection | `crsp_ciz_2005_2025_remote_v1_20260723` |
| Parquet dataset | `crsp_ciz_2005_2025_remote_v1_20260723` |
| top-cap rolling run | `rolling_topcap_200601_202511_remote_v2_20260724` |
| nested-random rolling run | `rolling_nested_random_top3000_200601_202511_v1_20260811` |

About the names:

- `remote` records that the original was extracted on WRDS Cloud.
- `v2` of the top-cap run records one fix relative to `v1`. In v2 the Jung
  count is selected from the 63-day formation window only, before any
  out-of-sample check (`run_spec.json → protocol.jung_selection_scope =
  formation_63_day_only`). v1 and v2 have SHA-identical `selected_universes`,
  `observable_error_floors`, `error_decomposition_diagnostics`,
  `extreme_events` and `qa_records` files in all 20 years. They differ in the
  Jung-dependent files (`factor_*`, `variance_comparisons`) and, through a
  schema change, in `skipped_cells.json`. The current code is v2 plus the
  nested-random design option added on 11 Aug. Re-running two years of the top-cap run
  with the packaged code on the original Parquet data reproduced the v2 files
  (docs/10 §1).

## 3. Choose a route

| route | what runs where | when to use it |
|---|---|---|
| **A. Local** | your machine talks to WRDS PostgreSQL directly and does everything else locally | simplest; recommended for a collaborator |
| **B. Local + WRDS Cloud export** | WRDS Cloud runs the SQL, gzips the result server-side and you download it over SSH; everything else local | if route A's download is too slow for you |
| **C. WRDS Cloud grid** | extraction, Parquet conversion and the rolling run all run as SGE jobs on WRDS Cloud; you mirror the results back | what the original run did. More moving parts (git deploy, Duo per connection) |

All three produce the same directory layout and formats.

### Route A — everything local

```bash
# A1. Reference tables: 4 small files
python -m src.download_crsp_reference \
    --start 1972-01-01 --end 2025-12-31 \
    --snapshot-id reference_1972_2025_remote_v1_20260723

# A2. Daily CIZ rows, one immutable snapshot per year; a collection manifest at the end
python -m src.download_crsp_collection \
    --start-year 2005 --end-year 2025 \
    --snapshot-prefix crsp_ciz_remote_v1_20260723 \
    --collection-id crsp_ciz_2005_2025_remote_v1_20260723
#   optional: --chunksize (rows per fetch, default 250000)

# A3. Convert to Parquet with extract QA
python -m src.prepare_crsp_parquet \
    --start-year 2005 --end-year 2025 \
    --snapshot-prefix crsp_ciz_remote_v1_20260723 \
    --dataset-id crsp_ciz_2005_2025_remote_v1_20260723

# A4. Top-cap rolling run (the main panel)
python -m src.run_rolling_experiments \
    --dataset-root data/interim/crsp_ciz_parquet/crsp_ciz_2005_2025_remote_v1_20260723 \
    --reference-snapshot data/raw/wrds/crsp_ciz_reference/reference_1972_2025_remote_v1_20260723 \
    --run-id rolling_topcap_200601_202511_remote_v2_20260724

# A5. Nested-random rolling run (needed by the companion's quick stage: emp_kfixed, emp_random; and by kcount_extra)
#     The nested design is a command-line option, not part of config/research.yml.
python -m src.run_rolling_experiments \
    --dataset-root data/interim/crsp_ciz_parquet/crsp_ciz_2005_2025_remote_v1_20260723 \
    --reference-snapshot data/raw/wrds/crsp_ciz_reference/reference_1972_2025_remote_v1_20260723 \
    --run-id rolling_nested_random_top3000_200601_202511_v1_20260811 \
    --selection-design top3000_nested_random            # seed defaults to 20260811
```

Notes on each step:

- **A2 is resumable.** Each year is built in a hidden staging directory
  (`.<snapshot>.staging-<hex>`) and renamed into place only when complete, so
  an interrupted year leaves only a staging directory behind (never deleted
  automatically; remove it by hand). On a re-run, finished years are
  re-verified against their manifests and reused; a year that fails
  verification stops the run. The collection manifest is written only after
  all years succeed, and the script refuses to run if it already exists. The SQL prefilters to PERMNOs that were
  ever conventional U.S. common stock on NYSE, NYSE American or Nasdaq in the
  year or the year before. It returns *all* their daily rows, including halts
  and terminal rows, and leaves the exact point-in-time filters to the rolling
  run (`crsp.build_dsf_v2_query`).
- **A3** writes `year=<y>/daily.parquet` plus per-year manifests, a
  `dataset_spec.json` and a `collection_manifest.json`. It checks each raw
  snapshot's SHA-256 before converting. Expect 0.91–1.22 million rows and
  15–23 MB per year.
- **A4/A5** loop over years. Each year is written to a staging directory and
  renamed to `year=<y>/` with its own `manifest.json`. On a re-run, finished
  years that verify against the same run signature are reused, so an
  interrupted run can be restarted with the same command. The **run
  signature** covers the code, the config, the input manifests, the resolved
  absolute input paths, and the Python and package versions. So a restart
  must also use the same environment and the same data locations. Otherwise
  the script stops with "run-id is already bound to different inputs,
  parameters, or code". A year that fails verification stops the run: move it
  to `legacy/` or choose a new run ID. The run-level `collection_manifest.json`
  is written last. Once it exists, the run is complete and the script refuses
  to touch it.
- **Time.** One year of the rolling run took 1–2 minutes on a 2-core machine,
  about 40 minutes per run in total.
- The rolling run checks that the constants compiled into the code (window 63,
  Jung α and limits, the fixed-q grid, localization gates) equal
  `config/research.yml`. It stops on any drift. It also refuses a sample or
  size grid different from the config unless you pass
  `--allow-config-override`. Use that only for a labelled robustness run, and
  use `--start/--end/--start-year/--end-year/--sizes` to define it.
- `--output-root` (default `data/processed/rolling`) and `--config` (default
  `config/research.yml`) can be changed. The companion expects the default
  location.

### Route B — local, with WRDS Cloud doing the SQL export

Replace A2 with:

```bash
python -m src.probe_wrds_cloud          # once: SSH login test, pins the host key
python -m src.download_crsp_cloud_collection \
    --start-year 2005 --end-year 2025 \
    --snapshot-prefix crsp_ciz_remote_v1_20260723 \
    --collection-id crsp_ciz_2005_2025_remote_v1_20260723
```

For each year it runs `COPY (<the same query>) TO STDOUT | gzip -n -9` on WRDS
Cloud with `psql`. It checks the archive there, downloads it over SFTP and
verifies it locally. The snapshot layout is identical to route A
(`daily/year=<y>/dsf_v2.csv.gz`), so A3–A5 run unchanged. You will get a Duo
push for the SSH login. `python -m src.download_crsp_cloud_ciz` does a single
year or date range.

### Route C — on the WRDS Cloud grid (how the original run was made)

Prerequisites: docs/02 §3 (SSH, Duo, host key, `git init` in `emp/`).

```bash
# C0. Ship an immutable source snapshot (a git commit) to ~/empirical_jse_projects/<commit> on WRDS Cloud
python -m src.deploy_wrds_cloud_project                 # --git-ref HEAD by default
COMMIT=$(git rev-parse HEAD)
RUN=empirical_topcap_2005_2025_v2_20260723              # or any new run name you like

# C1. Download + reference + Parquet as one grid job (IDs are fixed inside the job script)
python -m src.submit_wrds_cloud_phase --phase extract_prepare --git-commit $COMMIT --run-id $RUN

# C2. Rolling runs, submitted only AFTER the C1 job has finished (the job script starts with
#     `test -f status/extract_prepare.complete` and dies immediately otherwise; no qsub dependency is set).
#     Pass the rolling IDs explicitly: the script's default is the old v1 name.
python -m src.submit_wrds_cloud_phase --phase rolling --git-commit $COMMIT --run-id $RUN \
    --rolling-id rolling_topcap_200601_202511_remote_v2_20260724 --submission-tag topcap_v2
python -m src.submit_wrds_cloud_phase --phase rolling --git-commit $COMMIT --run-id $RUN \
    --rolling-id rolling_nested_random_top3000_200601_202511_v1_20260811 \
    --selection-design top3000_nested_random --submission-tag nested_v1

# Watch a job (prints qstat, the status/ directory and the log tails)
python -m src.monitor_wrds_cloud_run --run-id $RUN --phase rolling_topcap_v2 --job-id <id printed by submit>

# C3. Bring the whole run home (data/ and output/ trees, SHA-256 verified, resumable)
python -m src.sync_wrds_cloud_workspace inventory --run-id $RUN --receipt notes/wrds_cloud_runs/$RUN/local_workspace_mirror_v1.json
python -m src.sync_wrds_cloud_workspace sync      --run-id $RUN --receipt notes/wrds_cloud_runs/$RUN/local_workspace_mirror_v1.json
python -m src.sync_wrds_cloud_workspace verify    --run-id $RUN --receipt notes/wrds_cloud_runs/$RUN/local_workspace_mirror_v1.json
```

What to know about route C:

- **Remote layout.** Sources live in `~/empirical_jse_projects/<commit>/`. Runs
  live in `~/empirical_jse_runs/<run-id>/`, with the same `data/` and
  `output/` layout as locally, plus `status/*.started|complete` markers and
  `job_*.sh`, `*.out` and `*.err`.
- **Job scripts.** They are written once and never overwritten. Each job runs
  `python3 -m src.…` with `PYTHONPATH=<project>` under
  `/gridware/sge/bin/lx-amd64/qsub`. The exact commands are in
  `src/submit_wrds_cloud_phase.py: _phase_script`.
- **Receipts.** Every submit writes a receipt under
  `notes/wrds_cloud_runs/<run-id>/`, and the deploy writes one under
  `notes/wrds_cloud_deployments/`. The July report builder
  (`build_report_artifacts`) insists on these receipts (docs/05). They are not
  in this package because they record the original author's account.
- **Grid environment.** The original grid jobs ran on the WRDS Cloud system
  Python: 3.14.5 with numpy 2.4.6, scipy 1.17.1, pyarrow 24.0.0 (recorded in
  `run_spec.json → runtime`). A full local replay with the pinned environment,
  on 12 Aug, reproduced all 239 months of both runs. Discrete columns matched
  exactly and floats agreed to 5e-11 (`src/cross_validate_rolling_replay.py`).
- `sync_wrds_cloud_workspace` mirrors **everything**, including licensed
  security-level files. The narrower `sync_wrds_cloud_results`,
  `sync_wrds_cloud_observable_floors` and friends download only aggregates.
  They were built for the original run and its provenance bundle (docs/05).

## 4. What one rolling run computes

For every formation month-end t (2006-01 … 2025-11) and every P in
{500, 1000, 2000, 3000}, `run_rolling_experiments` does the following. The
work happens in `src/run_rolling_experiments.py` and
`src/empirical_jse/{crsp,experiment,qa}.py`.

1. **Universe.** Apply the point-in-time eligibility rules on t
   (`config/research.yml → universe`). Represent each PERMCO by its
   largest-cap eligible PERMNO, rank by total company cap, and take the top P
   (top-cap). For the nested design, take the top 3,000 and draw one random
   permutation of it per month, seeded by (master seed 20260811, year, month).
   Its first 500/1000/2000/3000 names are the nested universes, so P = 3,000
   equals top-cap. →
   `selected_universes.parquet`.
2. **Formation panel.** The 63 × P matrix of CIZ `dlyret` over the 63 trading
   days ending on t. Every security must have a finite return on all 63 days.
   Panel QA goes to `qa_records.json`.
3. **Factor count from the formation window only.** Jung's test on the
   centered panel at α ∈ {0.05, 0.10, 0.20}, up to 30 factors (50 on retry). →
   `factor_selections.parquet`, `factor_diagnostics.parquet`, and the full
   residual-length paths `factor_residual_lengths.parquet`. This is written for
   **all 956 panels**.
4. **Next-month evaluation (July study).** Build the out-of-sample month. If a
   security's terminal return is incomplete there, the whole
   (month, P) cell is skipped and recorded in `skipped_cells.json` (140 cells
   in the paper's run). For the other 816 panels, compute the floors A_j
   (centered, uncentered, MP-adjusted) for every fixed q and for the Jung q,
   and write them to `observable_error_floors.parquet`. Also compute the
   JSE/H♯ corrected versus uncorrected minimum-variance portfolios and their
   realized variances (`variance_comparisons.parquet`), the odd/even
   cross-fit error decomposition (`error_decomposition_diagnostics.parquet`),
   and extreme single-name return events (`extreme_events.parquet`).

This is why the September companion rebuilds the centered floor from the
residual-length paths (step 3). That way it covers all 956 panels, not only
the 816 that passed step 4's gate.

## 5. Checking your build

- Every directory has a `manifest.json` with per-file SHA-256. The loaders
  verify it (`dataset.ParquetCIZCollection`, `analyze_rolling_results.verify_and_load_run`).
- Expected shape: 239 formation dates; 956 (date, P) cells in
  `factor_selections` and `selected_universes`; 816 cells in
  `observable_error_floors`; no censored Jung selections at α = 0.10.
- `python companion/code/paths.py` (from the package root) reports whether the
  three directories the companion needs exist and are complete (have a
  `collection_manifest.json`). It exits non-zero otherwise.
- A one-line check of the counts, from `emp/`:

  ```bash
  python -c "import pandas as pd, glob; r='data/processed/rolling/rolling_topcap_200601_202511_remote_v2_20260724'; s=pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(r+'/year=*/factor_selections.parquet'))); f=pd.concat(pd.read_parquet(f, columns=['formation_date','n_assets']) for f in sorted(glob.glob(r+'/year=*/observable_error_floors.parquet'))); print(s.formation_date.nunique(), len(s[['formation_date','n_assets']].drop_duplicates()), len(f.drop_duplicates()), int(s[s.alpha.round(2)==0.10].censored.sum()))"
  # expect: 239 956 816 0
  ```
- Your numbers will not be bit-identical to ours, because CRSP revises
  history. `docs/10_reference_values.md` lists the original values. The
  quickest comparison is `cd companion && bash run_all.sh quick`, which
  writes `out/emp_kfixed.json` (docs/10 §3) and `out/floor_vs_m.parquet`
  (docs/10 §4). Then `bash run_all.sh run_rawdiag2 emp_main`, about 5 min,
  gives the headline levels in `out/emp_main.json` (docs/10 §2).

## 6. Optional: the July analysis of a rolling run

```bash
python -m src.analyze_rolling_results \
    --run-root data/processed/rolling/rolling_topcap_200601_202511_remote_v2_20260724 \
    --analysis-id analysis_topcap_200601_202511_remote_v4_20260724
```

This verifies the run and writes the July tables and figures to
`output/analysis/<analysis-id>/`, with 10,000-draw, 12-month block-bootstrap
intervals. The companion does not need it. docs/05 describes what builds on it.
