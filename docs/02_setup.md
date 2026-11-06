# 02 — Setup: Python, WRDS credentials, WRDS Cloud, tests

## 1. Python environment

The original work ran on Windows 11 with **Python 3.13** in a virtual
environment inside `emp/`. The exact versions are pinned in
`emp/requirements.txt`:

```
wrds==3.5.0  paramiko==5.0.0  numpy==2.5.1  pandas==2.2.3  scipy==1.18.0
pyarrow==25.0.0  matplotlib==3.11.1  seaborn==0.13.2  PyYAML==6.0.3  pytest==9.1.1
```

The same pins install cleanly on Linux with Python 3.13. The test suite was
re-run there while this package was assembled. macOS should behave the same.

```bash
cd hl_hh/emp
python3.13 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**One environment for the whole package.** `companion/`, `wpca/` and
`research_tools/` need nothing beyond these pins (numpy, pandas, scipy,
pyarrow, matplotlib). Activate the same `.venv` for them.

**Optional extras** are needed only for specific scripts:

| package / tool | needed by |
|---|---|
| `psycopg2-binary` (installed automatically by `wrds`) | `emp/configure_wrds.py` |
| `Pillow` (installed with matplotlib) | `research_tools/audit_empirical_estimator_paths.py`, `emp/src/cross_validate_rolling_analysis.py`, `emp/src/make_pdf_contact_sheets.py` |
| `reportlab` | `research_tools/build_*_pdf.py` (short August PDF notes) |
| `pypdf` | `emp/src/extract_pdf_text.py` (a literature helper; not needed for any result) |
| a TeX distribution (`pdflatex`, `bibtex`) and Poppler (`pdfinfo`) | `companion/build/build.sh`; `emp/src/finalize_report_pdf.py` and its test |

`emp/requirements-optional.txt` lists the pip-installable ones.

**Python version.** The pins need **Python ≥ 3.12**: numpy 2.5.1 and scipy
1.18.0 do not install on 3.11. The code itself only needs ≥ 3.10 (`X | Y`
type hints, `zip(strict=True)`). With 3.10 or 3.11 you would have to relax the
pins, which is untested. Floating-point results can differ between
environments in the last few digits. That is at the level the project's own
cross-machine audits tolerate (5e-11).

Under Python 3.12+, `companion/code/latex.py` and `sim.py` print a
`SyntaxWarning: invalid escape sequence` the first time they are compiled.
The strings are LaTeX and docstrings, and the warning is harmless.

**Windows.** `emp/` runs natively. The command blocks in these docs are bash:
`\` line continuations, `export`, `$(…)`, `bash run_all.sh`, `build.sh`.
Use Git Bash or WSL for them, or translate them for PowerShell:

- continue a line with a backtick instead of `\`;
- set a variable with `$env:HLHH_EMP_DATA = 'D:\data\emp\data'` instead of `export`;
- call a script with `.\.venv\Scripts\python.exe`.

If PowerShell refuses to run `Activate.ps1`, allow it for the session with
`Set-ExecutionPolicy -Scope Process RemoteSigned`.

**Threads.** numpy and scipy use a multi-threaded BLAS. On a shared machine,
or when you run several scripts at once, set `OPENBLAS_NUM_THREADS=1` (and
`MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). Oversubscription can make a
1-minute script take 6. Results change only at round-off level.

## 2. WRDS credentials (needed only to build the data)

You need a WRDS account whose subscription includes CRSP. The code reads
`crsp_a_stock.dsf_v2`, `stksecurityinfohist`, `metaexchangecalendar`,
`stkdelists`, `wrds_dailyindexret_query` and `metaflaginfo`.

Run once per machine, from `emp/` with the venv active:

```bash
python configure_wrds.py
```

It asks for your WRDS username and password (and ends with "Press Enter to
close this window"). It checks them against
`wrds-pgdata.wharton.upenn.edu:9737` and lets the official `wrds` client
write the standard PostgreSQL password file:

- Windows: `%APPDATA%\postgresql\pgpass.conf`
- macOS / Linux: `~/.pgpass`

The password is **never written into the project**. The username goes into
`emp/.wrds/username`, and a status file goes into `emp/.wrds/setup_status.json`.
`.wrds/` is git-ignored. Despite its docstring, which says "Windows machine",
the script is cross-platform. `setup_wrds.ps1` is only a Windows convenience
wrapper that runs it with `.venv\Scripts\python.exe`.

Check it worked:

```bash
python wrds_client.py      # "Connected to WRDS as <you>.", the number of accessible libraries, and the first 12 names
```

Every download script gets its connection from `wrds_client.connect()`,
which reads `.wrds/username` and relies on the pgpass file for the password.
WRDS may send a Duo push on the first connection of a session.

`wrds_client.py` only lists libraries. Before starting a long download, check
that you can read the table itself:

```bash
python -c "from wrds_client import connect; db = connect(); print(db.raw_sql('select count(*) from crsp_a_stock.dsf_v2 where dlycaldt = %(d)s', params={'d': '2025-12-31'})); db.close()"
```

## 3. WRDS Cloud (optional; only for the grid route in docs/03)

The original frozen run was computed on the WRDS Cloud SGE grid. That route
needs SSH access to `wrds-cloud.wharton.upenn.edu`. The helpers in
`emp/src/wrds_cloud.py` do the following:

- They read the password from the same pgpass file. On macOS and Linux this is
  `~/.pgpass`, or wherever `PGPASSFILE` points. This package adds the
  non-Windows default; the original code required `%APPDATA%`.
- They log in with keyboard-interactive authentication. The code answers the
  Duo prompt with option `1` (push), so **approve a Duo push on your phone for
  every connection**. The full-workspace mirror opens nine connections.
- They pin the server's host key on first use in `emp/.wrds/known_hosts`. After
  that, any mismatch aborts.

The first thing to run is the read-only login probe. It also pins the host key:

```bash
python -m src.probe_wrds_cloud
```

The probe runs `psql -d wrds -c 'select current_user, current_database()'`
on WRDS Cloud. If that line of its output shows your username, the grid jobs
will be able to query WRDS. Otherwise set up database access on WRDS Cloud
first, as WRDS documents for its cloud users.

Most `sync_*` scripts refuse to run until the host key has been pinned this way.
One exception to know about: `sync_wrds_cloud_observable_floors.py` compares
against a fingerprint hard-coded from July 2026. If WRDS rotates its key, edit
`EXPECTED_HOST_KEY` there.

The grid route deploys a **git commit** of `emp/`
(`deploy_wrds_cloud_project.py` runs `git archive`). This package has no git
history, so make `emp/` a repository first:

```bash
cd emp && git init && git add -A && git status   # review what is staged, then:
git commit -m "hl_hh package"
```

`emp/.gitignore` keeps licensed rows, `output/` and `legacy/` out of that
repository. It deliberately tracks the small `manifest.json`,
`collection_manifest.json`, `dataset_spec.json` and `run_spec.json` files
under `data/`, which record your local paths. That is fine for a private
repository. Look at `git status` before pushing it anywhere. The deploy step
only ships `src/`, `tests/`, `config/`, `wrds_client.py`, `requirements.txt`
and `README.md`.

The WRDS Cloud system `python3` must have numpy, pandas, scipy, pyarrow,
matplotlib, wrds **and PyYAML**. The deploy step checks all of them except
PyYAML and fails loudly if one is missing. `python -m
src.probe_wrds_cloud_packages` lists what is there. Install anything missing
with `pip install --user` on WRDS Cloud.

## 4. Running the tests

From the package root `hl_hh/`, with the venv active:

```bash
cd emp
python -m pytest -q                      # 242 passed, 29 skipped (~20 s, no WRDS, no data)
python -m src.run_theory_simulations --simulation-id hdlss_jse_v2   # optional, ~1 min, no WRDS
python -m pytest -q                      # now 266 passed, 5 skipped
cd ..
python -m pytest -q research_tools       # 6 passed
cd companion && python code/sim_selftest.py && cd ..   # < 1 min; prints [ok] lines
```

`emp/tests/conftest.py` skips the tests that verify artifacts which are not
shipped, and prints the reason; pass `-rs` to see it. The skips are:

- **24 tests that need `emp/output/simulations/hdlss_jse_v2`.** This is a pure
  simulation (the second command above). After it has run, all 24 pass.
- **5 tests that need the July variance-reduction analysis outputs.** These
  exist only after the full WRDS pipeline (docs/05).
- **22 more if Poppler's `pdfinfo` is not on PATH.** Most of
  `tests/test_finalize_report_pdf.py` calls it. You then get 220 passed,
  51 skipped. Install Poppler (`brew install poppler`,
  `apt install poppler-utils`, or a Windows build) to run them.

## 5. Disk, memory and time

| step | disk | notes |
|---|---|---|
| raw CIZ snapshots 2005–2025 (csv.gz) | larger than the Parquet copy (not measured here) | `emp/data/raw/` |
| Parquet collection | ≈ 380 MB (21 files, 0.91–1.22 M rows/year) | `emp/data/interim/` |
| one rolling run (top-cap or nested) | ≈ 65–70 MB (20 year directories) | `emp/data/processed/rolling/<id>/` |
| companion `out/` | intermediate Parquet/NPZ plus the JSON results | `companion/out/` |

Memory: `companion/code/rawpanel.py` pivots all daily returns into one
dates × PERMNO matrix per process, and peaks at about 3 GB RAM. Everything in
this package ran on a 2-core Linux machine with 7 GB.

Time: one rolling run takes about 40 minutes on that machine (1–2 min per
year). The WRDS download (A2 in docs/03) depends on your connection to WRDS
and was not timed for this package. The companion takes several hours in
total. docs/10 §7 lists measured times per step.
