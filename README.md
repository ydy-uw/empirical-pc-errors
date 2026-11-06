# Code for "Empirical PC Errors" (version of 17 October 2026)

This repository holds the code behind the 17 October 2026 version of the paper, from the WRDS data
download to the final tables and figures. It applies Theorem 2 of Bernstein et al., "Principal
component error in high-dimensional factor models" (arXiv:2609.20550), to CRSP daily stock returns.

**No data is included.** CRSP, Compustat and SDC data are licensed. To run the code you need your own
WRDS account with access to CRSP (CIZ), the CRSP-Compustat link and SDC New Issues. `.gitignore`
blocks the data folders and data file types. Do not commit them.

## Layout

The scripts find each other through paths relative to this folder, so keep the layout as it is.

| folder | content |
|---|---|
| `emp/` | WRDS login, CRSP download, Parquet conversion and the rolling run that builds the universes |
| `rerun_20260928/code/` | `r01`-`r04` and `spac_flags.py`: sample cleaning. `r10`, `r12`-`r20`: checks and numbers |
| `rerun_20260928/code/outlier_cleaning/`, `.../figure_code/` | the versions that the final results come from; they write to `rerun_20260928/work/` |
| `outlier_cleaning/code/`, `figure_code/outline_*/` | the originals of those versions; `outlier_cleaning/code/` also gives the uncleaned reference run |
| `figure_code/`, `figure_code/paper/`, `companion/code/` | shared functions: theory formulas, factor counts, panel loading |
| `figure_code/out/finite_p_error.csv` | aggregate simulation output; `r13` reads it for a comparison |
| `docs/` | setup, data pipeline and data dictionary. They also mention folders that are not in this repository |

## 1. Environment

The work ran on Windows 11 with Python 3.13. From `emp/`:

```bash
python3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`rerun_20260928/run_all.sh` calls `emp/.venv/Scripts/python.exe`, so run it from Git Bash on Windows.
On Linux or macOS, change the `PY=` line to `emp/.venv/bin/python`, and use `emp/.venv/bin/python` in
the commands below.

## 2. WRDS login

No username or password is stored in this repository. From `emp/`, run this once:

```bash
.venv/Scripts/python.exe configure_wrds.py
```

It asks for your username and password, and then stores them outside version control:

| what | where | content |
|---|---|---|
| username | `emp/.wrds/username` (see `emp/.wrds/username.example`) | `YOUR_WRDS_USERNAME` |
| password | `%APPDATA%\postgresql\pgpass.conf` on Windows, `~/.pgpass` on Linux and macOS | `wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD` |

WRDS can ask for a Duo approval at each login. Details: `docs/02_setup.md`.

## 3. Reproduce the results

Run step 3.1 from `emp/` and all other steps from this folder. The scripts use fixed data IDs, so keep the IDs below.
Use `emp/.venv/Scripts/python.exe` as `python`.

**3.1 Download CRSP and build the top-cap universes** (`docs/03_data_pipeline.md`, route A)

```bash
python -m src.download_crsp_reference --start 1972-01-01 --end 2025-12-31 --snapshot-id reference_1972_2025_remote_v1_20260723
python -m src.download_crsp_collection --start-year 2005 --end-year 2025 --snapshot-prefix crsp_ciz_remote_v1_20260723 --collection-id crsp_ciz_2005_2025_remote_v1_20260723
python -m src.prepare_crsp_parquet --start-year 2005 --end-year 2025 --snapshot-prefix crsp_ciz_remote_v1_20260723 --dataset-id crsp_ciz_2005_2025_remote_v1_20260723
python -m src.run_rolling_experiments --dataset-root data/interim/crsp_ciz_parquet/crsp_ciz_2005_2025_remote_v1_20260723 --reference-snapshot data/raw/wrds/crsp_ciz_reference/reference_1972_2025_remote_v1_20260723 --run-id rolling_topcap_200601_202511_remote_v2_20260724
```

The original extract used the WRDS Cloud grid (route C in `docs/03_data_pipeline.md`). Those scripts are
also here. Route C expects `emp/` to be its own git repository, and it was not tested from this layout.

**3.2 Download the name histories and clean the sample** (SPACs and low-volatility stocks)

```bash
python rerun_20260928/code/r01_fetch_names.py
python rerun_20260928/code/r01b_fetch_compustat.py
python rerun_20260928/code/r01d_fetch_sdc_ipos.py
python rerun_20260928/code/r02_candidates.py 2006 2025
python rerun_20260928/code/r03_build_universes.py
python rerun_20260928/code/r04_check_universes.py
```

`r01c_fetch_sdc.py` only lists the SDC schema. `spac_flags.py` holds the SPAC rule. `r05_make_code_copies.py`
made the versions in `rerun_20260928/code/` from the originals. You do not need to run it again.

**3.3 Run the uncleaned reference** (only `r10` needs it, to measure the effect of the cleaning)

```bash
python outlier_cleaning/code/panels.py build
python outlier_cleaning/code/run_method.py 00_raw
python outlier_cleaning/code/run_method.py 01_fixed_clip
python outlier_cleaning/code/run_method.py 01_fixed_clip --heavy-setting c050
```

**3.4 Run the main pipeline on the cleaned sample**

```bash
export HLHH_ROLLING_DIR=$PWD/emp/data/processed/rolling/rolling_topcap_clean_200601_202511_20260928
export HLHH_RERUN_OUT=$PWD/rerun_20260928/work
bash rerun_20260928/run_all.sh
```

This step writes the tables and eight of the nine figures to `paper/overleaf_upload_20260928/`. The folder
name is the date of the rerun. These are the tables and figures of the 17 October version.

**3.5 Run the checks and the last figure** (keep the two variables of step 3.4 set)

```bash
python rerun_20260928/code/r12_null_checks.py
python rerun_20260928/code/r13_finite_p_recalibrated.py
python rerun_20260928/code/r15_leverage_check.py
python rerun_20260928/code/r17_comove_pca.py
python rerun_20260928/code/r18_dayvar_leverage_alt.py
python rerun_20260928/code/r19_comove_fa_shuffled.py
python rerun_20260928/code/r20_extra_shuffled.py
python rerun_20260928/code/r16_fig_leverage.py paper/overleaf_upload_20260928
python rerun_20260928/code/r10_appendix_numbers.py
```

Keep this order: `r18` and `r19` read the output of `r15`, and `r16` reads the output of `r18`. `r15`,
`r18`, `r19` and `r20` start 16 worker processes. The paper quotes the results of `r12`, `r13`, `r15`
and `r17`-`r20` by hand. Its Appendix B lists every script and its outputs. `r06_data_report.py` makes the
separate report on the cleaned data. It uses Microsoft Edge on Windows to make the PDF.

All outputs go to `emp/data/`, `rerun_20260928/work/` or `paper/`. Some of them hold licensed returns, so
do not share or commit them.
