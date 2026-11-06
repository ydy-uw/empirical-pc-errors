# 04 — Data dictionary

This document gives column-level descriptions of every file the pipeline
writes, in the order they are created. Types are Parquet/Arrow types. "Cell"
means one (formation_date, n_assets) pair.

> **Licensed content.** Files marked 🔒 contain security-level CRSP rows
> (PERMNO, prices, returns). Treat them as WRDS data under your institution's
> licence and never commit or share them. All other files are aggregates.

## 1. Raw snapshots — `emp/data/raw/wrds/`

### 1.1 Reference snapshot `crsp_ciz_reference/<reference-id>/` (`src.download_crsp_reference`)

| file | WRDS source | content |
|---|---|---|
| `calendar/calendar.csv.gz` | `crsp_a_stock.metaexchangecalendar` | caldt, tradingflg, holidayflg, weekendflg, datestatusflg, weekday, exchangegroup. **The trading calendar**: formation dates, windows and OOS months are built from it. |
| `delists/stkdelists.csv.gz` 🔒 | `crsp_a_stock.stkdelists` | delisting records (permno, delistingdt, …); used by the OOS-missing audit |
| `flags/metaflaginfo.csv.gz` | `crsp_a_stock.metaflaginfo` | dictionary of CIZ flag codes |
| `market/daily_index.csv.gz` | `crsp_a_stock.wrds_dailyindexret_query` | value-weighted market index returns |
| `manifest.json` | — | SQL text, parameters, row counts, byte sizes and SHA-256 of every file |

### 1.2 Annual daily snapshots 🔒 `crsp_ciz/annual/<prefix>_<year>/`

`daily/year=<year>/dsf_v2.csv.gz` holds every `crsp_a_stock.dsf_v2` row in the
calendar year for PERMNOs that were conventional U.S. common stock (NS / EQTY /
COM / usincflg Y / ACOR or CORP / primary exchange N, A or Q) at some point in
that year or the year before. There are 21 columns (the list is in
`config/research.yml → wrds.columns`). `manifest.json` records the SQL, row
counts and SHA-256. `collections/<collection-id>/collection_manifest.json` lists
the 21 years with their manifest hashes.

## 2. Interim Parquet 🔒 — `emp/data/interim/crsp_ciz_parquet/<dataset-id>/`

`year=<y>/daily.parquet` (`src.prepare_crsp_parquet`) contains one row per
(permno, trading day). String flags are stored as dictionary-encoded strings.

| column | type | meaning |
|---|---|---|
| permno, permco | int64 | security and company identifiers |
| sharetype, securitytype, securitysubtype, usincflg, issuertype, primaryexch, conditionaltype, tradingstatusflg | string | point-in-time security attributes used by the eligibility rules |
| dlycaldt | timestamp | trading date |
| dlydelflg | string | delisting-day flag |
| dlyprc, dlyprcflg | double, string | price and its flag |
| dlycap, dlycapflg | double, string | market capitalization and its flag (ranking uses the sum over a PERMCO's share classes) |
| **dlyret** | double | CIZ total return. It already contains the terminal (delisting) return, so it is never compounded with a separate delisting return. **This is the return the whole project uses.** |
| dlyretmissflg, dlyretdurflg, dlydistretflg | string | missing-return reason, return duration (D1–D4, DU accepted), distribution flag |
| dlyvol | double | volume |

The directory also contains `year=<y>/manifest.json`, `dataset_spec.json` (the
normalization spec) and `collection_manifest.json`.

## 3. Rolling run — `emp/data/processed/rolling/<rolling-id>/`

Run-level files:

- `run_spec.json` records the code hashes, input manifest hashes, the protocol
  (window, Jung settings, q grid, selection design and seed), the requested
  formation dates, and the Python and package versions.
- `collection_manifest.json` lists the per-year manifest hashes and the run
  signature. It is written last and marks the run as complete.

Each `year=<y>/` directory contains the following files.

### 3.1 `observable_error_floors.parquet` — the floors A_j (816 cells)

There is one row per (cell, q_factors, q_source, factor_position). The file
exists only for cells that passed the next-month gate (docs/03 §4).

| column | meaning |
|---|---|
| formation_date, n_assets | the cell |
| q_factors | k, the factor count used to split spike and bulk |
| q_source | `fixed` (the grid 1–15, 20, 30) or `jung` (the Jung α = 0.10 count for that cell) |
| factor_position | j = 1 … k |
| spectral_floor_status | `ok`, or `q_exceeds_rank` |
| **observable_floor_centered** | A_j = ℓ/θ_j on the window-demeaned panel; ℓ = mean of the non-factor eigenvalues |
| **observable_floor_uncentered** | the same on the raw panel (the theorem's normalization) |
| mp_adjusted_floor_centered | the same with the Marchenko–Pastur bulk adjustment (Gurdogan & Shkolnik 2024, eq. 25) |
| centered_sample_eigenvalue / uncentered_sample_eigenvalue | θ_j (= s_j²/N) |
| centered_bulk_eigenvalue / uncentered_bulk_eigenvalue | ℓ |
| rotation_component_observable | always False: the in-subspace rotation is not identified from one panel |
| correction_status, theory_applicable, status | the H♯ correction's status for that (cell, q). `localized_factor` means a factor loads on too few names (participation < 1 % or a loading > 0.5), so the correction's theory does not apply |

### 3.2 `factor_residual_lengths.parquet` — Jung residual paths (all 956 cells)

There is one row per (cell, tested_k, observation). R_r(k) = ‖x_r − x̂_r(k)‖²/P
is the squared residual of day r after removing k principal components of the
centered panel, divided by P.

| column | meaning |
|---|---|
| formation_date, n_assets | the cell |
| selection_scope | `formation_63_day_only` (v2 protocol) |
| formation_observations | 63 |
| initial_max_factors, retry_max_factors, retried_at_higher_m, max_tested_factor | Jung search limits (30, retry at 50) |
| tested_k | k = 0, 1, …, max_tested_factor |
| observation_index | r = 0 … 62 |
| residual_length | R_r(k) |

This is what makes the centered spectrum recoverable for every cell without
the returns: T_k = P·Σ_r R_r(k), θ_j = (T_{j−1} − T_j)/N and
ℓ(k) = T_k / (N(N − 1 − k)) (`companion/code/panel.py`). The companion checks
the reconstruction against the stored eigenvalues and finds a maximum
relative error around 1e-14 (`check_reconstruction.py`).

### 3.3 `factor_selections.parquet` — the Jung count (all 956 cells × 3 α)

This file has the same header columns as 3.2, plus `alpha` (0.05, 0.10 or
0.20), `selected_factors` (the first k whose one-sided skewness test does not
reject) and `censored` (True if no k up to the limit was accepted). The paper
uses α = 0.10.

### 3.4 `factor_diagnostics.parquet` — the full Jung test path

This has the same header columns, plus one row per tested k with
`skewness_z`, `pvalue` (D'Agostino one-sided), `residual_mean` and
`sample_eigenvalue_k`.

### 3.5 `selected_universes.parquet` 🔒 — who is in each universe (all 956 cells)

There is one row per (cell, rank). `rank` is the capitalization rank within
the cell. The columns are the identifiers and all interim columns on the
formation date, plus `company_cap` (the summed PERMCO capitalization used for
ranking). Runs made with the current code also carry `selection_design`
(`top_cap` or `top3000_nested_random`) and `selection_seed` columns. The
original top-cap run predates them. `companion/code/rawpanel.py` rebuilds every panel from this file and
the interim returns.

### 3.6 `variance_comparisons.parquet` — the July variance study (816 cells)

There is one row per (cell, q_factors, q_source, correction), where
`correction` ∈ {`multi_factor_sharp` (H♯, fixed and Jung q),
`one_factor_jse_centered_mp`, `one_factor_jse_goldberg2025`}. The columns hold
realized next-month variances of the uncorrected and corrected minimum-variance
portfolios (`variance_before/after/difference/ratio`) and the spectral fit
(`kappa2`, `gamma2`, `phi_norm`). They also include localization diagnostics,
portfolio weight QA (gross leverage, max weight, budget residual) and the JSE
intensity (`jse_intensity`, `jse_raw_intensity`, `positive_part`).

### 3.7 `error_decomposition_diagnostics.parquet` — cross-fit error proxy (816 cells)

The panel is split into odd and even days. Each half's ordered PCA axes serve
as the benchmark for the other half. The file records the total, out-of-subspace
and rotation errors before and after correction, per factor position and fold.
This is a basis-dependent proxy (`population_rotation_identified = False`).

### 3.8 `extreme_events.parquet` 🔒

These are single-name daily returns with |r| > 1.0 (more than 100 %) inside a
formation window. Each has the full interim row for that security and day.

### 3.9 `qa_records.json`, `skipped_cells.json`, `manifest.json`

- `qa_records.json` has one record per cell: eligibility counts at each filter
  step, estimation-panel QA and OOS QA.
- `skipped_cells.json` has one aggregate record per skipped cell: formation
  date, n_assets, reason (e.g. `incomplete_terminal_return`) and error code.
  It never contains identifiers or exception messages.
- `manifest.json` holds the per-file SHA-256, the run signature and the
  formation-date hash for the year.

## 4. Other processed data

| directory | written by | content |
|---|---|---|
| `data/processed/variance_reduction/<extension-id>/year=<y>/` | `src.run_variance_reduction_extension` | July variance-reduction extension, per-cell statistics (docs/05) |
| `data/processed/observable_floors_theorem2_200601_202511_v1_20260809/` and `factor_path_theorem2_…_20260809/` | `src.sync_wrds_cloud_observable_floors` | byte-identical copies of 3.1 and of 3.2 + 3.3 from the top-cap run, downloaded from WRDS Cloud in August. The August scripts read them; the companion reads the rolling run directly |
| `…_nested_random_top3000_…_20260811/` | same, nested run | same for the nested-random run |

## 5. Analysis outputs (not data, but also not shipped)

| directory | written by |
|---|---|
| `emp/output/analysis/<analysis-id>/{tables,figures}/` | `src.analyze_rolling_results`, `src.analyze_theorem2_visible_error`, … (docs/05) |
| `emp/output/simulations/<simulation-id>/` | `src.run_theory_simulations`, `src.run_localized_factor_simulation` |
| `emp/output/robustness/<id>/`, `emp/output/audits/` | robustness and reproduction audits (docs/05) |
| `research_output/…` (package root) | `research_tools/` (docs/05) |
| `companion/out/`, `companion/build/{generated,figures}/` | `companion/code/` (docs/06) |
| `wpca/out/` | `wpca/code/` (docs/07) |
