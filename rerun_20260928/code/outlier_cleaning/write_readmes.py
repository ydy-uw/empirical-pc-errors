# COPY for the 28 September 2026 rerun (rerun_20260928/README.md): output paths redirected to $HLHH_RERUN_OUT.
"""README.md for each experiment folder (run after the experiments).

    python outlier_cleaning/code/write_readmes.py
"""
from __future__ import annotations
import json
from oc_common import EXP
from treatments import METHODS

INFO = {
    "00_raw": ("Untreated returns (reference)", "00_untreated_returns",
               "No treatment. The CRSP daily simple returns enter the panels exactly as delivered. "
               "Every other experiment is compared with this one, and its outputs reproduce the "
               "stored outputs in figure_code/out (results/reproduction_check.json)."),
    "01_fixed_clip": ("Fixed clip", "01_fixed_clip",
                      "Every daily simple return is clipped to [-c, +c]. Because simple returns cannot fall "
                      "below -100%, only the upper bound binds for c >= 1."),
    "02_daily_winsorization": ("Daily cross-sectional winsorization", "02_daily_winsorization",
                               "On each day, returns are winsorized at that day's q and 1-q quantiles across the "
                               "names of the panel."),
    "03_cross_sectional_mad": ("Cross-sectional robust-scale clip", "03_cross_sectional_mad",
                               "On each day, returns are clipped at the day's median plus or minus c robust scale units, "
                               "one unit being 1.4826 times the day's median absolute deviation across names."),
    "04_time_series_mad": ("Time-series robust-scale clip", "04_time_series_mad",
                           "Each name's returns are clipped at its own median plus or minus c times its robust "
                           "scale (1.4826 times the MAD). The scale comes from the 63-day window itself or, in the "
                           "setting prior_c10, from the 63 trading days before the window."),
    "05_log_returns": ("Log returns", "05_log_returns",
                       "Simple returns r are replaced by log(1 + r)."),
    "06_rank_transform": ("Rank transform", "06_rank_transform",
                          "Each name's 63 returns are replaced by normal scores of their ranks. In the baseline setting the "
                          "scores are multiplied by the name's robust scale; in 'unit' they are not (Gaussian rank "
                          "correlation); 'standard' is the control without ranks (each name divided by its standard deviation)."),
    "07_drop_names": ("Dropping names with extreme returns", "07_drop_names",
                      "Every name with a daily return above +c (or below -c) inside the window is removed "
                      "from that panel."),
    "08_reversal_filter": ("Spike-and-reversal filter", "08_reversal_filter",
                           "For consecutive days (t-1, t) of a name, both returns are flagged when either exceeds "
                           "+up and the two-day compound return is below rev; flagged returns are replaced by the "
                           "day's cross-sectional median. Only days up to the formation date are used. The setting "
                           "then_clip applies a fixed clip at +/-100% afterwards."),
}


def main():
    for m, (fname, settings) in METHODS.items():
        d = EXP / m
        if not d.is_dir():
            continue
        title, report, text = INFO[m]
        cfg = json.load(open(d / "config.json")) if (d / "config.json").is_file() else {}
        rows = "\n".join(f"| `{s}`{' (baseline)' if s == 'main' else ''} | "
                         f"{', '.join(f'{k} = {v}' for k, v in p.items()) or 'none'} |" for s, p in settings.items())
        files = sorted(p.name for p in (d / "results").iterdir())
        hs = cfg.get("heavy_settings_run", [])
        extra = (" and setting " + ", ".join(f"`{h}`" for h in hs) + " (files with that suffix)") if hs else " only"
        md = f"""# Experiment {m}: {title}

**Treatment.** {text} Function `{fname}` in `outlier_cleaning/code/treatments.py`.

| setting | parameters |
|---|---|
{rows}

**What was run.** Every setting (the code calls the baseline setting `main`): the per-panel diagnostics on all 956 formation panels
(factor counts under four rules, the Theorem 2 estimates at the treated and the untreated
count, the pervasiveness screen, residual dispersion, kurtosis and bulk dispersion).
Baseline setting{extra}: the residual size blocks (P = 500 and 3,000), the factor-analytic
estimate (all panels) and the disjoint-window check (P = 500 and 3,000).

**Report.** `outlier_cleaning/overleaf_reports/{report}.tex`; the compiled PDF is in
`outlier_cleaning/reports/`.

**Files.**

- `config.json`: treatment, settings, code hashes, software versions, run times, and the
  result of the input check (inputs unchanged by the run: {cfg.get('inputs_unchanged', 'see logs')}).
- `results/panels_<setting>.csv`: one row per panel; what the treatment changed and every
  per-panel diagnostic.
- `results/spectra_<setting>.npz`: the centered dual spectrum of every treated panel.
- `results/summary_<setting>.json`: the aggregates quoted in the report.
- `results/blocks.csv`, `fa.csv`, `oos.csv`: the slower diagnostics, baseline setting.
- `logs/run.log`, `logs/input_check_before.json`, `logs/input_check_after.json`.

Files present: {', '.join(f'`{f}`' for f in files)}.

**Data.** The treatment is applied in memory to panels read from the CRSP extract under
`emp/data`, which is opened read-only and never written. The input check compares the
SHA-256 hash of every input file before and after the run. All files here are aggregates
with no security identifiers.

**Rerun.** From the project root, with `OPENBLAS_NUM_THREADS=1`:
`python outlier_cleaning/code/run_method.py {m}`{'' if m == '00_raw' else ' (after `00_raw`, whose counts are the reference)'}, then
`spectra.py {m}`, `summarize.py`, `make_tables.py`, `figures.py` in the same folder.
"""
        open(d / "README.md", "w").write(md)
        print("README", m)


if __name__ == "__main__":
    main()
