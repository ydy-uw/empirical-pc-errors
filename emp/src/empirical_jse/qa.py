"""Data-reliability checks for CRSP CIZ extracts and rolling panels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .crsp import ReturnPanel, verify_snapshot


@dataclass(frozen=True)
class ExtractQA:
    rows: int
    permnos: int
    trading_dates: int
    duplicate_permno_dates: int
    finite_return_rows: int
    finite_return_with_non_na_flag: int
    missing_return_with_na_flag: int
    returns_below_minus_one: int
    absolute_returns_above_ten: int
    terminal_rows: int
    terminal_rows_missing_return: int
    nonpositive_valid_cap_rows: int


@dataclass(frozen=True)
class PanelQA:
    n_observations: int
    p_assets: int
    finite: bool
    rank: int
    zero_variance_assets: int
    overall_zero_fraction: float
    max_zero_run_p50: float
    max_zero_run_p90: float
    max_zero_run_p99: float
    annualized_vol_p01: float
    annualized_vol_p50: float
    annualized_vol_p99: float
    max_absolute_return: float
    smallest_nonzero_singular_value: float


@dataclass(frozen=True)
class OOSQA:
    n_observations: int
    p_assets: int
    missing_cells_before_mark_to_market: int
    assets_with_any_missing: int
    assets_with_trailing_missing: int
    incomplete_flagged_cells: int
    mv_flagged_cells: int
    terminal_assets: int
    terminal_return_reused: int


def load_snapshot_daily(snapshot: str | Path) -> pd.DataFrame:
    """Verify and load every annual ``csv.gz`` artifact in a raw snapshot."""
    root = Path(snapshot)
    verification = verify_snapshot(root)
    if not verification.ok:
        raise ValueError(
            "snapshot verification failed: " + "; ".join(verification.errors)
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    paths = [
        root / item["path"]
        for item in manifest["files"]
        if item.get("format") == "csv.gz"
        and str(item.get("path", "")).startswith("daily/")
    ]
    if not paths:
        raise ValueError("snapshot has no daily csv.gz artifacts")
    frames = [
        pd.read_csv(
            path,
            keep_default_na=False,
            low_memory=False,
            parse_dates=["dlycaldt"],
        )
        for path in sorted(paths)
    ]
    daily = pd.concat(frames, ignore_index=True)
    daily["dlycaldt"] = pd.to_datetime(daily["dlycaldt"]).dt.normalize()
    daily["permno"] = pd.to_numeric(daily["permno"], errors="raise").astype(int)
    return daily


def audit_extract(daily: pd.DataFrame) -> ExtractQA:
    """Run invariant and flag-consistency checks on a CIZ daily extract."""
    required = {
        "permno",
        "dlycaldt",
        "dlyret",
        "dlyretmissflg",
        "dlydelflg",
        "dlycap",
        "dlycapflg",
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily extract lacks columns: {sorted(missing)}")
    returns = pd.to_numeric(daily["dlyret"], errors="coerce")
    finite = returns.notna() & np.isfinite(returns)
    available_flag = daily["dlyretmissflg"].astype("string").eq("NA")
    cap = pd.to_numeric(daily["dlycap"], errors="coerce")
    valid_cap = daily["dlycapflg"].astype("string").eq("BP")
    terminal = daily["dlydelflg"].astype("string").eq("Y")
    return ExtractQA(
        rows=int(len(daily)),
        permnos=int(daily["permno"].nunique()),
        trading_dates=int(daily["dlycaldt"].nunique()),
        duplicate_permno_dates=int(
            daily.duplicated(["permno", "dlycaldt"]).sum()
        ),
        finite_return_rows=int(finite.sum()),
        finite_return_with_non_na_flag=int((finite & ~available_flag).sum()),
        missing_return_with_na_flag=int((~finite & available_flag).sum()),
        returns_below_minus_one=int((returns < -1.0).sum()),
        absolute_returns_above_ten=int((returns.abs() > 10.0).sum()),
        terminal_rows=int(terminal.sum()),
        terminal_rows_missing_return=int((terminal & ~finite).sum()),
        nonpositive_valid_cap_rows=int((valid_cap & ~(cap > 0)).sum()),
    )


def longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of true values."""
    vector = np.asarray(values, dtype=bool).reshape(-1)
    if not vector.any():
        return 0
    padded = np.concatenate(([False], vector, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return int(np.max(stops - starts))


def audit_estimation_panel(
    returns: pd.DataFrame | np.ndarray,
    *,
    annualization: float = 252.0,
) -> PanelQA:
    """Audit a synchronized estimation matrix without forming ``p x p``."""
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a two-dimensional matrix")
    n_observations, p_assets = matrix.shape
    finite = bool(np.isfinite(matrix).all())
    if not finite:
        raise ValueError("estimation matrix contains missing or infinite values")
    variances = np.var(matrix, axis=0, ddof=1)
    volatilities = np.sqrt(np.maximum(variances, 0.0) * annualization)
    zero_runs = np.array(
        [longest_true_run(matrix[:, j] == 0.0) for j in range(p_assets)],
        dtype=np.float64,
    )
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(centered.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    nonzero = singular_values[singular_values > tolerance]
    return PanelQA(
        n_observations=n_observations,
        p_assets=p_assets,
        finite=finite,
        rank=int(nonzero.size),
        zero_variance_assets=int(np.count_nonzero(variances <= 0)),
        overall_zero_fraction=float(np.mean(matrix == 0.0)),
        max_zero_run_p50=float(np.quantile(zero_runs, 0.50)),
        max_zero_run_p90=float(np.quantile(zero_runs, 0.90)),
        max_zero_run_p99=float(np.quantile(zero_runs, 0.99)),
        annualized_vol_p01=float(np.quantile(volatilities, 0.01)),
        annualized_vol_p50=float(np.quantile(volatilities, 0.50)),
        annualized_vol_p99=float(np.quantile(volatilities, 0.99)),
        max_absolute_return=float(np.max(np.abs(matrix))),
        smallest_nonzero_singular_value=(
            float(nonzero[-1]) if nonzero.size else 0.0
        ),
    )


def prepare_oos_mark_to_market(panel: ReturnPanel) -> tuple[pd.DataFrame, OOSQA]:
    """Fill nonterminal OOS gaps with zero and report their exact incidence.

    CIZ terminal rows have already been validated and incorporated once by the
    panel builder.  Zero filling here means carrying the last observed price
    through a halt/gap and holding cash strictly after a terminal row.
    """
    returns = panel.out_of_sample.copy()
    missing = returns.isna()
    if panel.out_of_sample_return_flags is None:
        return_flags = pd.DataFrame(
            "NA",
            index=returns.index,
            columns=returns.columns,
        )
    else:
        return_flags = panel.out_of_sample_return_flags.reindex_like(returns)
    string_flags = return_flags.astype("string")
    incomplete_flags = ~string_flags.isin(["NA", "CASH"])
    incomplete_flags = incomplete_flags.fillna(True)
    trailing_assets = 0
    for column in returns.columns:
        observed = np.flatnonzero(returns[column].notna().to_numpy())
        if observed.size and observed[-1] < len(returns) - 1:
            trailing_assets += 1
        elif not observed.size:
            trailing_assets += 1

    filled = returns.fillna(0.0)
    terminal_reused = 0
    for permno, terminal_date in panel.terminal_dates.items():
        after = filled.index > terminal_date
        terminal_reused += int(np.count_nonzero(filled.loc[after, permno] != 0.0))
    qa = OOSQA(
        n_observations=int(filled.shape[0]),
        p_assets=int(filled.shape[1]),
        missing_cells_before_mark_to_market=int(missing.sum().sum()),
        assets_with_any_missing=int(missing.any(axis=0).sum()),
        assets_with_trailing_missing=trailing_assets,
        incomplete_flagged_cells=int(incomplete_flags.sum().sum()),
        mv_flagged_cells=int(string_flags.eq("MV").sum().sum()),
        terminal_assets=len(panel.terminal_dates),
        terminal_return_reused=terminal_reused,
    )
    return filled, qa


def qa_to_dict(value: Any) -> dict[str, Any]:
    """Convert one of this module's frozen QA records to a JSON-safe dict."""
    return asdict(value)
