"""Matched before/after variance experiments for one rolling panel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .corrections import (
    WeakFactorError,
    basis_error_decomposition,
    decompose_returns,
    factor_localization,
    fit_spectral_decomposition,
    one_factor_jse,
    optimization_bias_correction,
)
from .crsp import ReturnPanel
from .jung import JungFactorCount, jung_factor_count, select_from_pvalues
from .portfolio import minimum_variance_weights, realized_variance
from .qa import OOSQA, prepare_oos_mark_to_market


@dataclass(frozen=True)
class PanelExperiment:
    """Variance comparisons and factor-selection diagnostics for one panel."""

    comparisons: pd.DataFrame
    factor_diagnostics: pd.DataFrame
    factor_selections: pd.DataFrame
    factor_residuals: pd.DataFrame
    observable_error_floors: pd.DataFrame
    error_decomposition_diagnostics: pd.DataFrame
    jung: JungFactorCount
    oos_qa: OOSQA


@dataclass(frozen=True)
class FormationJungDiagnostics:
    """Jung selection computed only from one 63-day formation matrix."""

    formation_date: pd.Timestamp
    n_assets: int
    matrix_sha256: str
    primary_alpha: float
    initial_max_factors: int
    retry_max_factors: int
    sensitivity_alphas: tuple[float, ...]
    retried_at_higher_m: bool
    jung: JungFactorCount
    factor_diagnostics: pd.DataFrame
    factor_selections: pd.DataFrame
    factor_residuals: pd.DataFrame


@dataclass(frozen=True)
class VariancePair:
    """Matched realized variances and diagnostics for the exact two weights."""

    variance_before: float
    variance_after: float
    weight_diagnostics: dict[str, float]


def _matrix_sha256(matrix: np.ndarray) -> str:
    values = np.ascontiguousarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def compute_formation_jung_diagnostics(
    centered_estimation: pd.DataFrame | np.ndarray,
    *,
    formation_date: pd.Timestamp,
    primary_alpha: float = 0.10,
    initial_max_factors: int = 30,
    retry_max_factors: int = 50,
    sensitivity_alphas: Iterable[float] = (0.05, 0.10, 0.20),
) -> FormationJungDiagnostics:
    """Compute and retain Jung diagnostics before any OOS validation."""

    x = np.asarray(centered_estimation, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("centered_estimation must be a finite matrix")
    n_observations, n_assets = x.shape
    if n_observations != 63:
        raise ValueError("formation Jung selection requires exactly 63 rows")
    alphas = tuple(sorted({float(value) for value in sensitivity_alphas}))
    if not alphas or any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("sensitivity_alphas must lie strictly between zero and one")
    if float(primary_alpha) not in alphas:
        raise ValueError("primary_alpha must be included in sensitivity_alphas")

    jung = jung_factor_count(
        x,
        alpha=primary_alpha,
        max_factors=initial_max_factors,
        center=False,
    )
    selected_by_alpha = {
        alpha: select_from_pvalues(jung.pvalues, alpha) for alpha in alphas
    }
    retried = jung.censored or any(
        selected is None for selected in selected_by_alpha.values()
    )
    if retried and retry_max_factors > initial_max_factors:
        jung = jung_factor_count(
            x,
            alpha=primary_alpha,
            max_factors=retry_max_factors,
            center=False,
        )
        selected_by_alpha = {
            alpha: select_from_pvalues(jung.pvalues, alpha)
            for alpha in alphas
        }
    else:
        retried = False

    common = {
        "formation_date": pd.Timestamp(formation_date).normalize(),
        "n_assets": n_assets,
        "selection_scope": "formation_63_day_only",
        "formation_observations": n_observations,
        "initial_max_factors": int(initial_max_factors),
        "retry_max_factors": int(retry_max_factors),
        "retried_at_higher_m": bool(retried),
        "max_tested_factor": int(jung.tested_k[-1]),
    }
    factor_diagnostics = pd.DataFrame(
        {
            **common,
            "alpha": jung.alpha,
            "selected_factors": jung.selected,
            "censored": jung.censored,
            "tested_k": jung.tested_k,
            "skewness_z": jung.statistics,
            "pvalue": jung.pvalues,
            "residual_mean": jung.residual_means,
            "sample_eigenvalue_k": jung.sample_eigenvalues[jung.tested_k],
        }
    )
    factor_residuals = pd.DataFrame(
        {
            **common,
            "tested_k": np.repeat(
                jung.tested_k,
                jung.residual_lengths.shape[1],
            ),
            "observation_index": np.tile(
                np.arange(jung.residual_lengths.shape[1]),
                jung.residual_lengths.shape[0],
            ),
            "residual_length": jung.residual_lengths.reshape(-1),
        }
    )
    factor_selections = pd.DataFrame(
        [
            {
                **common,
                "alpha": alpha,
                "selected_factors": selected_by_alpha[alpha],
                "censored": selected_by_alpha[alpha] is None,
            }
            for alpha in alphas
        ]
    )
    return FormationJungDiagnostics(
        formation_date=pd.Timestamp(formation_date).normalize(),
        n_assets=n_assets,
        matrix_sha256=_matrix_sha256(x),
        primary_alpha=float(primary_alpha),
        initial_max_factors=int(initial_max_factors),
        retry_max_factors=int(retry_max_factors),
        sensitivity_alphas=alphas,
        retried_at_higher_m=bool(retried),
        jung=jung,
        factor_diagnostics=factor_diagnostics,
        factor_selections=factor_selections,
        factor_residuals=factor_residuals,
    )


def _variance_pair(
    oos: np.ndarray,
    *,
    before_loadings: np.ndarray,
    after_loadings: np.ndarray,
    factor_eigenvalues: np.ndarray,
    gamma2: float,
) -> VariancePair:
    before_weights = minimum_variance_weights(
        before_loadings,
        factor_eigenvalues,
        gamma2,
    )
    after_weights = minimum_variance_weights(
        after_loadings,
        factor_eigenvalues,
        gamma2,
    )
    before = realized_variance(oos, before_weights, annualization=252.0)
    after = realized_variance(oos, after_weights, annualization=252.0)
    diagnostics = {
        "gross_leverage_before": float(np.abs(before_weights).sum()),
        "gross_leverage_after": float(np.abs(after_weights).sum()),
        "max_absolute_weight_before": float(np.abs(before_weights).max()),
        "max_absolute_weight_after": float(np.abs(after_weights).max()),
        "weight_sum_squares_before": float(before_weights @ before_weights),
        "weight_sum_squares_after": float(after_weights @ after_weights),
        "budget_residual_before": float(before_weights.sum() - 1.0),
        "budget_residual_after": float(after_weights.sum() - 1.0),
        "l1_weight_correction_shift": float(
            np.abs(after_weights - before_weights).sum()
        ),
    }
    if not np.isfinite(np.fromiter(diagnostics.values(), dtype=float)).all():
        raise FloatingPointError("portfolio-weight diagnostics are not finite")
    return VariancePair(
        variance_before=before,
        variance_after=after,
        weight_diagnostics=diagnostics,
    )


def evaluate_panel(
    panel: ReturnPanel,
    *,
    fixed_factor_counts: Iterable[int] = (1, 2, 3, 5, 10, 15, 20, 30),
    jung_alpha: float = 0.10,
    jung_max_factors: int = 30,
    jung_retry_max_factors: int = 50,
    jung_alpha_sensitivity: Iterable[float] = (0.05, 0.10, 0.20),
    mp_adjust: bool = True,
    min_participation_fraction: float = 0.01,
    max_absolute_loading: float = 0.50,
    formation_jung: FormationJungDiagnostics | None = None,
) -> PanelExperiment:
    """Run matched PCA-vs-correction comparisons on one monthly panel.

    Every pair uses exactly the same assets, OOS returns, eigenvalue estimates,
    specific variance, and budget constraint.  The loading correction is the
    only changing ingredient.
    """
    raw_x = panel.raw_estimation.to_numpy(dtype=np.float64)
    x = panel.centered_estimation.to_numpy(dtype=np.float64)
    decomposition = decompose_returns(x, center=False)
    paper_decomposition = decompose_returns(raw_x, center=False)
    sensitivity_alphas = sorted(
        {float(value) for value in jung_alpha_sensitivity}
    )
    if formation_jung is None:
        formation_jung = compute_formation_jung_diagnostics(
            x,
            formation_date=panel.window.formation,
            primary_alpha=jung_alpha,
            initial_max_factors=jung_max_factors,
            retry_max_factors=jung_retry_max_factors,
            sensitivity_alphas=sensitivity_alphas,
        )
    else:
        expected_metadata = (
            pd.Timestamp(panel.window.formation).normalize(),
            x.shape[1],
            _matrix_sha256(x),
            float(jung_alpha),
            int(jung_max_factors),
            int(jung_retry_max_factors),
            tuple(sensitivity_alphas),
        )
        supplied_metadata = (
            formation_jung.formation_date,
            formation_jung.n_assets,
            formation_jung.matrix_sha256,
            formation_jung.primary_alpha,
            formation_jung.initial_max_factors,
            formation_jung.retry_max_factors,
            formation_jung.sensitivity_alphas,
        )
        if supplied_metadata != expected_metadata:
            raise ValueError(
                "formation_jung does not exactly match the panel formation "
                "matrix and requested Jung protocol"
            )
    jung = formation_jung.jung
    oos_frame, oos_qa = prepare_oos_mark_to_market(panel)
    oos = oos_frame.to_numpy(dtype=np.float64)
    p_assets = x.shape[1]
    direction = np.ones(p_assets, dtype=np.float64)

    fixed = sorted({int(q) for q in fixed_factor_counts if int(q) > 0})
    requests: list[tuple[str, int]] = [("fixed", q) for q in fixed]
    if jung.selected is not None and jung.selected > 0:
        requests.append(("jung", int(jung.selected)))

    floor_rows: list[dict[str, object]] = []
    for source, q in requests:
        if q >= min(
            decomposition.n_nonzero,
            paper_decomposition.n_nonzero,
        ):
            floor_rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": q,
                    "q_source": source,
                    "factor_position": np.nan,
                    "status": "q_exceeds_rank",
                }
            )
            continue
        centered_bulk = float(
            decomposition.eigenvalues[q : decomposition.n_nonzero].mean()
        )
        raw_bulk = float(
            paper_decomposition.eigenvalues[
                q : paper_decomposition.n_nonzero
            ].mean()
        )
        centered_aspect = decomposition.n_observations / p_assets
        centered_mp_bulk = float(
            (1.0 + centered_aspect)
            / (
                decomposition.n_nonzero
                - q
                + centered_aspect
            )
            * decomposition.eigenvalues[
                q : decomposition.n_nonzero
            ].sum()
        )
        for factor_position in range(q):
            centered_leading = float(
                decomposition.eigenvalues[factor_position]
            )
            raw_leading = float(
                paper_decomposition.eigenvalues[factor_position]
            )
            floor_rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": q,
                    "q_source": source,
                    "factor_position": factor_position + 1,
                    "status": "ok",
                    "observable_floor_centered": (
                        centered_bulk / centered_leading
                    ),
                    "observable_floor_uncentered": raw_bulk / raw_leading,
                    "mp_adjusted_floor_centered": (
                        centered_mp_bulk / centered_leading
                    ),
                    "centered_sample_eigenvalue": centered_leading,
                    "uncentered_sample_eigenvalue": raw_leading,
                    "centered_bulk_eigenvalue": centered_bulk,
                    "uncentered_bulk_eigenvalue": raw_bulk,
                    "rotation_component_observable": False,
                }
            )

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for source, q in requests:
        if q >= decomposition.n_nonzero:
            rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": q,
                    "q_source": source,
                    "correction": "multi_factor_sharp",
                    "status": "q_exceeds_rank",
                }
            )
            if q == 1:
                rows.append(
                    {
                        "formation_date": panel.window.formation,
                        "n_assets": p_assets,
                        "q_factors": q,
                        "q_source": source,
                        "correction": "one_factor_jse_centered_mp",
                        "status": "q_exceeds_rank",
                        "theory_applicable": False,
                        "centered": True,
                        "mp_adjusted": mp_adjust,
                        "positive_part": True,
                    }
                )
            continue
        try:
            fit = fit_spectral_decomposition(
                decomposition,
                q,
                mp_adjust=mp_adjust,
            )
        except WeakFactorError as exc:
            rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": q,
                    "q_source": source,
                    "correction": "multi_factor_sharp",
                    "status": "weak_factor",
                    "error": str(exc),
                }
            )
            if q == 1:
                rows.append(
                    {
                        "formation_date": panel.window.formation,
                        "n_assets": p_assets,
                        "q_factors": q,
                        "q_source": source,
                        "correction": "one_factor_jse_centered_mp",
                        "status": "weak_factor",
                        "theory_applicable": False,
                        "centered": True,
                        "mp_adjusted": mp_adjust,
                        "positive_part": True,
                        "error": str(exc),
                    }
                )
            continue

        localization = factor_localization(fit)
        localized = bool(
            np.any(
                localization.participation_fraction
                < min_participation_fraction
            )
            or np.any(
                localization.maximum_absolute_loading
                > max_absolute_loading
            )
        )
        sharp = optimization_bias_correction(fit, direction)
        variance_pair = _variance_pair(
            oos,
            before_loadings=fit.loadings,
            after_loadings=sharp.loadings,
            factor_eigenvalues=fit.factor_eigenvalues,
            gamma2=fit.gamma2,
        )
        key = (source, q, "multi_factor_sharp")
        if key not in seen:
            rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": q,
                    "q_source": source,
                    "correction": "multi_factor_sharp",
                    "status": "localized_factor" if localized else "ok",
                    "theory_applicable": not localized,
                    "centered": True,
                    "mp_adjusted": mp_adjust,
                    "variance_before": variance_pair.variance_before,
                    "variance_after": variance_pair.variance_after,
                    "variance_difference": (
                        variance_pair.variance_after
                        - variance_pair.variance_before
                    ),
                    "variance_ratio": (
                        variance_pair.variance_after
                        / variance_pair.variance_before
                        if variance_pair.variance_before > 0
                        else np.nan
                    ),
                    "kappa2": fit.kappa2,
                    "gamma2": fit.gamma2,
                    "phi_norm": float(np.linalg.norm(sharp.phi)),
                    "min_effective_assets": float(
                        np.min(localization.effective_assets)
                    ),
                    "min_participation_fraction": float(
                        np.min(localization.participation_fraction)
                    ),
                    "max_absolute_loading": float(
                        np.max(localization.maximum_absolute_loading)
                    ),
                    **variance_pair.weight_diagnostics,
                }
            )
            seen.add(key)

        if q == 1:
            jse = one_factor_jse(fit, direction, positive_part=True)
            jse_pair = _variance_pair(
                oos,
                before_loadings=fit.loadings,
                after_loadings=jse.loading[:, None],
                factor_eigenvalues=fit.factor_eigenvalues,
                gamma2=fit.gamma2,
            )
            key = (source, q, "one_factor_jse")
            if key not in seen:
                rows.append(
                    {
                        "formation_date": panel.window.formation,
                        "n_assets": p_assets,
                        "q_factors": q,
                        "q_source": source,
                        "correction": "one_factor_jse_centered_mp",
                        "status": "localized_factor" if localized else "ok",
                        "theory_applicable": not localized,
                        "centered": True,
                        "mp_adjusted": mp_adjust,
                        "positive_part": True,
                        "variance_before": jse_pair.variance_before,
                        "variance_after": jse_pair.variance_after,
                        "variance_difference": (
                            jse_pair.variance_after
                            - jse_pair.variance_before
                        ),
                        "variance_ratio": (
                            jse_pair.variance_after
                            / jse_pair.variance_before
                            if jse_pair.variance_before > 0
                            else np.nan
                        ),
                        "kappa2": fit.kappa2,
                        "gamma2": fit.gamma2,
                        "jse_intensity": jse.intensity,
                        "jse_raw_intensity": jse.raw_intensity,
                        "min_effective_assets": float(
                            np.min(localization.effective_assets)
                        ),
                        "min_participation_fraction": float(
                            np.min(localization.participation_fraction)
                        ),
                        "max_absolute_loading": float(
                            np.max(localization.maximum_absolute_loading)
                        ),
                        **jse_pair.weight_diagnostics,
                    }
                )
                seen.add(key)

    # Basis-dependent diagnostic for the concentration-paper decomposition.
    # Odd and even estimation dates form two nonoverlapping views of the same
    # asset panel.  Each ordered PCA basis is used once as candidate and once
    # as benchmark.  For q > 1 the resulting "rotation" depends on those
    # ordered axes and is neither invariant to within-subspace rotations nor
    # an identified population rotation component.
    split_matrices = (
        raw_x[::2] - raw_x[::2].mean(axis=0, keepdims=True),
        raw_x[1::2] - raw_x[1::2].mean(axis=0, keepdims=True),
    )
    split_decompositions = tuple(
        decompose_returns(matrix, center=False) for matrix in split_matrices
    )
    error_rows: list[dict[str, object]] = []
    error_cache: dict[int, list[dict[str, object]]] = {}
    for source, q in requests:
        if q not in error_cache:
            cached_rows: list[dict[str, object]] = []
            for fold, (candidate_index, benchmark_index) in enumerate(
                ((0, 1), (1, 0)),
                start=1,
            ):
                candidate_decomposition = split_decompositions[candidate_index]
                benchmark_decomposition = split_decompositions[benchmark_index]
                if q >= min(
                    candidate_decomposition.n_nonzero,
                    benchmark_decomposition.n_nonzero,
                ):
                    cached_rows.append(
                        {
                            "fold": fold,
                            "factor_position": np.nan,
                            "status": "q_exceeds_split_rank",
                        }
                    )
                    continue
                try:
                    split_fit = fit_spectral_decomposition(
                        candidate_decomposition,
                        q,
                        mp_adjust=mp_adjust,
                    )
                    benchmark_fit = fit_spectral_decomposition(
                        benchmark_decomposition,
                        q,
                        mp_adjust=mp_adjust,
                    )
                except WeakFactorError as exc:
                    cached_rows.append(
                        {
                            "fold": fold,
                            "factor_position": np.nan,
                            "status": "weak_factor",
                            "error": str(exc),
                        }
                    )
                    continue
                candidate_localization = factor_localization(split_fit)
                benchmark_localization = factor_localization(benchmark_fit)
                candidate_localized = bool(
                    np.any(
                        candidate_localization.participation_fraction
                        < min_participation_fraction
                    )
                    or np.any(
                        candidate_localization.maximum_absolute_loading
                        > max_absolute_loading
                    )
                )
                benchmark_localized = bool(
                    np.any(
                        benchmark_localization.participation_fraction
                        < min_participation_fraction
                    )
                    or np.any(
                        benchmark_localization.maximum_absolute_loading
                        > max_absolute_loading
                    )
                )
                diagnostic_localized = (
                    candidate_localized or benchmark_localized
                )
                split_corrected = optimization_bias_correction(
                    split_fit,
                    direction,
                )
                benchmark = benchmark_fit.loadings
                before_error = basis_error_decomposition(
                    split_fit.loadings,
                    benchmark,
                )
                after_error = basis_error_decomposition(
                    split_corrected.loadings,
                    benchmark,
                )
                for factor_position in range(q):
                    before_total = float(
                        before_error.total_sine_squared[factor_position]
                    )
                    before_out = float(
                        before_error.out_of_subspace[factor_position]
                    )
                    before_rotation = float(
                        before_error.in_subspace_rotation[factor_position]
                    )
                    after_total = float(
                        after_error.total_sine_squared[factor_position]
                    )
                    after_out = float(
                        after_error.out_of_subspace[factor_position]
                    )
                    after_rotation = float(
                        after_error.in_subspace_rotation[factor_position]
                    )
                    cached_rows.append(
                        {
                            "fold": fold,
                            "factor_position": factor_position + 1,
                            "status": (
                                "localized_factor"
                                if diagnostic_localized
                                else "ok"
                            ),
                            "theory_applicable": not diagnostic_localized,
                            "candidate_localized": candidate_localized,
                            "benchmark_localized": benchmark_localized,
                            "candidate_min_participation_fraction": float(
                                np.min(
                                    candidate_localization.participation_fraction
                                )
                            ),
                            "candidate_max_absolute_loading": float(
                                np.max(
                                    candidate_localization.maximum_absolute_loading
                                )
                            ),
                            "benchmark_min_participation_fraction": float(
                                np.min(
                                    benchmark_localization.participation_fraction
                                )
                            ),
                            "benchmark_max_absolute_loading": float(
                                np.max(
                                    benchmark_localization.maximum_absolute_loading
                                )
                            ),
                            "matched_benchmark_before": int(
                                before_error.matched_benchmark_index[
                                    factor_position
                                ]
                                + 1
                            ),
                            "matched_benchmark_after": int(
                                after_error.matched_benchmark_index[
                                    factor_position
                                ]
                                + 1
                            ),
                            "total_error_before": before_total,
                            "out_of_subspace_before": before_out,
                            "rotation_before": before_rotation,
                            "total_error_after": after_total,
                            "out_of_subspace_after": after_out,
                            "rotation_after": after_rotation,
                            "decomposition_identity_error_before": (
                                before_total - before_out - before_rotation
                            ),
                            "decomposition_identity_error_after": (
                                after_total - after_out - after_rotation
                            ),
                        }
                    )
            error_cache[q] = cached_rows
        for cached in error_cache[q]:
            error_rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": q,
                    "q_source": source,
                    "diagnostic": "odd_even_crossfit_ordered_basis_proxy",
                    "benchmark_definition": (
                        "other_half_sample_ordered_pca_axes"
                    ),
                    "basis_dependent_proxy": True,
                    "population_rotation_identified": False,
                    **cached,
                }
            )

    # Exact one-factor arm from Goldberg et al. (2025): uncentered YY'/n,
    # ell^2 equal to the nonleading eigenvalue average, and no MP adjustment.
    if paper_decomposition.n_nonzero <= 1:
        rows.append(
            {
                "formation_date": panel.window.formation,
                "n_assets": p_assets,
                "q_factors": 1,
                "q_source": "goldberg2025_exact",
                "correction": "one_factor_jse_goldberg2025",
                "status": "insufficient_rank",
                "centered": False,
                "mp_adjusted": False,
                "positive_part": True,
                "error": (
                    "the uncentered return matrix has fewer than two "
                    "nonzero singular values"
                ),
            }
        )
    else:
        try:
            paper_fit = fit_spectral_decomposition(
                paper_decomposition,
                1,
                mp_adjust=False,
            )
        except WeakFactorError as exc:
            rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": 1,
                    "q_source": "goldberg2025_exact",
                    "correction": "one_factor_jse_goldberg2025",
                    "status": "weak_factor",
                    "centered": False,
                    "mp_adjusted": False,
                    "positive_part": True,
                    "error": str(exc),
                }
            )
        else:
            paper_localization = factor_localization(paper_fit)
            paper_localized = bool(
                paper_localization.participation_fraction[0]
                < min_participation_fraction
                or paper_localization.maximum_absolute_loading[0]
                > max_absolute_loading
            )
            paper_jse = one_factor_jse(
                paper_fit,
                direction,
                positive_part=True,
            )
            paper_pair = _variance_pair(
                oos,
                before_loadings=paper_fit.loadings,
                after_loadings=paper_jse.loading[:, None],
                factor_eigenvalues=paper_fit.factor_eigenvalues,
                gamma2=paper_fit.gamma2,
            )
            rows.append(
                {
                    "formation_date": panel.window.formation,
                    "n_assets": p_assets,
                    "q_factors": 1,
                    "q_source": "goldberg2025_exact",
                    "correction": "one_factor_jse_goldberg2025",
                    "status": (
                        "localized_factor" if paper_localized else "ok"
                    ),
                    "theory_applicable": not paper_localized,
                    "variance_before": paper_pair.variance_before,
                    "variance_after": paper_pair.variance_after,
                    "variance_difference": (
                        paper_pair.variance_after
                        - paper_pair.variance_before
                    ),
                    "variance_ratio": (
                        paper_pair.variance_after
                        / paper_pair.variance_before
                        if paper_pair.variance_before > 0
                        else np.nan
                    ),
                    "kappa2": paper_fit.kappa2,
                    "gamma2": paper_fit.gamma2,
                    "jse_intensity": paper_jse.intensity,
                    "jse_raw_intensity": paper_jse.raw_intensity,
                    "centered": False,
                    "mp_adjusted": False,
                    "positive_part": True,
                    "min_effective_assets": float(
                        paper_localization.effective_assets[0]
                    ),
                    "min_participation_fraction": float(
                        paper_localization.participation_fraction[0]
                    ),
                    "max_absolute_loading": float(
                        paper_localization.maximum_absolute_loading[0]
                    ),
                    **paper_pair.weight_diagnostics,
                }
            )

    factor_diagnostics = formation_jung.factor_diagnostics.copy()
    factor_residuals = formation_jung.factor_residuals.copy()
    factor_selections = formation_jung.factor_selections.copy()
    comparison_frame = pd.DataFrame(rows)
    floor_frame = pd.DataFrame(floor_rows)
    if not floor_frame.empty and not comparison_frame.empty:
        status_columns = [
            "formation_date",
            "n_assets",
            "q_factors",
            "q_source",
            "status",
        ]
        if "theory_applicable" in comparison_frame:
            status_columns.append("theory_applicable")
        correction_status = (
            comparison_frame.loc[
                comparison_frame["correction"].eq("multi_factor_sharp"),
                status_columns,
            ]
            .drop_duplicates(
                ["formation_date", "n_assets", "q_factors", "q_source"]
            )
            .rename(columns={"status": "correction_status"})
        )
        floor_frame = floor_frame.rename(
            columns={"status": "spectral_floor_status"}
        ).merge(
            correction_status,
            on=["formation_date", "n_assets", "q_factors", "q_source"],
            how="left",
            validate="many_to_one",
        )
        floor_frame = floor_frame.assign(
            status=floor_frame["correction_status"].fillna(
                floor_frame["spectral_floor_status"]
            )
        )
        if "theory_applicable" not in floor_frame:
            floor_frame = floor_frame.assign(theory_applicable=False)
        floor_frame = floor_frame.assign(
            theory_applicable=floor_frame["theory_applicable"].eq(True)
        )
    return PanelExperiment(
        comparisons=comparison_frame,
        factor_diagnostics=factor_diagnostics,
        factor_selections=factor_selections,
        factor_residuals=factor_residuals,
        observable_error_floors=floor_frame,
        error_decomposition_diagnostics=pd.DataFrame(error_rows),
        jung=jung,
        oos_qa=oos_qa,
    )
