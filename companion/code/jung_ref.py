"""Jung (2018) sequential skewness estimator of the factor count."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import skewtest


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JungFactorCount:
    """Sequential residual-skewness results."""

    selected: int | None
    censored: bool
    alpha: float
    tested_k: NDArray[np.int64]
    statistics: FloatArray
    pvalues: FloatArray
    residual_means: FloatArray
    residual_lengths: FloatArray
    sample_eigenvalues: FloatArray


def select_from_pvalues(pvalues: ArrayLike, alpha: float) -> int | None:
    """Return the first ``k`` whose one-sided p-value exceeds ``alpha``."""
    values = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("pvalues must be a nonempty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pvalues must lie in the closed unit interval")
    accepted = np.flatnonzero(values > alpha)
    return int(accepted[0]) if accepted.size else None


def jung_factor_count(
    x: ArrayLike,
    *,
    alpha: float = 0.10,
    max_factors: int = 15,
    center: bool = True,
) -> JungFactorCount:
    """Estimate the number of pervasive factors using Jung's sequential test.

    For each ``k``, the observation-level statistic is
    ``R_j(k) = ||x_j - reconstruction_j(k)||^2 / p``.  A one-sided
    D'Agostino skewness test rejects ``m=k`` for right skewness.  The estimate
    is the first non-rejected ``k``.
    """
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("x must be a finite observations-by-variables matrix")
    n_observations, p_variables = matrix.shape
    if n_observations < 8:
        raise ValueError("D'Agostino's skewness test requires at least 8 observations")
    if p_variables < 2:
        raise ValueError("x must contain at least two variables")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if (
        isinstance(max_factors, bool)
        or not isinstance(max_factors, (int, np.integer))
        or max_factors < 0
    ):
        raise ValueError("max_factors must be a nonnegative integer")

    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    rank_tolerance = (
        np.finfo(np.float64).eps
        * max(matrix.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    last_k = min(max_factors, max(0, rank - 1))

    row_total = np.einsum("ij,ij->i", matrix, matrix)
    explained_by_component = (u * singular_values[None, :]) ** 2
    cumulative_explained = np.column_stack(
        [
            np.zeros(n_observations, dtype=np.float64),
            np.cumsum(explained_by_component[:, :last_k], axis=1),
        ]
    )

    tested_k = np.arange(last_k + 1, dtype=np.int64)
    statistics = np.empty(last_k + 1, dtype=np.float64)
    pvalues = np.empty(last_k + 1, dtype=np.float64)
    residual_means = np.empty(last_k + 1, dtype=np.float64)
    residual_path = np.empty(
        (last_k + 1, n_observations),
        dtype=np.float64,
    )
    residual_tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(row_total)))
    )
    for k in tested_k:
        residual_squares = row_total - cumulative_explained[:, k]
        if float(np.min(residual_squares)) < -residual_tolerance:
            raise FloatingPointError(
                "principal-component residual norm became materially negative"
            )
        residual_lengths = np.maximum(0.0, residual_squares) / p_variables
        residual_path[k] = residual_lengths
        result = skewtest(residual_lengths, alternative="greater")
        statistics[k] = float(result.statistic)
        pvalues[k] = float(result.pvalue)
        residual_means[k] = float(residual_lengths.mean())

    selected = select_from_pvalues(pvalues, alpha)
    censored = selected is None
    return JungFactorCount(
        selected=selected,
        censored=censored,
        alpha=alpha,
        tested_k=tested_k,
        statistics=statistics,
        pvalues=pvalues,
        residual_means=residual_means,
        residual_lengths=residual_path,
        sample_eigenvalues=(singular_values[:rank] ** 2 / n_observations),
    )
