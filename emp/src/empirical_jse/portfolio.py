"""Low-rank covariance operations and realized-variance evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def minimum_variance_weights(
    loadings: ArrayLike,
    factor_eigenvalues: ArrayLike,
    gamma2: float,
    *,
    budget_vector: ArrayLike | None = None,
) -> FloatArray:
    """Return minimum-variance weights without forming a dense covariance.

    The covariance is ``gamma2 * I + H diag(d) H.T`` with orthonormal columns
    in ``H``.  The equality constraint is ``budget_vector.T @ w = 1``; the
    default budget vector is all ones.
    """
    h = np.asarray(loadings, dtype=np.float64)
    d = np.asarray(factor_eigenvalues, dtype=np.float64).reshape(-1)
    if h.ndim != 2 or h.shape[1] != d.size:
        raise ValueError("loadings and factor_eigenvalues have incompatible shapes")
    if not np.isfinite(h).all() or not np.isfinite(d).all():
        raise ValueError("covariance ingredients must be finite")
    if gamma2 <= 0 or np.any(d < 0):
        raise ValueError("gamma2 must be positive and factor eigenvalues nonnegative")
    gram = h.T @ h
    if not np.allclose(gram, np.eye(h.shape[1]), atol=1e-8, rtol=1e-8):
        raise ValueError("loadings must have orthonormal columns")

    if budget_vector is None:
        zeta = np.ones(h.shape[0], dtype=np.float64)
    else:
        zeta = np.asarray(budget_vector, dtype=np.float64).reshape(-1)
    if zeta.size != h.shape[0] or not np.isfinite(zeta).all():
        raise ValueError("budget_vector must be finite and match the asset dimension")

    shrink = d / (gamma2 + d)
    inverse_times_budget = (zeta - h @ (shrink * (h.T @ zeta))) / gamma2
    normalizer = float(zeta @ inverse_times_budget)
    if not np.isfinite(normalizer) or abs(normalizer) <= 1e-14:
        raise ValueError("budget constraint has a singular normalization")
    return inverse_times_budget / normalizer


def portfolio_returns(
    returns: ArrayLike,
    weights: ArrayLike,
    *,
    missing_after_terminal_as_cash: bool = False,
) -> FloatArray:
    """Calculate portfolio returns from an OOS returns matrix.

    Primary panels should already encode terminal CIZ returns once and zeros
    thereafter.  ``missing_after_terminal_as_cash`` is provided only for a
    validated panel where remaining missing observations are known to occur
    strictly after each security's terminal row.
    """
    matrix = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != w.size:
        raise ValueError("returns and weights have incompatible shapes")
    if missing_after_terminal_as_cash:
        matrix = np.nan_to_num(matrix, nan=0.0)
    elif not np.isfinite(matrix).all():
        raise ValueError("OOS returns contain missing or infinite values")
    return matrix @ w


def realized_variance(
    returns: ArrayLike,
    weights: ArrayLike,
    *,
    ddof: int = 1,
    annualization: float | None = None,
) -> float:
    """Return sample variance of daily OOS portfolio returns."""
    series = portfolio_returns(returns, weights)
    if series.size <= ddof:
        raise ValueError("not enough OOS observations for the requested ddof")
    value = float(np.var(series, ddof=ddof))
    return value if annualization is None else value * annualization
