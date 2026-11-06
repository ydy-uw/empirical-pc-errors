"""Spectral estimators and strategy-specific eigenvector corrections.

The matrix convention is ``x.shape == (n_observations, p_assets)``.  The
centered sample covariance is ``x.T @ x / n``; its nonzero spectrum is computed
through a thin SVD, so no ``p x p`` covariance matrix is formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

FloatArray = NDArray[np.float64]


class WeakFactorError(ValueError):
    """Raised when a requested factor is not separated from the bulk estimate."""


@dataclass(frozen=True)
class SpectralDecomposition:
    """Reusable nonzero sample spectrum for one rolling return matrix."""

    loadings: FloatArray
    eigenvalues: FloatArray
    n_observations: int
    n_nonzero: int
    p_assets: int
    centered: bool


@dataclass(frozen=True)
class SpectralFit:
    """PCA factor fit with HDLSS eigenvalue and idiosyncratic corrections."""

    loadings: FloatArray
    sample_eigenvalues: FloatArray
    factor_eigenvalues: FloatArray
    kappa2: float
    psi: FloatArray
    gamma2: float
    n_observations: int
    n_nonzero: int
    p_assets: int
    q_factors: int
    centered: bool
    mp_adjusted: bool


@dataclass(frozen=True)
class OneFactorCorrection:
    """One-factor James-Stein correction toward a reference subspace."""

    loading: FloatArray
    intensity: float
    raw_intensity: float
    target_projection_norm2: float
    alpha: float
    intensity_was_capped: bool


@dataclass(frozen=True)
class MultiFactorCorrection:
    """Multi-factor correction candidate with stability diagnostics.

    ``raw_phi`` is the unregularized full-correction coefficient and
    ``phi_scale`` is the total scale imposed by the residual floor and norm
    cap, excluding ``alpha``.  It equals one when neither safeguard binds.
    """

    loadings: FloatArray
    phi: FloatArray
    residual_direction: FloatArray
    singular_values: FloatArray
    raw_phi: FloatArray
    alpha: float
    phi_scale: float
    residual_norm: float


@dataclass(frozen=True)
class FixedQCorrection:
    """Spectral fit and correction for one prespecified factor count."""

    fit: SpectralFit
    correction: MultiFactorCorrection
    q_factors: int
    alpha: float


@dataclass(frozen=True)
class FactorLocalization:
    """Finite-sample diagnostics for the pervasive-loading assumption."""

    inverse_participation_ratio: FloatArray
    effective_assets: FloatArray
    participation_fraction: FloatArray
    maximum_absolute_loading: FloatArray


@dataclass(frozen=True)
class BasisErrorDecomposition:
    """Matched vector error split into leakage and within-subspace rotation."""

    matched_benchmark_index: NDArray[np.int64]
    total_sine_squared: FloatArray
    out_of_subspace: FloatArray
    in_subspace_rotation: FloatArray


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    return value


def _validate_nonnegative_optional(
    value: float | None,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _as_finite_matrix(x: ArrayLike) -> FloatArray:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional observations-by-assets matrix")
    if min(matrix.shape) < 2:
        raise ValueError("x must contain at least two observations and two assets")
    if not np.isfinite(matrix).all():
        raise ValueError("x contains NaN or infinite values")
    return matrix


def _orient_columns(matrix: FloatArray) -> FloatArray:
    """Fix arbitrary SVD signs for byte-stable downstream artifacts."""
    result = np.array(matrix, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def decompose_returns(
    x: ArrayLike,
    *,
    center: bool = True,
) -> SpectralDecomposition:
    """Compute the thin-SVD spectrum once for reuse across factor counts."""
    matrix = _as_finite_matrix(x)
    n_observations, p_assets = matrix.shape
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = singular_values**2 / n_observations
    if eigenvalues[0] <= 0:
        raise ValueError("x has zero total variation")

    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * float(eigenvalues[0])
    n_nonzero = int(np.count_nonzero(eigenvalues > tolerance))
    return SpectralDecomposition(
        loadings=_orient_columns(vt[:n_nonzero].T),
        eigenvalues=eigenvalues[:n_nonzero].copy(),
        n_observations=n_observations,
        n_nonzero=n_nonzero,
        p_assets=p_assets,
        centered=center,
    )


def fit_spectral_decomposition(
    decomposition: SpectralDecomposition,
    q_factors: int,
    *,
    mp_adjust: bool = True,
    min_psi2: float = 1e-10,
) -> SpectralFit:
    """Fit HDLSS covariance ingredients from a reusable decomposition."""
    n_observations = decomposition.n_observations
    p_assets = decomposition.p_assets
    n_nonzero = decomposition.n_nonzero
    eigenvalues = decomposition.eigenvalues
    if not 1 <= q_factors < n_nonzero:
        raise ValueError(f"q_factors must be between 1 and n_nonzero-1={n_nonzero - 1}")

    tail_sum = float(eigenvalues[q_factors:n_nonzero].sum())
    if mp_adjust:
        aspect = n_observations / p_assets
        kappa2 = (1.0 + aspect) / (n_nonzero - q_factors + aspect) * tail_sum
    else:
        kappa2 = tail_sum / (n_nonzero - q_factors)

    leading = eigenvalues[:q_factors]
    psi2 = 1.0 - kappa2 / leading
    if np.any(psi2 <= min_psi2):
        failing = np.flatnonzero(psi2 <= min_psi2).tolist()
        raise WeakFactorError(
            "requested factors are not separated from the estimated bulk; "
            f"zero-based positions={failing}, kappa2={kappa2:.8g}"
        )

    loadings = decomposition.loadings[:, :q_factors].copy()
    psi = np.sqrt(psi2)
    factor_eigenvalues = leading * psi2
    gamma2 = n_observations * kappa2 / p_assets
    return SpectralFit(
        loadings=loadings,
        sample_eigenvalues=leading.copy(),
        factor_eigenvalues=factor_eigenvalues,
        kappa2=kappa2,
        psi=psi,
        gamma2=gamma2,
        n_observations=n_observations,
        n_nonzero=n_nonzero,
        p_assets=p_assets,
        q_factors=q_factors,
        centered=decomposition.centered,
        mp_adjusted=mp_adjust,
    )


def fit_spectral_model(
    x: ArrayLike,
    q_factors: int,
    *,
    center: bool = True,
    mp_adjust: bool = True,
    min_psi2: float = 1e-10,
) -> SpectralFit:
    """Fit the fixed-``n``, large-``p`` factor-covariance ingredients.

    ``kappa2`` is the average bulk sample eigenvalue.  With ``mp_adjust=True``
    it uses equation (25) of Gurdogan and Shkolnik (2024).  A requested factor
    must satisfy ``sample_eigenvalue > kappa2``; otherwise it is not a valid
    strong spike for the correction and :class:`WeakFactorError` is raised.
    """
    decomposition = decompose_returns(x, center=center)
    return fit_spectral_decomposition(
        decomposition,
        q_factors,
        mp_adjust=mp_adjust,
        min_psi2=min_psi2,
    )


def orthonormal_basis(vectors: ArrayLike, *, tolerance: float = 1e-12) -> FloatArray:
    """Return an orthonormal basis for the columns of ``vectors``."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("vectors must be a finite vector or matrix")
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    rank = int(np.count_nonzero(singular_values > tolerance * max(1.0, scale)))
    if rank == 0:
        raise ValueError("reference subspace has rank zero")
    return _orient_columns(u[:, :rank])


def factor_localization(fit: SpectralFit) -> FactorLocalization:
    """Measure whether each estimated factor is cross-sectionally pervasive."""
    squared = fit.loadings**2
    ipr = np.sum(squared**2, axis=0)
    effective = 1.0 / ipr
    return FactorLocalization(
        inverse_participation_ratio=ipr,
        effective_assets=effective,
        participation_fraction=effective / fit.p_assets,
        maximum_absolute_loading=np.max(np.abs(fit.loadings), axis=0),
    )


def basis_error_decomposition(
    estimated: ArrayLike,
    benchmark: ArrayLike,
    *,
    orthogonality_tolerance: float = 1e-8,
) -> BasisErrorDecomposition:
    """Decompose matched eigenvector error relative to a benchmark basis.

    For an estimated column ``h_j`` and its optimally matched benchmark column
    ``b_j`` in the benchmark subspace ``B``, the exact identity is

    ``1 - (b_j' h_j)^2
      = 1 - ||B' h_j||^2
      + ||B' h_j||^2 - (b_j' h_j)^2``.

    The first term is subspace leakage and is invariant to rotations of the
    benchmark basis. The second is the within-subspace rotation component.
    Squared correlations make the result invariant to arbitrary SVD signs.
    """
    h = np.asarray(estimated, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if (
        h.ndim != 2
        or b.ndim != 2
        or h.shape != b.shape
        or not np.isfinite(h).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "estimated and benchmark must be finite matrices of equal shape"
        )
    if h.shape[1] < 1:
        raise ValueError("bases must contain at least one column")
    identity = np.eye(h.shape[1])
    if (
        np.linalg.norm(h.T @ h - identity, ord="fro") > orthogonality_tolerance
        or np.linalg.norm(b.T @ b - identity, ord="fro") > orthogonality_tolerance
    ):
        raise ValueError("estimated and benchmark columns must be orthonormal")

    cross = b.T @ h
    squared = cross**2
    benchmark_rows, estimated_columns = linear_sum_assignment(-squared)
    matched = np.empty(h.shape[1], dtype=np.int64)
    matched[estimated_columns] = benchmark_rows
    matched_squared = squared[matched, np.arange(h.shape[1])]
    projected_squared = np.sum(squared, axis=0)

    tolerance = 100.0 * np.finfo(np.float64).eps
    out = 1.0 - projected_squared
    rotation = projected_squared - matched_squared
    total = 1.0 - matched_squared
    for values in (out, rotation, total):
        values[np.abs(values) < tolerance] = 0.0
    if (
        np.min(out) < -orthogonality_tolerance
        or np.min(rotation) < -orthogonality_tolerance
        or np.min(total) < -orthogonality_tolerance
    ):
        raise FloatingPointError("basis error decomposition became negative")
    return BasisErrorDecomposition(
        matched_benchmark_index=matched,
        total_sine_squared=np.maximum(total, 0.0),
        out_of_subspace=np.maximum(out, 0.0),
        in_subspace_rotation=np.maximum(rotation, 0.0),
    )


def one_factor_jse(
    fit: SpectralFit,
    reference: ArrayLike,
    *,
    positive_part: bool = True,
    alpha: float = 1.0,
    max_intensity: float | None = None,
    tolerance: float = 1e-12,
) -> OneFactorCorrection:
    """Apply the Goldberg-Kercheval-Shkolnik one-factor JSE.

    ``reference`` contains one or more vectors spanning the strategy's
    constraint/reference subspace.  The raw formula is retained in the returned
    diagnostics.  ``alpha`` continuously scales the raw correction: zero is
    the exact PCA baseline and one reproduces the original full correction.
    ``positive_part=True`` clips the scaled finite-sample intensity to
    ``[0, 1]`` and is the prespecified empirical safeguard; the optional
    ``max_intensity`` can impose a tighter stability cap.  The asymptotic
    theory places the unscaled intensity strictly inside ``(0, 1)``.
    """
    if fit.q_factors != 1:
        raise ValueError("one_factor_jse requires a q_factors=1 spectral fit")
    alpha = _validate_alpha(alpha)
    max_intensity = _validate_nonnegative_optional(
        max_intensity,
        name="max_intensity",
    )
    basis = orthonormal_basis(reference)
    if basis.shape[0] != fit.p_assets:
        raise ValueError("reference dimension does not match the asset dimension")

    h = fit.loadings[:, 0]
    projection = basis @ (basis.T @ h)
    projection_norm2 = float(projection @ projection)
    complement_norm2 = max(0.0, 1.0 - projection_norm2)
    denominator = float(fit.sample_eigenvalues[0] * complement_norm2)
    if denominator <= tolerance:
        return OneFactorCorrection(
            loading=h.copy(),
            intensity=0.0,
            raw_intensity=np.nan,
            target_projection_norm2=projection_norm2,
            alpha=alpha,
            intensity_was_capped=False,
        )

    raw_intensity = fit.kappa2 / denominator
    scaled_intensity = alpha * raw_intensity
    if positive_part:
        upper = 1.0 if max_intensity is None else min(1.0, max_intensity)
        intensity = float(np.clip(scaled_intensity, 0.0, upper))
    elif max_intensity is not None:
        intensity = float(np.clip(scaled_intensity, -max_intensity, max_intensity))
    else:
        intensity = float(scaled_intensity)
    intensity_was_capped = intensity != scaled_intensity
    if intensity == 0.0:
        return OneFactorCorrection(
            loading=h.copy(),
            intensity=intensity,
            raw_intensity=float(raw_intensity),
            target_projection_norm2=projection_norm2,
            alpha=alpha,
            intensity_was_capped=intensity_was_capped,
        )
    candidate = intensity * projection + (1.0 - intensity) * h
    norm = float(np.linalg.norm(candidate))
    if norm <= tolerance:
        raise ValueError("JSE correction collapsed to a zero vector")
    candidate /= norm
    candidate = _orient_columns(candidate[:, None])[:, 0]
    return OneFactorCorrection(
        loading=candidate,
        intensity=intensity,
        raw_intensity=float(raw_intensity),
        target_projection_norm2=projection_norm2,
        alpha=alpha,
        intensity_was_capped=intensity_was_capped,
    )


def optimization_bias_correction(
    fit: SpectralFit,
    optimization_direction: ArrayLike,
    *,
    alpha: float = 1.0,
    residual_floor: float | None = None,
    max_phi_norm: float | None = None,
    tolerance: float = 1e-12,
) -> MultiFactorCorrection:
    """Construct a full or partial candidate toward the ``H_sharp`` loadings.

    This implements equations (35)--(37) of Gurdogan and Shkolnik (2024).
    The direction is normalized internally; for a fully invested portfolio it
    is the all-ones vector.  ``alpha=0`` returns the exact PCA loading matrix,
    while ``alpha=1`` with inactive safeguards reproduces the original
    optimization-bias-free ``H_sharp`` correction.  Intermediate values
    interpolate the pre-orthogonalization candidate matrix.  The returned
    oriented left-singular basis is not guaranteed to trace a continuous path
    because singular values can coincide or cross.
    ``residual_floor`` regularizes the denominator used to construct ``phi``
    and ``max_phi_norm`` caps the norm of the applied out-of-subspace
    perturbation.  Both safeguards are inactive by default; once either binds,
    the result is a stabilized candidate rather than the exact theoretical
    full correction.
    """
    alpha = _validate_alpha(alpha)
    residual_floor = _validate_nonnegative_optional(
        residual_floor,
        name="residual_floor",
    )
    max_phi_norm = _validate_nonnegative_optional(
        max_phi_norm,
        name="max_phi_norm",
    )
    z = np.asarray(optimization_direction, dtype=np.float64).reshape(-1)
    if z.size != fit.p_assets or not np.isfinite(z).all():
        raise ValueError("optimization_direction must be finite and match p_assets")
    z_norm = float(np.linalg.norm(z))
    if z_norm <= tolerance:
        raise ValueError("optimization_direction has zero norm")
    z = z / z_norm

    h = fit.loadings
    scores = h.T @ z
    residual = z - h @ scores
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= tolerance:
        raise ValueError("optimization direction lies in the estimated factor subspace")
    z_perp = residual / residual_norm
    numerator = ((1.0 / fit.psi) - fit.psi) * scores
    raw_phi = numerator / residual_norm
    denominator = (
        residual_norm if residual_floor is None else max(residual_norm, residual_floor)
    )
    phi_scale = residual_norm / denominator
    phi = alpha * raw_phi * phi_scale
    phi_norm = float(np.linalg.norm(phi))
    if max_phi_norm is not None and phi_norm > max_phi_norm:
        cap_scale = max_phi_norm / phi_norm
        phi_scale *= cap_scale
        phi = phi * cap_scale

    if alpha == 0.0 or not np.any(phi):
        scales = np.ones_like(fit.psi) + alpha * (fit.psi - 1.0)
        return MultiFactorCorrection(
            loadings=h.copy(),
            phi=phi,
            residual_direction=z_perp,
            singular_values=scales,
            raw_phi=raw_phi,
            alpha=alpha,
            phi_scale=phi_scale,
            residual_norm=residual_norm,
        )

    if alpha == 1.0 and phi_scale == 1.0:
        scales = fit.psi
        phi = raw_phi
    else:
        scales = np.ones_like(fit.psi) + alpha * (fit.psi - 1.0)
    candidate = h * scales[None, :] + np.outer(z_perp, phi)

    corrected, singular_values, _ = np.linalg.svd(candidate, full_matrices=False)
    corrected = _orient_columns(corrected[:, : fit.q_factors])
    return MultiFactorCorrection(
        loadings=corrected,
        phi=phi,
        residual_direction=z_perp,
        singular_values=singular_values[: fit.q_factors],
        raw_phi=raw_phi,
        alpha=alpha,
        phi_scale=phi_scale,
        residual_norm=residual_norm,
    )


def fixed_q_optimization_correction(
    decomposition: SpectralDecomposition,
    q_factors: int,
    optimization_direction: ArrayLike,
    *,
    alpha: float = 1.0,
    mp_adjust: bool = True,
    min_psi2: float = 1e-10,
    residual_floor: float | None = None,
    max_phi_norm: float | None = None,
    tolerance: float = 1e-12,
) -> FixedQCorrection:
    """Fit and correct one prespecified ``q`` from a reusable decomposition.

    This convenience interface is intended for controlled ``(q, alpha)``
    grids.  Call :func:`decompose_returns` once, then call this function for
    each fixed factor count.  It deliberately leaves factor-count selection
    outside the correction so that experiments cannot tune ``q`` on the
    evaluation returns.
    """
    fit = fit_spectral_decomposition(
        decomposition,
        q_factors,
        mp_adjust=mp_adjust,
        min_psi2=min_psi2,
    )
    correction = optimization_bias_correction(
        fit,
        optimization_direction,
        alpha=alpha,
        residual_floor=residual_floor,
        max_phi_norm=max_phi_norm,
        tolerance=tolerance,
    )
    return FixedQCorrection(
        fit=fit,
        correction=correction,
        q_factors=fit.q_factors,
        alpha=correction.alpha,
    )
