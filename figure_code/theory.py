"""Closed-form finite-p approximations for the Theorem 2 estimate A_j = ell/theta_j.

Setting: n non-degenerate dual directions (n = N - 1 after demeaning), k pervasive
factors, specific returns Z = Sigma^{1/2} E with the same Sigma on every date, and the
effective cross-section p_eff = (tr Sigma)^2 / tr(Sigma^2).  Write a_j = OE_j/(1-OE_j).

First order (mean zero; outline Proposition A):
    sd(A_j - OE_j) ~= OE_j * sqrt( 2 (1 + 1/(n-k)) / p_eff )            Gaussian E
Second order (the leading bias; outline Proposition B), Gaussian specific returns:
    E[A_j - OE_j] ~= -(OE_j/p_eff) * [ 2 (n-k) a_j + sum_i a_i
                                        + 2 sum_{l != j} (1/a_j - 1/a_l)^{-1} - 2 OE_j ]
The last term is the covariance between the numerator and the denominator of
(ell - rho_j)/theta_j.  Both formulas are checked against simulation in
p03_finite_p_error.py and figure_code/checks/bias_check.py.  The bias formula needs
(n-k) a_j^2 / p_eff small and well-separated directions; it is unreliable when two
directions are nearly degenerate (the pair term diverges).  Heavy tails (kurtosis)
enlarge both the sd and the bias.
"""
from __future__ import annotations
import numpy as np


def sd_first_order(oe, peff, n=62, k=3):
    oe = np.asarray(oe, float)
    return oe * np.sqrt(2 * (1 + 1 / (n - k)) / peff)


def bias_second_order(oe, peff, n=62):
    """Leading bias E[A_j - OE_j] for every j, given the vector OE_1..OE_k."""
    oe = np.asarray(oe, float)
    k = len(oe)
    a = oe / (1 - oe)
    out = np.empty(k)
    for j in range(k):
        pair = sum(1.0 / (1.0 / a[j] - 1.0 / a[l]) for l in range(k) if l != j)
        out[j] = -(oe[j] / peff) * (2 * (n - k) * a[j] + a.sum() + 2 * pair - 2 * oe[j])
    return out


def bias_corrected(A, peff, n=62):
    """Plug-in correction A_j - bias(A): removes the leading bias to first order."""
    A = np.asarray(A, float)
    return A - bias_second_order(A, peff, n)


def reversal_rate(oe, peff, n=62):
    """Approximate Pr(A_j > OE_j) = Phi(bias/sd) from Propositions A and B (Gaussian)."""
    from math import erf, sqrt
    oe = np.asarray(oe, float)
    b = bias_second_order(oe, peff, n)
    sd = sd_first_order(oe, peff, n, len(oe))
    return np.array([0.5 * (1 + erf(x / sqrt(2))) for x in b / sd])


def min_relative_gap(theta):
    """Smallest relative gap (theta_j - theta_{j+1})/theta_j among consecutive spikes."""
    t = np.sort(np.asarray(theta, float))[::-1]
    return float(np.min((t[:-1] - t[1:]) / t[:-1])) if len(t) > 1 else np.inf
