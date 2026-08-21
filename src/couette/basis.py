"""Polynomial Galerkin bases and Gauss-Legendre quadrature on y in [-1, 1].

The wall-normal discretisation used throughout is a *Galerkin* one built from
Chebyshev polynomials multiplied by boundary bubble factors, so that every
basis function satisfies the boundary conditions exactly:

    v-basis   phi_n(y) = (1 - y^2)^2 T_n(y)     ->  phi = phi' = 0 at y = +-1
    eta-basis psi_n(y) = (1 - y^2)   T_n(y)     ->  psi      = 0 at y = +-1

All inner products are evaluated with Gauss-Legendre quadrature using enough
nodes that every integrand (a polynomial) is integrated *exactly* in exact
arithmetic; the only error is floating-point round-off.  This is what makes the
resulting matrices exactly symmetric/Hermitian, which in turn lets us use a
symmetric generalised eigensolver and get real eigenvalues to machine precision.
"""

from __future__ import annotations

import numpy as np


def gauss_legendre(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [-1, 1] (exact for degree <= 2n-1)."""
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    return nodes, weights


def chebyshev_values(n_max: int, y: np.ndarray, n_deriv: int = 0) -> np.ndarray:
    """Values of T_0..T_{n_max} and their derivatives at the points ``y``.

    Returns an array of shape ``(n_deriv + 1, n_max + 1, len(y))`` where entry
    ``[d, n, :]`` holds ``d^d/dy^d T_n(y)``.

    Uses the Chebyshev series machinery in numpy, which evaluates by the
    Clenshaw recurrence and differentiates in coefficient space; both are
    numerically stable for the moderate degrees (<= a few hundred) used here.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros((n_deriv + 1, n_max + 1, y.size))
    for n in range(n_max + 1):
        coef = np.zeros(n + 1)
        coef[n] = 1.0
        for d in range(n_deriv + 1):
            c = np.polynomial.chebyshev.chebder(coef, m=d) if d > 0 else coef
            out[d, n, :] = np.polynomial.chebyshev.chebval(y, c)
    return out


def bubble_values(power: int, y: np.ndarray, n_deriv: int = 0) -> np.ndarray:
    """Values and derivatives of ``b(y) = (1 - y^2)^power`` at ``y``.

    Returns shape ``(n_deriv + 1, len(y))``.
    """
    y = np.asarray(y, dtype=float)
    # (1 - y^2)^power as a plain power series, then differentiate exactly.
    base = np.polynomial.polynomial.polypow(np.array([1.0, 0.0, -1.0]), power)
    out = np.zeros((n_deriv + 1, y.size))
    for d in range(n_deriv + 1):
        c = np.polynomial.polynomial.polyder(base, m=d) if d > 0 else base
        out[d, :] = np.polynomial.polynomial.polyval(y, c)
    return out


def _leibniz(bub: np.ndarray, cheb: np.ndarray, order: int) -> np.ndarray:
    """d^order/dy^order of ``b(y) * T_n(y)`` via the Leibniz rule.

    ``bub[d, :]`` are derivatives of the bubble, ``cheb[d, n, :]`` of T_n.
    Returns shape ``(n_max + 1, len(y))``.
    """
    from math import comb

    total = np.zeros_like(cheb[0])
    for j in range(order + 1):
        total += comb(order, j) * bub[j][None, :] * cheb[order - j]
    return total


class WallNormalBasis:
    """Galerkin basis ``(1 - y^2)^power * T_n(y)``, n = 0..n_max.

    Attributes
    ----------
    d : list of arrays
        ``d[k]`` has shape ``(n_max + 1, n_quad)`` and holds the k-th derivative
        of each basis function at the quadrature nodes.
    """

    def __init__(self, n_max: int, power: int, nodes: np.ndarray,
                 weights: np.ndarray, max_deriv: int = 2):
        self.n_max = n_max
        self.power = power
        self.nodes = nodes
        self.weights = weights
        self.size = n_max + 1

        cheb = chebyshev_values(n_max, nodes, n_deriv=max_deriv)
        bub = bubble_values(power, nodes, n_deriv=max_deriv)
        self.d = [_leibniz(bub, cheb, k) for k in range(max_deriv + 1)]

    def gram(self, i: int, j: int, weight: np.ndarray | None = None) -> np.ndarray:
        """Matrix ``G[m, n] = \\int w(y) * f_m^{(i)}(y) * f_n^{(j)}(y) dy``."""
        w = self.weights if weight is None else self.weights * weight
        return np.einsum("mq,nq,q->mn", self.d[i], self.d[j], w)

    def cross(self, other: "WallNormalBasis", i: int, j: int,
              weight: np.ndarray | None = None) -> np.ndarray:
        """Matrix ``C[m, n] = \\int w(y) * self_m^{(i)} * other_n^{(j)} dy``."""
        w = self.weights if weight is None else self.weights * weight
        return np.einsum("mq,nq,q->mn", self.d[i], other.d[j], w)

    def evaluate(self, coeffs: np.ndarray, y: np.ndarray, deriv: int = 0) -> np.ndarray:
        """Evaluate ``sum_n coeffs[n] * f_n^{(deriv)}(y)`` at arbitrary points."""
        cheb = chebyshev_values(self.n_max, y, n_deriv=deriv)
        bub = bubble_values(self.power, y, n_deriv=deriv)
        vals = _leibniz(bub, cheb, deriv)
        return coeffs @ vals


def make_bases(n_max: int, n_quad: int | None = None,
               max_deriv: int = 2) -> tuple[WallNormalBasis, WallNormalBasis,
                                            np.ndarray, np.ndarray]:
    """Build the ``v`` (clamped) and ``eta`` (Dirichlet) bases sharing one quadrature.

    The quadrature is chosen so that the highest-degree integrand appearing in
    the Reynolds-Orr forms is integrated exactly.  The largest integrand degree
    is ``2 * (n_max + 4)`` (from ``\\int phi_m phi_n`` with the quartic bubble),
    so ``n_quad = n_max + 6`` nodes already suffice; we add a margin.
    """
    if n_quad is None:
        n_quad = n_max + 12
    nodes, weights = gauss_legendre(n_quad)
    v_basis = WallNormalBasis(n_max, power=2, nodes=nodes, weights=weights,
                              max_deriv=max_deriv)
    eta_basis = WallNormalBasis(n_max, power=1, nodes=nodes, weights=weights,
                                max_deriv=max_deriv)
    return v_basis, eta_basis, nodes, weights
