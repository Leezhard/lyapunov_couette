"""The shear-shifted Reynolds-Orr problem, and the optimal shift profile.

Background (see ``docs/03_quartic_lyapunov.md`` for the derivation).  Adding a
term ``2 E <W, u>`` to a quartic Lyapunov functional changes its *quartic* part
in exactly one way: the base-flow rate-of-strain ``S^U`` is replaced by the
shifted strain ``S^U - S^W``.  For a streamwise shift field ``W = (g(y), 0, 0)``
this means the Reynolds-Orr weight becomes

    Sigma(y) = 1 - g'(y).

Two structural constraints pin down what is admissible:

  * ``g(+-1) = 0`` is forced if we want to control ``<W, Laplacian u>`` by
    ``||u||_{L^2}`` alone (otherwise a wall-shear-stress boundary term survives
    and needs H^2 control that ``dV/dt`` does not provide);
  * therefore ``int_{-1}^{1} g' dy = g(1) - g(-1) = 0``, so ``Sigma`` has mean
    exactly 1.  Shear can only be *redistributed* across the channel, never
    reduced on average.

The useful fact, established numerically in ``scripts/03_shear_shift.py``, is
that redistribution alone is already very effective, because the critical mode
concentrates its Reynolds stress in the core while the no-slip condition
suppresses it near the walls.

First-order theory
------------------
Let ``u_c`` be the critical mode normalised so that its dissipation integral is
1, and define the *Reynolds-stress density*

    Phi(y) = - Re[ u_c-hat(y) conj(v_c-hat(y)) ]        (so lambda_0 = int Phi dy).

Perturbing the weight by ``delta Sigma = -g'`` gives, by Hellmann-Feynman,

    delta lambda = - int g'(y) Phi(y) dy = + int g(y) Phi'(y) dy

after an integration by parts that uses ``g(+-1) = 0``.  Hence the shift that
buys the most reduction of ``lambda`` per unit of ``||g||_{L^2}`` is

    g_opt(y) = - c Phi'(y),        delta lambda = - c ||Phi'||^2 < 0,

and the minimum L^2 cost of raising the threshold from ``Re_E`` to ``Re`` is

    ||g||_{L^2} >= ( 1/Re_E - 1/Re ) / ||Phi'||_{L^2}   (to leading order).
"""

from __future__ import annotations

import numpy as np

from . import reynolds_orr as ro
from .basis import gauss_legendre, make_bases


def reynolds_stress_density(alpha: float, beta: float, n_max: int = 48,
                            n_eval: int = 2001, sigma=None):
    """``(y, Phi, Phi')`` for the critical mode, normalised to unit dissipation.

    ``Phi(y) = -Re[u conj(v)]`` is the wall-normal density of the Reynolds
    stress that drives the energy growth; ``lambda_max = int Phi dy``.
    """
    sol = ro.solve_mode(alpha, beta, n_max=n_max, sigma=sigma)
    y = np.linspace(-1.0, 1.0, n_eval)
    u, du, v, dv, _, _ = ro.mode_profiles_and_derivatives(sol, y)

    # Normalise so that the dissipation integral equals 1.  Both the integral
    # and the derivatives below are exact, so Phi'(+-1) = 0 holds to round-off
    # -- which is what makes the shift automatically admissible (docs/03 Sec 3.5).
    scale = 1.0 / np.sqrt(ro.dissipation(sol))
    u, du, v, dv = u * scale, du * scale, v * scale, dv * scale

    phi = -np.real(u * np.conj(v))
    dphi = -np.real(du * np.conj(v) + u * np.conj(dv))
    return y, phi, dphi, sol


def optimal_shift_L2(alpha: float = 0.0, beta: float = 1.5581617774,
                     n_max: int = 48, n_eval: int = 2001):
    """The L^2-cheapest shift direction ``g ~ -Phi'`` and its efficiency.

    Returns ``(y, g_hat, efficiency)`` where ``g_hat`` has unit L^2 norm and
    ``efficiency = ||Phi'||_{L2}`` is the reduction in ``lambda`` bought per unit
    of ``||g||_{L2}`` at first order.
    """
    y, phi, dphi, sol = reynolds_stress_density(alpha, beta, n_max=n_max,
                                                n_eval=n_eval)
    norm = np.sqrt(np.trapezoid(dphi ** 2, y))
    g_hat = -dphi / norm
    return y, g_hat, float(norm), sol


def min_cost_for_target(re_target: float, re_e: float, efficiency: float) -> float:
    """Leading-order minimum ``||g||_{L2}`` needed to reach ``Re_E[Sigma] = re_target``."""
    return (1.0 / re_e - 1.0 / re_target) / efficiency


class ShearShiftBasis:
    """Odd polynomial shifts ``g(y) = sum_n c_n (1 - y^2) T_{2n+1}(y)``.

    Odd ``g`` gives even ``g'``, hence an even weight ``Sigma`` -- which is what
    the reflection symmetry ``(x, y, z, u, v, w) -> (-x, -y, z, -u, -v, w)`` of
    plane Couette flow demands of an optimal shift.  The factor ``(1 - y^2)``
    enforces ``g(+-1) = 0``.
    """

    def __init__(self, n_modes: int, n_quad: int = 200):
        self.n_modes = n_modes
        self.nodes, self.weights = gauss_legendre(n_quad)
        self._cache: dict[int, np.ndarray] = {}
        for d in (0, 1, 2):
            self._cache[d] = self._build(d, self.nodes)

    def _build(self, deriv: int, y: np.ndarray) -> np.ndarray:
        from .basis import bubble_values, chebyshev_values, _leibniz
        n_cheb = 2 * self.n_modes  # need T_1, T_3, ..., T_{2m-1}
        cheb = chebyshev_values(n_cheb, y, n_deriv=deriv)
        bub = bubble_values(1, y, n_deriv=deriv)
        vals = _leibniz(bub, cheb, deriv)          # (n_cheb+1, len(y))
        return vals[1::2][: self.n_modes]          # keep odd Chebyshev indices

    def g(self, c: np.ndarray, deriv: int = 0, y: np.ndarray | None = None):
        if y is None:
            return c @ self._cache[deriv]
        return c @ self._build(deriv, y)

    def norm_sq(self, c: np.ndarray, deriv: int = 0) -> float:
        vals = self.g(c, deriv=deriv)
        return float(np.sum(self.weights * vals ** 2))

    def sigma(self, c: np.ndarray):
        """Return a callable ``Sigma(y) = 1 - g'(y)`` for the solver."""
        def _sigma(y):
            return 1.0 - self.g(c, deriv=1, y=np.asarray(y, dtype=float))
        return _sigma


def lambda_max_over_grid(c: np.ndarray, sb: ShearShiftBasis,
                         wavenumbers, n_max: int = 32) -> float:
    """``max_{(alpha,beta) in grid} lambda(alpha, beta; Sigma[c])``.

    This is a maximum of functions affine in ``c``, hence *convex* in ``c`` --
    which is what makes the search for an optimal shift a convex problem.
    """
    sig = sb.sigma(c)
    return max(ro.growth(a, b, n_max=n_max, sigma=sig) for a, b in wavenumbers)
