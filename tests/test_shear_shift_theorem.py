"""Numerical verification of the shear-shift theorem, docs/03 Eq. (3.6).

The theorem states that for a divergence-free W with n.W = 0 on the walls,

    <W, F2(u)> = - int W_i u_j d_j u_i dV = int u_i u_j S^W_ij dV,

and for W = (g(y), 0, 0) the right-hand side is int g'(y) u v dV.  This is the
identity that turns the quartic part of dV/dt into a Reynolds-Orr problem with
shifted strain, so it is worth checking against a directly evaluated nonlinear
term rather than trusting the integration by parts.

For a single Fourier mode u = Re[uhat(y) e^{i(alpha x + beta z)}] the xz-average
of a product is (1/2)Re[a conj(b)], which gives (per unit xz-area)

    <(u.grad)u>_x  =  (1/2) Re[ vhat conj(D uhat) ]  +  (beta/2) Im[ what conj(uhat) ]

(the u d_x u contribution averages to zero).  Note the identity needs only
u.n = 0 on the walls, not g(+-1) = 0 -- that separate constraint comes from the
viscous term -- so the test deliberately uses a g that does NOT vanish there.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import reynolds_orr as ro       # noqa: E402


def _both_sides(alpha, beta, g, dg, n_max=36, n_eval=4001):
    sol = ro.solve_mode(alpha, beta, n_max=n_max)
    y = np.linspace(-1.0, 1.0, n_eval)
    u, du, v, dv, w, dw = ro.mode_profiles_and_derivatives(sol, y)

    # <W, F2(u)> per unit xz-area, evaluated from the advective term directly
    adv = 0.5 * np.real(v * np.conj(du)) + 0.5 * beta * np.imag(w * np.conj(u))
    lhs = -np.trapezoid(g(y) * adv, y)

    # int u_i u_j S^W_ij dV = int g'(y) u v dV, per unit xz-area
    rhs = np.trapezoid(dg(y) * 0.5 * np.real(u * np.conj(v)), y)
    scale = max(abs(lhs), abs(rhs))
    return lhs, rhs, scale


@pytest.mark.parametrize("alpha,beta", [(0.0, 1.5582), (0.4, 1.2), (0.9, 0.7),
                                        (1.5, 2.1)])
def test_shear_shift_identity(alpha, beta):
    """g(y) = y^3 - 0.4y + 0.3: deliberately non-zero at the walls."""
    g = lambda yy: yy ** 3 - 0.4 * yy + 0.3
    dg = lambda yy: 3.0 * yy ** 2 - 0.4
    lhs, rhs, scale = _both_sides(alpha, beta, g, dg)
    assert abs(lhs - rhs) / scale < 1e-8


@pytest.mark.parametrize("alpha,beta", [(0.0, 1.5582), (0.6, 1.4)])
def test_shear_shift_identity_trig_profile(alpha, beta):
    """A non-polynomial shift, to rule out a basis-specific coincidence."""
    g = lambda yy: np.sin(2.3 * yy) + 0.7 * np.cos(1.1 * yy)
    dg = lambda yy: 2.3 * np.cos(2.3 * yy) - 0.77 * np.sin(1.1 * yy)
    lhs, rhs, scale = _both_sides(alpha, beta, g, dg)
    assert abs(lhs - rhs) / scale < 1e-8


def test_uniform_shift_reproduces_production():
    """g = y (i.e. W = U) must give <W,F2(u)> = int u v dV exactly."""
    alpha, beta = 0.5, 1.3
    sol = ro.solve_mode(alpha, beta, n_max=36)
    y = np.linspace(-1.0, 1.0, 4001)
    u, du, v, dv, w, dw = ro.mode_profiles_and_derivatives(sol, y)
    adv = 0.5 * np.real(v * np.conj(du)) + 0.5 * beta * np.imag(w * np.conj(u))
    lhs = -np.trapezoid(y * adv, y)
    prod = np.trapezoid(0.5 * np.real(u * np.conj(v)), y)   # int u v, g' = 1
    assert abs(lhs - prod) / abs(prod) < 1e-8
