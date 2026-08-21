"""Regression and consistency tests for the Reynolds-Orr solver."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import analytic as an           # noqa: E402
from couette import reynolds_orr as ro       # noqa: E402

RE_E = 20.6625372177809
BETA_C = 1.55816177741056


def test_critical_value_matches_analytic():
    """Galerkin and the analytic determinant must agree to ~1e-12."""
    lam = ro.growth(0.0, BETA_C, n_max=40)
    assert abs(1.0 / lam - RE_E) < 1e-9


def test_analytic_determinant_root():
    R = an.critical_R(BETA_C, guess=10.33, dps=30)
    assert abs(float(2 * R) - RE_E) < 1e-10


def test_benard_equivalence():
    """The alpha=0 problem must reproduce the textbook Rayleigh-Benard values."""
    Ra, a_c = an.rayleigh_benard_Ra_c(BETA_C, RE_E)
    assert abs(float(Ra) - 1707.762) < 1e-2
    assert abs(float(a_c) - 3.117) < 1e-3


@pytest.mark.parametrize("n_max", [12, 20, 32, 48])
def test_resolution_independence(n_max):
    """The eigenvalue is converged already at very low order."""
    lam = ro.growth(0.0, BETA_C, n_max=n_max)
    assert abs(1.0 / lam - RE_E) < 1e-9


def test_alpha_zero_is_critical():
    """Re_E(alpha) must increase with alpha."""
    prev = -np.inf
    for a in (0.0, 0.1, 0.3, 0.6, 1.0):
        _, lam = ro.optimise_beta(a, n_max=32)
        assert 1.0 / lam > prev
        prev = 1.0 / lam


def test_matrices_are_hermitian():
    for alpha, beta in ((0.0, 1.5), (0.7, 1.2), (1.3, 0.4)):
        A, B, _, _ = ro.build_matrices(alpha, beta, 20)
        assert np.allclose(A, A.conj().T, atol=1e-14)
        assert np.allclose(B, B.conj().T, atol=1e-14)
        assert np.min(np.linalg.eigvalsh(B)) > 0


def test_mode_is_divergence_free():
    """u = (i/k^2)(alpha Dv - beta eta) etc. must satisfy continuity."""
    alpha, beta = 0.6, 1.2
    sol = ro.solve_mode(alpha, beta, n_max=30)
    y = np.linspace(-0.98, 0.98, 400)
    u, _, v, dv, w, _ = ro.mode_profiles_and_derivatives(sol, y)
    div = 1j * alpha * u + dv + 1j * beta * w
    assert np.max(np.abs(div)) / np.max(np.abs(v)) < 1e-12


def test_boundary_conditions():
    sol = ro.solve_mode(0.3, 1.4, n_max=30)
    y = np.array([-1.0, 1.0])
    u, v, w = ro.mode_profiles(sol, y)
    scale = np.max(np.abs(ro.mode_profiles(sol, np.linspace(-1, 1, 100))[0]))
    assert np.max(np.abs(u)) / scale < 1e-10
    assert np.max(np.abs(v)) / scale < 1e-10
    assert np.max(np.abs(w)) / scale < 1e-10


def test_production_scales_linearly_with_uniform_shear():
    """A constant weight Sigma = s must give Re_E[s] = Re_E / s exactly."""
    for s in (0.5, 0.8, 1.3, 2.0):
        lam = ro.growth(0.0, BETA_C, n_max=32, sigma=s)
        assert abs(1.0 / lam - RE_E / s) < 1e-8


def test_orr_two_dimensional_value():
    """beta = 0 must reproduce Orr's (1907) two-dimensional value ~44.3."""
    import scipy.optimize as so
    r = so.minimize_scalar(lambda a: -ro.growth(a, 0.0, n_max=40),
                           bounds=(0.3, 3.0), method="bounded",
                           options={"xatol": 1e-12})
    re_2d = -1.0 / r.fun
    assert abs(re_2d - 44.3035467005) < 1e-6
    assert abs(r.x - 1.8933674203) < 1e-6
    assert re_2d > 2 * RE_E          # the worst case is genuinely 3D
