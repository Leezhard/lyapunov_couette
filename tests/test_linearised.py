"""Validation of the linearised Orr-Sommerfeld/Squire operator.

The generator is checked against the matrices the Reynolds-Orr solver already
builds, and against the exact eigenvalues available at alpha = 0.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import linearised as lin       # noqa: E402

BETA_C = 1.5581617774


@pytest.mark.parametrize("alpha,beta,re", [
    (0.0, BETA_C, 20.7), (0.4, 1.2, 100.0), (1.3, 0.7, 500.0),
    (2.0, 2.0, 50.0), (0.0, 3.0, 1000.0), (0.05, 0.05, 20.7),
])
def test_energy_identity(alpha, beta, re):
    """M L + L^H M = 2 (A - B/Re): any sign or factor slip breaks this."""
    assert lin.verify_energy_identity(alpha, beta, re, n_max=30) < 1e-12


def test_alpha_zero_spectrum_is_exact():
    """At alpha = 0 the spectrum is the exact Squire + clamped-Stokes set."""
    beta, re = BETA_C, 20.7
    ev = lin.spectrum(0.0, beta, re, n_max=50)
    exact = np.sort(np.concatenate([
        lin.squire_eigenvalues_exact(beta, re, n=6),
        lin.stokes_eigenvalues_exact(beta, re, n=6)]))[::-1]
    assert np.max(np.abs(ev[:len(exact)].real - exact)) < 1e-9
    assert np.max(np.abs(ev[:len(exact)].imag)) < 1e-10   # real at alpha = 0


@pytest.mark.parametrize("re", [20.7, 100.0, 1000.0])
def test_linearly_stable(re):
    """Plane Couette is linearly stable at every Re."""
    for a in (0.0, 0.3, 0.7, 1.5):
        for b in (0.05, 0.8, 1.5582, 3.0):
            assert lin.spectrum(a, b, re, n_max=36)[0].real < 0.0


def test_abscissa_scales_as_poincare_over_re():
    """The least-damped decay rate approaches -pi^2/(4 Re) as k -> 0."""
    for re in (100.0, 1000.0):
        s = lin.spectrum(0.0, 1e-3, re, n_max=40)[0].real
        assert abs(s * re + np.pi ** 2 / 4) < 1e-4


def test_generator_resolution_independent():
    beta, re = BETA_C, 20.7
    a = lin.spectrum(0.0, beta, re, n_max=24)[0].real
    b = lin.spectrum(0.0, beta, re, n_max=48)[0].real
    assert abs(a - b) < 1e-10
