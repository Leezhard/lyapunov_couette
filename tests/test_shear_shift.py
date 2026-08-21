"""Tests for the shear-shifted problem and the first-order optimal shift."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import reynolds_orr as ro       # noqa: E402
from couette import shear_shift as ss        # noqa: E402

RE_E = 20.6625372177809
BETA_C = 1.55816177741056


def _sigma_from(y, dg):
    def _s(yy):
        return 1.0 - np.interp(np.asarray(yy, dtype=float), y, dg)
    return _s


def test_optimal_shift_is_admissible():
    """g ~ -Phi' must be odd and vanish at the walls without being forced to."""
    y, g, eff, _ = ss.optimal_shift_L2(0.0, BETA_C)
    scale = np.max(np.abs(g))
    assert abs(g[0]) / scale < 1e-4 and abs(g[-1]) / scale < 1e-4
    assert np.max(np.abs(g + g[::-1])) / scale < 1e-6      # odd
    assert eff > 0


def test_shift_is_mean_preserving():
    """int g' dy = g(1) - g(-1) must vanish, i.e. the mean of Sigma stays 1."""
    y, g, _, _ = ss.optimal_shift_L2(0.0, BETA_C)
    assert abs(g[-1] - g[0]) / np.max(np.abs(g)) < 1e-10


def test_first_order_prediction():
    """delta lambda = -||g|| ||Phi'|| must match the nonlinear solve at small amp."""
    y, g_hat, eff, sol = ss.optimal_shift_L2(0.0, BETA_C)
    dg_hat = np.gradient(g_hat, y, edge_order=2)
    for amp in (1e-4, 1e-3):
        _, lam = ro.optimise_beta(0.0, n_max=32, sigma=_sigma_from(y, amp * dg_hat),
                                  bracket=(0.8, 1.56, 3.0))
        pred = sol.growth_rate - amp * eff
        assert abs(lam - pred) / pred < 1e-5


def test_mean_preserving_shift_raises_threshold():
    """The central claim of Sec. 3.5: redistribution alone raises Re_E."""
    sigma = lambda y: 1.0 - 0.1 * (1.0 - 3.0 * np.asarray(y) ** 2)
    _, lam = ro.optimise_beta(0.0, n_max=32, sigma=sigma, bracket=(0.8, 1.56, 3.0))
    assert 1.0 / lam > RE_E * 1.05


def test_shear_basis_vanishes_at_walls():
    sb = ss.ShearShiftBasis(4)
    c = np.array([1.0, -0.5, 0.25, 0.1])
    ends = sb.g(c, deriv=0, y=np.array([-1.0, 1.0]))
    assert np.max(np.abs(ends)) < 1e-12
    mid = sb.g(c, deriv=0, y=np.linspace(-1, 1, 101))
    assert np.max(np.abs(mid + mid[::-1])) < 1e-12       # odd


def test_wall_layer_weight_is_mean_preserving_and_unbounded():
    """Sec. 3.4: Re_E[Sigma] is unbounded above on the mean-preserving class."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from importlib import import_module
    mod = import_module("03_shear_shift")

    prev = None
    for delta, n_max, bmax in ((0.2, 90, 14.0), (0.1, 120, 28.0)):
        s = mod.wall_layer_sigma(delta)
        y = np.linspace(-1.0, 1.0, 20001)
        assert abs(np.trapezoid(s(y), y) / 2.0 - 1.0) < 1e-6      # mean is 1
        _, lam = ro.optimise_beta(0.0, n_max=n_max, sigma=s,
                                  bracket=(0.5, 1.2 / delta, bmax))
        re = 1.0 / lam
        assert re > RE_E                       # far above the unshifted threshold
        assert abs(re * delta - 29.07) < 0.05  # the 1/delta law, constant 29.0713
        if prev is not None:
            assert re > prev                   # monotone as delta shrinks
        prev = re
