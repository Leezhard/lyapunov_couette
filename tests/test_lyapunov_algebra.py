"""Algebraic facts underpinning the quartic ansatz, docs/03 Eqs. (3.4) and (3.15).

These are statements about the *form* V = E^2 + 2E<W,u> + Q[u], independent of
the fluid dynamics, so they can be checked exactly in a finite-dimensional
surrogate where E = |u|^2/2 and Q, W are an arbitrary matrix and vector.
"""

from __future__ import annotations

import numpy as np
import pytest

rng = np.random.default_rng(20250821)


def V(u, W, Qop):
    E = 0.5 * u @ u
    return E ** 2 + 2 * E * (W @ u) + u @ Qop @ u


def test_positivity_criterion_is_sufficient():
    """Q > W (x) W  =>  V > 0 everywhere."""
    n = 5
    for _ in range(40):
        W = rng.normal(size=n)
        # Qop = W W^T + (something strictly positive definite)
        M = rng.normal(size=(n, n))
        Qop = np.outer(W, W) + M @ M.T + 0.1 * np.eye(n)
        for _ in range(200):
            u = rng.normal(size=n) * 10 ** rng.uniform(-3, 2)
            assert V(u, W, Qop) > 0


def test_positivity_criterion_is_necessary():
    """If Q - W (x) W has a negative direction, V dips below zero -- and does so
    at exactly the amplitude that annihilates the completed square."""
    n = 4
    for _ in range(40):
        W = rng.normal(size=n)
        d = rng.normal(size=n)
        if W @ d > 0:
            d = -d                       # ensure <W, d> < 0
        # Build Qop positive definite but with Q[d] < <W,d>^2 along d.
        basis = np.linalg.qr(np.column_stack([d] + [rng.normal(size=n)
                                                    for _ in range(n - 1)]))[0]
        deficit = 0.5 * (W @ (d / np.linalg.norm(d))) ** 2
        diag = np.diag([deficit] + [3.0] * (n - 1))
        Qop = basis @ diag @ basis.T
        dhat = d / np.linalg.norm(d)
        assert dhat @ Qop @ dhat < (W @ dhat) ** 2      # criterion violated
        # predicted amplitude: s = -<W,dhat> / E[dhat],  E[dhat] = 1/2
        s = -(W @ dhat) / 0.5
        assert s > 0
        assert V(s * dhat, W, Qop) < 0


@pytest.mark.parametrize("a2,a3,a4,expect", [
    (-1.0, 0.0, -1.0, True),
    (-1.0, -5.0, -1.0, True),
    (-1.0, 1.0, -1.0, True),        # a3^2 = 1 < 4
    (-1.0, 2.0, -1.0, False),       # a3^2 = 4 = 4 a2 a4 -> tangency, excluded
    (-1.0, 3.0, -1.0, False),
    (-1.0, -1.0, 1.0, False),       # a4 > 0
    (0.0, -1.0, -1.0, False),       # a2 = 0: no uniform margin
])
def test_negativity_criterion(a2, a3, a4, expect):
    """(3.15) must be exactly 'q < 0 on the closed half-line [0, inf)'."""
    crit = (a2 < 0) and (a4 <= 0) and (a3 <= 0 or a3 ** 2 < 4 * a2 * a4)
    assert crit is expect
    s = np.concatenate([[0.0], np.logspace(-4, 4, 20001)])
    q = a2 + a3 * s + a4 * s ** 2
    assert crit == bool(np.all(q < 0))


def test_no_quintic_term_requires_energy_neutrality():
    """d/dt(E^2) has no degree-5 part exactly because <u, F2(u)> = 0."""
    n = 6
    for _ in range(50):
        u = rng.normal(size=n)
        # any antisymmetric bilinear map is energy neutral in this surrogate
        A = rng.normal(size=(n, n, n))
        A = A - np.transpose(A, (2, 1, 0))          # F2_i = A_ijk u_j u_k
        F2 = np.einsum("ijk,j,k->i", A, u, u)
        assert abs(u @ F2) < 1e-10 * max(1.0, np.linalg.norm(F2))
