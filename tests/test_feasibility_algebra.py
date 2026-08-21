"""The rho-optimisation behind the feasibility criterion, docs/03 Eq. (3.18').

With Q = tau I + rho Qhat the constants scale as c2 = rho chat2 and
K = rho Khat, so the master inequality b^2 < 4 c2 c4 becomes

    (b0 + rho Bhat)^2 < 4 rho chat2 c4 ,     Bhat = (dcubic/dK) * Khat.

Maximising the slack over rho > 0 gives the threshold  chat2 c4 = b0 Bhat.
This got a factor of 4 wrong once; the test pins it down.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

LAMBDA_1 = np.pi ** 2 / 4


def test_threshold_symbolically():
    rho, b0, Bh, c2h, c4 = sp.symbols("rho b0 Bhat c2hat c4", positive=True)
    F = 4 * rho * c2h * c4 - (b0 + rho * Bh) ** 2
    rho_star = sp.solve(sp.diff(F, rho), rho)[0]
    Fmax = sp.simplify(F.subs(rho, rho_star))
    # F_max > 0  <=>  c2hat * c4 - b0 * Bhat > 0
    cond = sp.simplify(Fmax * Bh ** 2 / (4 * c2h * c4))
    assert sp.simplify(cond - (c2h * c4 - b0 * Bh)) == 0


@pytest.mark.parametrize("seed", range(8))
def test_threshold_numerically(seed):
    rng = np.random.default_rng(seed)
    grid = np.logspace(-8, 8, 100001)
    for _ in range(150):
        b0, Bh, c4 = 10 ** rng.uniform(-4, 1, 3)
        thr = b0 * Bh / c4                       # predicted threshold on chat2
        for c2h, expected in ((thr * 1.05, True), (thr * 0.95, False)):
            slack = 4 * grid * c2h * c4 - (b0 + grid * Bh) ** 2
            assert bool(np.any(slack > 0)) is expected


def test_required_ratio_formula():
    """With Bhat = 2 Khat / lambda_1, the requirement is 2 b0 / (lambda_1 c4)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        b0, c4, Khat = 10 ** rng.uniform(-4, 0, 3)
        Bhat = 2.0 * Khat / LAMBDA_1
        thr_c2 = b0 * Bhat / c4                  # chat2 threshold
        assert np.isclose(thr_c2 / Khat, 2.0 * b0 / (LAMBDA_1 * c4))


def test_master_inequality_gives_negative_derivative():
    """b^2 < 4 c2 c4 must actually make -c4 s^2 + b s - c2 < 0 on [0, inf)."""
    rng = np.random.default_rng(7)
    s = np.concatenate([[0.0], np.logspace(-6, 6, 60001)])
    for _ in range(400):
        c2, c4 = 10 ** rng.uniform(-3, 1, 2)
        b = np.sqrt(4 * c2 * c4) * rng.uniform(0.0, 1.6)
        q = -c4 * s ** 2 + b * s - c2
        assert bool(np.all(q < 0)) == bool(b ** 2 < 4 * c2 * c4)
