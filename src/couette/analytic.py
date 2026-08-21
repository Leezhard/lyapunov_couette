"""Independent, high-precision check of the streamwise-independent (alpha = 0) case.

For ``alpha = 0`` the Reynolds-Orr problem collapses to a scalar sixth-order
two-point boundary value problem which can be solved *analytically* up to the
root of a 3x3 determinant.  Because this route shares no code with the Galerkin
solver in ``reynolds_orr.py`` -- different variables, different discretisation,
different arithmetic (mpmath instead of LAPACK) -- agreement between the two is
a genuine cross-validation rather than a consistency check.

Derivation (see ``docs/01_reynolds_orr.md`` for the full version).  At
``alpha = 0`` continuity gives ``w = i v' / beta``, so the whole cross-stream
motion is carried by ``v`` alone and

    1/Re_E(beta) = max_{u, v}  ( - int u v dy )
                   / ( int (u'^2 + beta^2 u^2) dy + beta^{-2} int (v'' - beta^2 v)^2 dy )

whose Euler-Lagrange equations, with ``R = Re_E / 2``, are

    (D^2 - beta^2)   u = R v,        (D^2 - beta^2)^2 v = - R beta^2 u
    =>  (D^2 - beta^2)^3 u + R^2 beta^2 u = 0

with ``u = (D^2 - beta^2) u = D (D^2 - beta^2) u = 0`` at ``y = +-1``.  This is
the Rayleigh-Benard rigid-rigid problem in disguise, with ``Ra = 16 R^2`` and
``a = 2 beta`` after mapping ``[-1, 1]`` onto a unit-depth layer.

The critical mode is even in ``y``, so we set ``u = sum_j A_j cosh(s_j y)`` where
``s_j^2 = beta^2 + q_j`` and ``q_j`` are the three cube roots of ``-R^2 beta^2``.
Note every boundary functional below is even in ``s_j`` (``cosh`` and
``s sinh`` both are), so no branch choice for the square root is needed.
"""

from __future__ import annotations

import mpmath as mp


def _roots_q(R, beta):
    """The three roots of ``q^3 = - R^2 beta^2``."""
    mag = (R ** 2 * beta ** 2) ** (mp.mpf(1) / 3)
    return [mag * mp.e ** (1j * mp.pi * mp.mpf(k) / 3) for k in (1, 3, 5)]


def determinant(R, beta):
    """The 3x3 boundary determinant whose zero fixes ``R`` at a given ``beta``.

    Rows are the three boundary conditions at ``y = 1``:
    ``u = 0``, ``(D^2 - beta^2) u = 0``, ``D (D^2 - beta^2) u = 0``.
    """
    R = mp.mpf(R)
    beta = mp.mpf(beta)
    qs = _roots_q(R, beta)
    rows = [[], [], []]
    for q in qs:
        s = mp.sqrt(beta ** 2 + q)
        rows[0].append(mp.cosh(s))
        rows[1].append(q * mp.cosh(s))
        rows[2].append(q * s * mp.sinh(s))
    det = mp.det(mp.matrix(rows))
    # The three roots are {-m, m e^{i pi/3}, m e^{-i pi/3}}: closed under
    # conjugation, so conjugating the determinant swaps two columns and flips
    # its sign.  The determinant is therefore purely imaginary for real
    # (R, beta); its imaginary part carries the whole signal.
    return mp.im(det)


def critical_R(beta, guess=10.3, dps: int = 40):
    """Smallest positive ``R`` with ``determinant(R, beta) = 0``."""
    with mp.workdps(dps):
        return mp.findroot(lambda R: determinant(R, beta), mp.mpf(guess))


def critical_Re_E(beta, guess=10.3, dps: int = 40):
    """``Re_E(beta) = 2 R(beta)`` for streamwise-independent perturbations."""
    return 2 * critical_R(beta, guess=guess, dps=dps)


def minimise_over_beta(beta_guess=1.5585, R_guess=10.33, dps: int = 40):
    """Minimise ``Re_E(beta)`` over ``beta``.  Returns ``(beta_c, Re_E)``.

    ``Re_E(beta)`` is smooth and strictly convex near its minimum, so the
    minimiser is found by driving the central-difference derivative to zero
    with a secant iteration.  The inner root solve for ``R(beta)`` is run at
    higher precision than the outer one so that differencing does not amplify
    its truncation error.
    """
    inner = dps + 15
    with mp.workdps(inner):
        h = mp.mpf(10) ** (-(dps // 2))

        def re_of_beta(b):
            return critical_Re_E(b, guess=R_guess, dps=inner)

        def dre(b):
            return (re_of_beta(b + h) - re_of_beta(b - h)) / (2 * h)

        b0 = mp.mpf(beta_guess)
        beta_c = mp.findroot(dre, (b0, b0 + mp.mpf('1e-3')), solver="secant",
                             tol=mp.mpf(10) ** (-2 * dps // 3))
        return +beta_c, +re_of_beta(beta_c)


def rayleigh_benard_Ra_c(beta_c, Re_E):
    """The equivalent Rayleigh-Benard critical values ``(Ra_c, a_c)``.

    Mapping ``y in [-1, 1]`` onto a unit-depth layer sends ``a = 2 beta`` and
    ``Ra = 16 R^2 = 4 Re_E^2``.
    """
    return 4 * Re_E ** 2, 2 * beta_c
