"""The Reynolds-Orr energy-stability eigenvalue problem for plane Couette flow.

Conventions
-----------
Channel ``y in [-1, 1]`` (half-gap ``h = 1``), walls moving at ``+-U``, base flow
``U(y) = y``, Reynolds number ``Re = U h / nu``.  Perturbations are periodic in
``x`` (streamwise) and ``z`` (spanwise) and vanish on the walls.

The perturbation energy ``E = (1/2) \\int |u|^2 dV`` obeys the *exact* identity

    dE/dt = - \\int Sigma(y) u v dV - Re^{-1} \\int |grad u|^2 dV,     Sigma = U'(y) = 1

(the advective nonlinearity and the pressure do no net work in the energy norm).
Hence ``dE/dt < 0`` for every non-zero perturbation iff ``Re < Re_E`` where

    1 / Re_E = max_{u} ( - \\int Sigma u v dV ) / ( \\int |grad u|^2 dV )         (RO)

over divergence-free fields vanishing on the walls.  This module solves (RO).

A general weight ``Sigma(y)`` is carried through everywhere.  With ``Sigma = 1``
this is the classical problem whose answer is ``Re_E ~ 20.66``.  With
``Sigma(y) = 1 - sigma g'(y)`` it is the *shear-shifted* problem that controls
the quartic term of the Lyapunov functional built in ``docs/03_quartic_lyapunov.md``.

Discretisation
--------------
For one Fourier mode ``exp(i(alpha x + beta z))`` the divergence-free constraint
is eliminated exactly by using the Orr-Sommerfeld/Squire variables: the
wall-normal velocity ``v`` and the wall-normal vorticity
``eta = i beta u - i alpha w``.  With ``k^2 = alpha^2 + beta^2 > 0``,

    u = (i / k^2) (alpha D v - beta eta),      w = (i / k^2) (beta D v + alpha eta)

is divergence-free for *any* ``(v, eta)``, and the boundary conditions become
``v = Dv = 0`` and ``eta = 0``.  In these variables

    k^2 \\int |grad u|^2 dy = \\int |D^2 v|^2 + 2k^2 |Dv|^2 + k^4 |v|^2
                                  + |D eta|^2 + k^2 |eta|^2  dy
    k^2 ( - \\int Sigma u v dy ) = \\int Sigma Im[ (alpha D v - beta eta) conj(v) ] dy

Both sides carry the same factor ``1/k^2``, so the Rayleigh quotient is the
ratio of the two right-hand sides.  Each is discretised as an exactly Hermitian
matrix in the Galerkin bases of ``basis.py`` and the problem is solved as a
Hermitian-definite generalised eigenproblem, giving real eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt

from .basis import make_bases


@dataclass
class ROSolution:
    """Result of one Reynolds-Orr solve at a fixed wavenumber pair."""

    alpha: float
    beta: float
    n_max: int
    growth_rate: float          # lambda_max = 1 / Re_E(alpha, beta)
    re_e: float                 # Re_E(alpha, beta) = 1 / lambda_max
    v_coeffs: np.ndarray        # complex Galerkin coefficients of v-hat
    eta_coeffs: np.ndarray      # complex Galerkin coefficients of eta-hat
    spectrum: np.ndarray        # full sorted eigenvalue list (descending)


def _shear_at(sigma, nodes: np.ndarray) -> np.ndarray:
    """Evaluate the shear weight Sigma(y) at the quadrature nodes."""
    if sigma is None:
        return np.ones_like(nodes)
    if callable(sigma):
        return np.asarray(sigma(nodes), dtype=float)
    return np.full_like(nodes, float(sigma))


def build_matrices(alpha: float, beta: float, n_max: int,
                   sigma=None, n_quad: int | None = None):
    """Assemble the Hermitian pair ``(A, B)`` for the Reynolds-Orr quotient.

    ``A`` is ``k^2 *`` the production form, ``B`` is ``k^2 *`` the dissipation
    form, both in the ordered basis ``[v-coefficients, eta-coefficients]``.
    ``max eig(A, B) = 1 / Re_E(alpha, beta)``.
    """
    k2 = alpha * alpha + beta * beta
    if k2 <= 0:
        raise ValueError("k^2 = alpha^2 + beta^2 must be positive; the "
                         "(0, 0) mode has v == 0 and no production.")

    vb, eb, nodes, _ = make_bases(n_max, n_quad=n_quad, max_deriv=2)
    shear = _shear_at(sigma, nodes)
    nv, ne = vb.size, eb.size

    # ---- dissipation (block diagonal, real symmetric positive definite) ----
    B = np.zeros((nv + ne, nv + ne), dtype=complex)
    B[:nv, :nv] = vb.gram(2, 2) + 2.0 * k2 * vb.gram(1, 1) + k2 * k2 * vb.gram(0, 0)
    B[nv:, nv:] = eb.gram(1, 1) + k2 * eb.gram(0, 0)

    # ---- production (Hermitian, indefinite) ----
    A = np.zeros((nv + ne, nv + ne), dtype=complex)
    if alpha != 0.0:
        # alpha * Im[ D v conj(v) ] -> (alpha / 2i) (G - G^T),  G_mn = int Sigma phi_m D phi_n
        G = vb.gram(0, 1, weight=shear)
        A[:nv, :nv] = -0.5j * alpha * (G - G.T)
    if beta != 0.0:
        # -beta * Im[ eta conj(v) ]
        M = vb.cross(eb, 0, 0, weight=shear)      # M_mn = int Sigma phi_m psi_n
        A[:nv, nv:] = 0.5j * beta * M
        A[nv:, :nv] = -0.5j * beta * M.T

    # Kill any round-off asymmetry so eigh sees exactly Hermitian input.
    A = 0.5 * (A + A.conj().T)
    B = 0.5 * (B + B.conj().T)
    return A, B, vb, eb


def solve_mode(alpha: float, beta: float, n_max: int = 40,
               sigma=None, n_quad: int | None = None) -> ROSolution:
    """Solve the Reynolds-Orr problem at one wavenumber pair."""
    A, B, _, _ = build_matrices(alpha, beta, n_max, sigma=sigma, n_quad=n_quad)
    vals, vecs = sla.eigh(A, B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    lam = float(vals[0])
    nv = (A.shape[0]) // 2
    vec = vecs[:, 0]
    return ROSolution(
        alpha=alpha, beta=beta, n_max=n_max,
        growth_rate=lam,
        re_e=(np.inf if lam <= 0 else 1.0 / lam),
        v_coeffs=vec[:nv], eta_coeffs=vec[nv:],
        spectrum=vals,
    )


def growth(alpha: float, beta: float, n_max: int = 40, sigma=None) -> float:
    """Largest Reynolds-Orr eigenvalue ``1 / Re_E(alpha, beta)``."""
    A, B, _, _ = build_matrices(alpha, beta, n_max, sigma=sigma)
    return float(sla.eigh(A, B, eigvals_only=True, subset_by_index=[A.shape[0] - 1,
                                                                   A.shape[0] - 1])[0])


def optimise_beta(alpha: float, n_max: int = 40, sigma=None,
                  bracket: tuple[float, float, float] = (0.8, 1.6, 3.0)) -> tuple[float, float]:
    """Maximise the Reynolds-Orr eigenvalue over ``beta`` at fixed ``alpha``.

    Returns ``(beta_opt, lambda_max)``.
    """
    lo, mid, hi = bracket
    res = sopt.minimize_scalar(lambda b: -growth(alpha, b, n_max, sigma),
                               bracket=None, bounds=(lo, hi), method="bounded",
                               options={"xatol": 1e-10})
    return float(res.x), float(-res.fun)


def optimise_wavenumbers(n_max: int = 40, sigma=None,
                         alpha_bounds: tuple[float, float] = (0.0, 2.0),
                         beta_bounds: tuple[float, float] = (0.5, 3.0)
                         ) -> tuple[float, float, float]:
    """Maximise over both wavenumbers.  Returns ``(alpha, beta, lambda_max)``."""
    def neg(p):
        a, b = p
        if not (alpha_bounds[0] <= a <= alpha_bounds[1]):
            return 1e3
        if not (beta_bounds[0] <= b <= beta_bounds[1]):
            return 1e3
        return -growth(a, b, n_max, sigma)

    best = None
    for a0 in np.linspace(alpha_bounds[0], alpha_bounds[1], 5):
        for b0 in np.linspace(beta_bounds[0], beta_bounds[1], 6):
            r = sopt.minimize(neg, np.array([a0, b0]), method="Nelder-Mead",
                              options={"xatol": 1e-10, "fatol": 1e-14,
                                       "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    a, b = best.x
    return float(a), float(b), float(-best.fun)


def critical_reynolds(n_max: int = 40, sigma=None) -> dict:
    """Global energy-stability threshold: ``Re_E = 1 / max_{alpha, beta} lambda``.

    Returns a dict with the threshold and the maximising wavenumbers.
    """
    a, b, lam = optimise_wavenumbers(n_max=n_max, sigma=sigma)
    return {"Re_E": 1.0 / lam, "alpha": a, "beta": b, "lambda_max": lam,
            "n_max": n_max}


# --------------------------------------------------------------------------
# Reconstruction of the physical velocity field of a critical mode
# --------------------------------------------------------------------------

def mode_profiles(sol: ROSolution, y: np.ndarray, n_max: int | None = None):
    """Return ``(u_hat, v_hat, w_hat)`` of a solved mode on the grid ``y``.

    The returned complex profiles satisfy the divergence-free condition
    ``i alpha u + D v + i beta w = 0`` by construction.
    """
    u, _, v, _, w, _ = mode_profiles_and_derivatives(sol, y, n_max=n_max)
    return u, v, w


def mode_profiles_and_derivatives(sol: ROSolution, y: np.ndarray,
                                  n_max: int | None = None):
    """``(u, Du, v, Dv, w, Dw)`` of a solved mode, all differentiated *exactly*.

    Derivatives are taken in the Galerkin basis rather than by finite
    differencing the sampled profiles, so the divergence identity and the
    dissipation integral hold to round-off instead of to the accuracy of a
    difference stencil.
    """
    n_max = sol.n_max if n_max is None else n_max
    vb, eb, _, _ = make_bases(n_max, max_deriv=2)
    k2 = sol.alpha ** 2 + sol.beta ** 2

    v = vb.evaluate(sol.v_coeffs, y, deriv=0)
    dv = vb.evaluate(sol.v_coeffs, y, deriv=1)
    d2v = vb.evaluate(sol.v_coeffs, y, deriv=2)
    eta = eb.evaluate(sol.eta_coeffs, y, deriv=0)
    deta = eb.evaluate(sol.eta_coeffs, y, deriv=1)

    u = 1j / k2 * (sol.alpha * dv - sol.beta * eta)
    w = 1j / k2 * (sol.beta * dv + sol.alpha * eta)
    du = 1j / k2 * (sol.alpha * d2v - sol.beta * deta)
    dw = 1j / k2 * (sol.beta * d2v + sol.alpha * deta)
    return u, du, v, dv, w, dw


def dissipation(sol: ROSolution) -> float:
    """``int (|Du|^2 + k^2 |u|^2) dy`` of a solved mode, exactly.

    Read straight off the Galerkin dissipation matrix, which *is* ``k^2`` times
    this integral, so no quadrature of the sampled profiles is involved.
    """
    A, B, _, _ = build_matrices(sol.alpha, sol.beta, sol.n_max)
    k2 = sol.alpha ** 2 + sol.beta ** 2
    x = np.concatenate([sol.v_coeffs, sol.eta_coeffs])
    return float(np.real(x.conj() @ B @ x)) / k2


def normalise_mode(sol: ROSolution, y: np.ndarray):
    """Profiles rescaled to unit energy and rotated to a canonical phase."""
    u, v, w = mode_profiles(sol, y)
    # Fix the global complex phase by making the largest |u| entry real positive.
    idx = int(np.argmax(np.abs(u)))
    phase = np.exp(-1j * np.angle(u[idx])) if abs(u[idx]) > 0 else 1.0
    u, v, w = u * phase, v * phase, w * phase
    norm = np.sqrt(np.trapezoid(np.abs(u) ** 2 + np.abs(v) ** 2 + np.abs(w) ** 2, y))
    if norm > 0:
        u, v, w = u / norm, v / norm, w / norm
    return u, v, w
