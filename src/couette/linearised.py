"""The linearised Navier-Stokes operator for plane Couette flow.

This is the operator ``F1`` of ``docs/03_quartic_lyapunov.md``, restricted to one
Fourier mode and expressed in the same Galerkin bases as the Reynolds-Orr solver.
It is what the still-to-be-built quadratic form ``Q`` of Sec. 3.8 has to be a
Lyapunov function for.

In Orr-Sommerfeld/Squire variables, with ``Delta_k = D^2 - k^2`` and base flow
``U(y) = y`` (so ``U'' = 0``):

    d/dt (Delta_k v) = [ -i alpha y Delta_k + Re^{-1} Delta_k^2 ] v
    d/dt eta         = -i beta v + [ -i alpha y + Re^{-1} Delta_k ] eta

Testing the first against ``phi_m`` and the second against ``psi_m`` and using
``phi = D phi = 0``, ``psi = 0`` at the walls to move every derivative onto the
test function gives the generator assembled below.

Self-check
----------
The generator is not validated against remembered numbers but against the
matrices the Reynolds-Orr solver already builds.  Writing ``M`` for the energy
matrix, ``A`` for the production matrix and ``B`` for the dissipation matrix,
the exact energy identity ``dE/dt = P - Re^{-1} D`` is equivalent to the matrix
identity

    M L + L^H M  =  2 ( A - B / Re )                                    (LIN)

which ``verify_energy_identity`` checks to machine precision.  Any sign or
factor error in the assembly breaks it.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .basis import make_bases
from .reynolds_orr import build_matrices


def build_generator(alpha: float, beta: float, re: float, n_max: int,
                    n_quad: int | None = None):
    """Assemble ``(L, M)``: the generator ``dx/dt = L x`` and the energy matrix.

    The state is ``x = [v-coefficients, eta-coefficients]``; ``M`` is such that
    ``int |u-hat|^2 dy = x^H M x / k^2``.
    """
    k2 = alpha * alpha + beta * beta
    if k2 <= 0:
        raise ValueError("k^2 must be positive")

    vb, eb, nodes, _ = make_bases(n_max, n_quad=n_quad, max_deriv=2)
    nv, ne = vb.size, eb.size

    # --- pieces shared with the Reynolds-Orr assembly ---
    Lv = vb.gram(1, 1) + k2 * vb.gram(0, 0)          # -int phi_m Delta_k phi_n
    Leta = eb.gram(1, 1) + k2 * eb.gram(0, 0)        # -int psi_m Delta_k psi_n
    Bvv = vb.gram(2, 2) + 2 * k2 * vb.gram(1, 1) + k2 * k2 * vb.gram(0, 0)
    Meta = eb.gram(0, 0)
    C = eb.cross(vb, 0, 0)                           # C_mn = int psi_m phi_n

    # --- base-flow advection, int y phi_m Delta_k phi_n, moved onto phi_m ---
    #   int y phi_m D^2 phi_n = -int phi_m D phi_n - int y D phi_m D phi_n
    adv_v = -(vb.gram(0, 1) + vb.gram(1, 1, weight=nodes)
              + k2 * vb.gram(0, 0, weight=nodes))    # = int y phi_m Delta_k phi_n

    Lmat = np.zeros((nv + ne, nv + ne), dtype=complex)

    # OS block:  -Lv * dv/dt = [ -i alpha adv_v + Re^{-1} Bvv ] v
    rhs_v = -1j * alpha * adv_v + Bvv / re
    Lmat[:nv, :nv] = -sla.solve(Lv, rhs_v, assume_a="pos")

    # Squire block:  Meta * deta/dt = -i beta C v - i alpha (y-weighted) eta - Leta eta / Re
    rhs_eta_v = -1j * beta * C
    rhs_eta_eta = -1j * alpha * eb.gram(0, 0, weight=nodes) - Leta / re
    Lmat[nv:, :nv] = sla.solve(Meta, rhs_eta_v, assume_a="pos")
    Lmat[nv:, nv:] = sla.solve(Meta, rhs_eta_eta, assume_a="pos")

    M = np.zeros((nv + ne, nv + ne), dtype=complex)
    M[:nv, :nv] = Lv
    M[nv:, nv:] = Meta
    return Lmat, M


def verify_energy_identity(alpha: float, beta: float, re: float,
                           n_max: int = 30) -> float:
    """Relative residual of ``M L + L^H M = 2 (A - B/Re)``.

    Returns a number that should be at machine-precision level.
    """
    Lmat, M = build_generator(alpha, beta, re, n_max)
    A, B, _, _ = build_matrices(alpha, beta, n_max)
    lhs = M @ Lmat + Lmat.conj().T @ M
    rhs = 2.0 * (A - B / re)
    return float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))


def spectrum(alpha: float, beta: float, re: float, n_max: int = 40) -> np.ndarray:
    """Eigenvalues of the linearised operator, sorted by decreasing real part."""
    Lmat, _ = build_generator(alpha, beta, re, n_max)
    ev = np.linalg.eigvals(Lmat)
    return ev[np.argsort(-ev.real)]


def squire_eigenvalues_exact(beta: float, re: float, n: int = 6) -> np.ndarray:
    """Exact Squire eigenvalues at ``alpha = 0``: ``-(beta^2 + (m pi/2)^2)/Re``.

    At ``alpha = 0`` the Squire operator is just ``Re^{-1}(D^2 - beta^2)`` with
    Dirichlet conditions on ``[-1, 1]``, whose eigenfunctions are
    ``sin(m pi (y+1) / 2)``.
    """
    m = np.arange(1, n + 1)
    return -(beta ** 2 + (m * np.pi / 2.0) ** 2) / re


def stokes_eigenvalues_exact(beta: float, re: float, n_max: int = 60,
                             n: int = 6) -> np.ndarray:
    """Clamped Stokes ('Orr-Sommerfeld at alpha=0') eigenvalues, -lambda/Re.

    At ``alpha = 0`` the v-equation is ``Lv dv/dt = -Bvv v / Re``, so the decay
    rates are ``-lambda_j / Re`` with ``lambda_j`` the eigenvalues of the
    symmetric definite pencil ``(Bvv, Lv)``.  Solved here in the Galerkin basis,
    independently of ``build_generator``.
    """
    vb, _, _, _ = make_bases(n_max, max_deriv=2)
    k2 = beta * beta
    Lv = vb.gram(1, 1) + k2 * vb.gram(0, 0)
    Bvv = vb.gram(2, 2) + 2 * k2 * vb.gram(1, 1) + k2 * k2 * vb.gram(0, 0)
    lam = sla.eigh(Bvv, Lv, eigvals_only=True)
    return -np.sort(lam)[:n] / re
