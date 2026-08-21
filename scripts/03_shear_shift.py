"""The shear-shifted Reynolds-Orr problem: how much Re can a small W buy?

The quartic part of the Lyapunov functional V = E^2 + 2E<W,u> + Q is exactly
2E times the Reynolds-Orr functional with the base-flow strain S^U replaced by
S^U - S^W (docs/03).  For W = (g(y), 0, 0) that means the weight

    Sigma(y) = 1 - g'(y),        g(+-1) = 0  =>  mean of Sigma pinned to 1.

This script quantifies the trade-off: what does it cost, in norms of g, to push
Re_E[Sigma] above a target Re?

Writes results/03_shear_shift.json and results/03_optimal_shift.csv.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import reynolds_orr as ro       # noqa: E402
from couette import shear_shift as ss        # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

BETA_C = 1.5581617774
RE_E = 20.662537217781


def sigma_from_samples(y, dg):
    """Callable Sigma(y) = 1 - g'(y) built from sampled g'."""
    def _s(yy):
        return 1.0 - np.interp(np.asarray(yy, dtype=float), y, dg)
    return _s


def wall_layer_sigma(delta: float, sharp: float = 6.0):
    """Mean-preserving weight concentrated in wall layers of thickness ``delta``.

    ``Sigma ~ 1/delta`` for ``|y| > 1 - delta`` and ~0 in the core, smoothed by a
    tanh of width ``delta/sharp`` so that the Gauss-Legendre quadrature of the
    solver converges, and renormalised so that the mean over [-1, 1] is exactly 1.
    """
    w = delta / sharp
    yy = np.linspace(-1.0, 1.0, 40001)
    prof = 0.5 * (1.0 + np.tanh((np.abs(yy) - (1.0 - delta)) / w))
    mean = np.trapezoid(prof, yy) / 2.0

    def _s(y):
        y = np.asarray(y, dtype=float)
        return 0.5 * (1.0 + np.tanh((np.abs(y) - (1.0 - delta)) / w)) / mean
    return _s


def check_alpha0_still_critical(sigfun, n_max=32):
    """Confirm the shifted problem is still critical at alpha = 0."""
    worst, wa, wb = np.inf, None, None
    for a in (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
        b, lam = ro.optimise_beta(a, n_max=n_max, sigma=sigfun, bracket=(0.4, 1.6, 4.0))
        if 1.0 / lam < worst:
            worst, wa, wb = 1.0 / lam, a, b
    return worst, wa, wb


def main() -> None:
    out: dict = {}

    print("=" * 74)
    print("(a) The optimal shift direction, derived from the critical mode")
    print("=" * 74)

    y, g_hat, eff, sol = ss.optimal_shift_L2(0.0, BETA_C)
    dg_hat = np.gradient(g_hat, y, edge_order=2)
    d2g_hat = np.gradient(dg_hat, y, edge_order=2)
    n_g2 = float(np.sqrt(np.trapezoid(d2g_hat ** 2, y)))

    print("\n  Reynolds-stress density Phi(y) = -Re[u conj(v)] of the critical mode,")
    print("  normalised to unit dissipation.  First-order perturbation theory gives")
    print("      delta lambda = - int g'(y) Phi(y) dy = + int g(y) Phi'(y) dy,")
    print("  so the L2-cheapest shift is g ~ -Phi'.")
    print(f"\n     efficiency  ||Phi'||_L2      = {eff:.8f}")
    print(f"     ||g''||_L2 of the unit-norm optimal g = {n_g2:.6f}")
    print(f"     g is odd in y and vanishes at the walls: "
          f"g(+-1) = {g_hat[0]:.2e}, {g_hat[-1]:.2e}")
    out["efficiency_L2"] = eff
    out["g2norm_of_unit_g"] = n_g2

    np.savetxt(RESULTS / "03_optimal_shift.csv",
               np.column_stack([y, g_hat, dg_hat]), delimiter=",",
               header="y,g_hat,dg_hat", comments="")

    print("\n  minimum ||g||_L2 to certify a target Re (leading order):")
    print("       Re      ||g||_L2 min     ||W||=sqrt(LxLz)||g||  (per unit sqrt(LxLz))")
    costs = []
    for tgt in (20.7, 20.75, 21.0, 21.5, 22.0, 25.0, 30.0):
        c = ss.min_cost_for_target(tgt, RE_E, eff)
        costs.append({"Re": tgt, "min_g_L2": c})
        print(f"     {tgt:6.2f}   {c:.6e}")
    out["min_cost"] = costs

    print()
    print("=" * 74)
    print("(b) Validation: first-order prediction vs full nonlinear solve")
    print("=" * 74)
    print("\n     ||g||_L2     predicted Re_E[Sigma]   actual Re_E[Sigma]   rel.err")
    val = []
    for amp in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        dg = amp * dg_hat
        sigfun = sigma_from_samples(y, dg)
        pred = 1.0 / (sol.growth_rate - amp * eff)
        b, lam = ro.optimise_beta(0.0, n_max=36, sigma=sigfun, bracket=(0.8, 1.56, 3.0))
        act = 1.0 / lam
        val.append({"amp": amp, "pred": pred, "actual": act})
        print(f"     {amp:.1e}   {pred:18.8f}   {act:17.8f}   "
              f"{abs(act - pred) / act:.2e}")
    out["validation"] = val

    print()
    print("=" * 74)
    print("(c) Is alpha = 0 still the critical wavenumber after the shift?")
    print("=" * 74)
    print("\n     ||g||_L2    Re_E[Sigma]   attained at (alpha, beta)")
    crit = []
    for amp in (1e-3, 1e-2, 5e-2, 1e-1, 2e-1):
        sigfun = sigma_from_samples(y, amp * dg_hat)
        worst, wa, wb = check_alpha0_still_critical(sigfun)
        crit.append({"amp": amp, "Re_E": worst, "alpha": wa, "beta": wb})
        print(f"     {amp:.1e}   {worst:11.6f}   ({wa:.2f}, {wb:.5f})")
    out["criticality"] = crit

    print()
    print("=" * 74)
    print("(d) Poincare / Stokes constant lambda_1 = min D[u] / ||u||^2")
    print("=" * 74)
    print("\n  The infimum over all divergence-free no-slip fields is approached")
    print("  by a streamwise-uniform streak u = cos(pi y / 2) as k -> 0, giving")
    print(f"     lambda_1 = pi^2 / 4 = {np.pi ** 2 / 4:.10f}")
    lam1 = []
    for k in (0.05, 0.2, 0.5, 1.0, BETA_C, 3.0):
        A, B, vb, eb = ro.build_matrices(0.0, k, 30)
        # min of dissipation / energy over the (v, eta) space at this wavenumber
        import scipy.linalg as sla
        nv = vb.size
        M = np.zeros_like(B)
        M[:nv, :nv] = vb.gram(1, 1) + k * k * vb.gram(0, 0)
        M[nv:, nv:] = eb.gram(0, 0)
        ev = sla.eigh(B, M, eigvals_only=True)
        lam1.append({"k": k, "lambda_min": float(ev[0])})
        print(f"     k = {k:6.3f}   min D/||u||^2 = {ev[0]:.8f}")
    out["stokes"] = {"lambda_1_infimum": np.pi ** 2 / 4, "by_wavenumber": lam1}

    print()
    print("=" * 74)
    print("(e) Mean-preservation is not a limitation: Re_E[Sigma] is unbounded")
    print("=" * 74)
    print("\n  Weight concentrated in wall layers of thickness delta (mean still 1),")
    print("  leaving the core shear-free.  Re_E[Sigma_delta] should grow like 1/delta.")
    print(f"\n{'delta':>9} {'N':>5} {'beta_opt':>10} {'beta*delta':>11} "
          f"{'Re_E[Sig]':>12} {'Re_E*delta':>11}")
    wall = []
    for delta, N, bmax in ((0.4, 70, 8.0), (0.3, 80, 10.0), (0.2, 90, 14.0),
                           (0.15, 100, 20.0), (0.1, 120, 28.0),
                           (0.07, 140, 40.0), (0.05, 170, 56.0)):
        s = wall_layer_sigma(delta)
        b, lam = ro.optimise_beta(0.0, n_max=N, sigma=s,
                                  bracket=(0.5, 1.2 / delta, bmax))
        wall.append({"delta": delta, "beta_opt": b, "Re_E": 1.0 / lam})
        print(f"{delta:9.3f} {N:5d} {b:10.4f} {b * delta:11.4f} "
              f"{1.0 / lam:12.4f} {delta / lam:11.4f}")
    out["wall_layer"] = wall
    print("\n  Re_E[Sigma_delta] -> 29.0713 / delta.  The mean of Sigma is pinned by")
    print("  no-slip (the shifted profile Utilde = y - g must still reach +-1 at the")
    print("  walls), but that pins the shape of the shift, not the size of the gain.")

    (RESULTS / "03_shear_shift.json").write_text(json.dumps(out, indent=2))
    print(f"\n  written: {RESULTS / '03_shear_shift.json'}")


if __name__ == "__main__":
    main()
