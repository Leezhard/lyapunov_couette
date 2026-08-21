"""Reproduce the classical energy-stability threshold Re_E for plane Couette flow.

Runs three mutually independent computations and cross-checks them:

  (a) Galerkin solution of the full 3D Reynolds-Orr problem in (v, eta)
      variables, at general (alpha, beta), with a resolution study.
  (b) The analytic 3x3 boundary determinant for the reduced alpha = 0 sixth-order
      problem, evaluated in mpmath at high precision.
  (c) The implied Rayleigh-Benard rigid-rigid critical values, which must
      reproduce the independently known Ra_c = 1707.762, a_c = 3.117.

Writes results/01_ReE.json and prints a report.
"""

from __future__ import annotations

import json
import pathlib
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import analytic as an           # noqa: E402
from couette import reynolds_orr as ro       # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    out: dict = {}

    print("=" * 74)
    print("(a) Galerkin solve of the full 3D Reynolds-Orr problem")
    print("=" * 74)

    print("\n  resolution study at (alpha, beta) = (0, 1.5581617774):")
    beta_ref = 1.5581617774
    conv = []
    for n in (10, 15, 20, 25, 30, 40, 60, 80):
        s = ro.solve_mode(0.0, beta_ref, n_max=n)
        conv.append({"n_max": n, "Re_E": s.re_e, "lambda_max": s.growth_rate})
        print(f"     N = {n:3d}   lambda_max = {s.growth_rate:.15f}   "
              f"Re_E = {s.re_e:.13f}")
    out["convergence"] = conv

    print("\n  optimisation over (alpha, beta):")
    a, b, lam = ro.optimise_wavenumbers(n_max=40)
    print(f"     alpha_c = {a:.10f}")
    print(f"     beta_c  = {b:.10f}")
    print(f"     Re_E    = {1.0 / lam:.12f}")
    out["galerkin_optimum"] = {"alpha": a, "beta": b, "Re_E": 1.0 / lam}

    print("\n  Re_E(alpha, beta) slice showing alpha = 0 is optimal:")
    slice_rows = []
    for aa in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0):
        bb, ll = ro.optimise_beta(aa, n_max=40)
        slice_rows.append({"alpha": aa, "beta_opt": bb, "Re_E": 1.0 / ll})
        print(f"     alpha = {aa:4.2f}   beta_opt = {bb:.8f}   "
              f"Re_E = {1.0 / ll:.10f}")
    out["alpha_slice"] = slice_rows

    print()
    print("=" * 74)
    print("(b) Analytic determinant for the reduced alpha = 0 problem (mpmath)")
    print("=" * 74)
    bc, ree = an.minimise_over_beta(dps=25)
    print(f"\n     beta_c = {mp.nstr(bc, 18)}")
    print(f"     Re_E   = {mp.nstr(ree, 18)}")
    out["analytic"] = {"beta_c": float(bc), "Re_E": float(ree),
                       "Re_E_str": mp.nstr(ree, 18), "beta_c_str": mp.nstr(bc, 18)}

    print()
    print("=" * 74)
    print("(c) Equivalent Rayleigh-Benard rigid-rigid problem")
    print("=" * 74)
    Ra, ac = an.rayleigh_benard_Ra_c(bc, ree)
    print(f"\n     Ra_c = {mp.nstr(Ra, 12)}      (literature: 1707.762)")
    print(f"     a_c  = {mp.nstr(ac, 12)}      (literature: 3.117)")
    out["benard"] = {"Ra_c": float(Ra), "a_c": float(ac)}

    print()
    print("=" * 74)
    print("Cross-check")
    print("=" * 74)
    d_re = abs(1.0 / lam - float(ree))
    d_beta = abs(b - float(bc))
    print(f"\n     |Re_E(Galerkin) - Re_E(analytic)|  = {d_re:.3e}")
    print(f"     |beta_c(Galerkin) - beta_c(analytic)| = {d_beta:.3e}")
    print(f"\n     ==> Re_E = {float(ree):.10f}   at   "
          f"(alpha_c, beta_c) = (0, {float(bc):.10f})")
    print("\n     Note: the value often quoted in the literature as '20.65' or")
    print("     '20.7' is this same number rounded; to 15 digits it is")
    print(f"     Re_E = {mp.nstr(ree, 15)}.")
    out["agreement"] = {"delta_Re_E": d_re, "delta_beta": d_beta}

    path = RESULTS / "01_ReE.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n     written: {path}")


if __name__ == "__main__":
    main()
