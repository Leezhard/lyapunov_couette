"""Evaluate the master inequality b^2 < 4 c2 c4 and the feasibility ratio (3.18).

All constants are the explicit ones of docs/03_quartic_lyapunov.md Sec. 3.8.
They come from Cauchy-Schwarz and Poincare and are certainly not sharp; the
purpose here is to produce a concrete, reproducible target for the next stage
(the construction of the finite-rank quadratic form Q), not a final answer.

Writes results/04_feasibility.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import shear_shift as ss        # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

RE_E = 20.662537217781
BETA_C = 1.5581617774
LAMBDA_1 = np.pi ** 2 / 4          # inf of D[u]/||u||^2, approached as k -> 0


def feasibility(re_target: float, safety: float, lx: float, lz: float,
                eff: float, g2_ratio: float) -> dict:
    """Constants of Sec. 3.8 for one choice of target Re and shift amplitude.

    ``safety`` is the factor by which ||g|| exceeds the leading-order minimum
    (3.14); ``g2_ratio = ||g''||/||g||`` for the optimal shift direction.
    """
    g_min = (1.0 / RE_E - 1.0 / re_target) / eff
    g_norm = safety * g_min

    # Re_E[Sigma] to leading order in the shift amplitude.
    lam_shift = 1.0 / RE_E - g_norm * eff
    re_e_sigma = 1.0 / lam_shift
    c4 = 1.0 / re_target - 1.0 / re_e_sigma

    vol_factor = np.sqrt(lx * lz)          # ||W||_{L2(Omega)} = sqrt(Lx Lz) ||g||
    W = vol_factor * g_norm
    W2 = vol_factor * g2_ratio * g_norm

    # b = b0 + rho * Bhat,  with b0 independent of the size of Qtilde
    b0 = 2.0 * (1.0 / RE_E + 1.0 / re_target) * W + W2 / (re_target * LAMBDA_1)
    Bhat_per_Khat = 2.0 / LAMBDA_1

    # 4 c4 c2 - b^2 > 0 for some rho  <=>  4 c4 chat2 > 16 Bhat b0
    required_ratio = 8.0 * b0 / (LAMBDA_1 * c4) if c4 > 0 else np.inf

    return {
        "Re_target": re_target, "safety": safety, "Lx": lx, "Lz": lz,
        "g_min_L2": g_min, "g_norm_L2": g_norm,
        "Re_E_sigma": re_e_sigma, "c4": c4,
        "norm_W": W, "norm_Wpp": W2, "b0": b0,
        "Bhat_per_Khat": Bhat_per_Khat,
        "required_c2_over_K": required_ratio,
    }


def main() -> None:
    y, g_hat, eff, sol = ss.optimal_shift_L2(0.0, BETA_C)
    d2 = np.gradient(np.gradient(g_hat, y, edge_order=2), y, edge_order=2)
    g2_ratio = float(np.sqrt(np.trapezoid(d2 ** 2, y)))   # ||g''|| for unit ||g||

    lz = 2.0 * np.pi / BETA_C
    print("=" * 78)
    print("Feasibility of the master inequality  b^2 < 4 c2 c4")
    print("=" * 78)
    print(f"\n  efficiency ||Phi'||       = {eff:.8f}")
    print(f"  ||g''||/||g|| (optimal g)  = {g2_ratio:.6f}")
    print(f"  lambda_1 = pi^2/4          = {LAMBDA_1:.8f}")
    print(f"  box: Lz = 2 pi / beta_c    = {lz:.6f}")

    print("\n  Required ratio  chat2 / Khat  for the construction to close:")
    print("  (smaller is easier; chat2 = certified linear decay rate of Qhat,")
    print("   Khat = certified nonlinear constant of Qhat)")
    print()
    print(f"     {'Re':>7} {'safety':>7} {'||g||_L2':>11} {'Re_E[Sig]':>11} "
          f"{'c4':>11} {'b0':>11} {'req c2/K':>11}")
    rows = []
    for re_t in (20.70, 20.75, 21.00, 22.00):
        for safety in (1.5, 2.0, 4.0, 10.0):
            r = feasibility(re_t, safety, lz, lz, eff, g2_ratio)
            rows.append(r)
            print(f"     {re_t:7.2f} {safety:7.1f} {r['g_norm_L2']:11.4e} "
                  f"{r['Re_E_sigma']:11.5f} {r['c4']:11.4e} {r['b0']:11.4e} "
                  f"{r['required_c2_over_K']:11.4e}")

    best = min(rows, key=lambda r: r["required_c2_over_K"])
    print(f"\n  easiest case in this sweep: Re = {best['Re_target']}, "
          f"safety = {best['safety']}  ->  chat2/Khat > {best['required_c2_over_K']:.3f}")

    print("\n  NOTE.  b0 grows like sqrt(Lx Lz) because <W,u> was bounded by")
    print("  Cauchy-Schwarz.  That loses the sign information carried by the")
    print("  regrouping 2 Edot (E + <W,u>); redoing Sec. 3.8 with that")
    print("  regrouping should remove the box dependence and lower these")
    print("  targets substantially.  The numbers below are therefore an upper")
    print("  bound on the difficulty, not an estimate of it.")

    (RESULTS / "04_feasibility.json").write_text(json.dumps(
        {"efficiency": eff, "g2_ratio": g2_ratio, "lambda_1": LAMBDA_1,
         "rows": rows}, indent=2))
    print(f"\n  written: {RESULTS / '04_feasibility.json'}")


if __name__ == "__main__":
    main()
