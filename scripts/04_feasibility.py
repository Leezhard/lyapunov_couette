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
                eff: float, ratios: dict) -> dict:
    """Constants for one choice of target Re and shift amplitude.

    Computes the required ratio chat2/Khat by BOTH routes: the plain
    Cauchy-Schwarz bound of Sec. 3.8 and the regrouped bound of Sec. 3.9.

    ``safety`` is the factor by which ||g|| exceeds the leading-order minimum
    (3.14); ``ratios`` holds ||g'||/||g||, ||g''||/||g|| and ||g'||_inf/||g||
    for the optimal shift direction.
    """
    g_min = (1.0 / RE_E - 1.0 / re_target) / eff
    g_norm = safety * g_min

    # Re_E[Sigma] to leading order in the shift amplitude.
    lam_shift = 1.0 / RE_E - g_norm * eff
    re_e_sigma = 1.0 / lam_shift
    c4 = 1.0 / re_target - 1.0 / re_e_sigma

    vol = np.sqrt(lx * lz)                 # ||W||_{L2(Omega)} = sqrt(Lx Lz) ||g||
    W = vol * g_norm
    W1 = vol * ratios["d1"] * g_norm
    W2 = vol * ratios["d2"] * g_norm
    dg_inf = ratios["d1inf"] * g_norm

    # --- Sec. 3.8: Cauchy-Schwarz route ---
    b0_cs = (2.0 * (1.0 / RE_E + 1.0 / re_target) * W
             + W2 / (re_target * LAMBDA_1))

    # --- Sec. 3.9: regrouped route, Eq. (3.20) ---
    b0_rg = (W1 / (re_target * np.sqrt(LAMBDA_1))      # first order in the shift
             + 2.0 * c4 * W                            # second order
             + W * dg_inf / LAMBDA_1)                  # second order

    # Maximising F(rho) = 4 rho chat2 c4 - (b0 + rho Bhat)^2 over rho > 0 gives
    #     F_max > 0  <=>  chat2 c4 > b0 Bhat,
    # verified symbolically and numerically in tests/test_feasibility_algebra.py.
    # Two admissible choices of the cubic bound on 2<Qu, F2(u)>:
    #   (i)  <= (2 Khat / lambda_1)     ||u|| D,  Khat  = sup ||grad(Qtilde u)||_inf / ||u||
    #   (ii) <= (2 Kphat / sqrt(lambda_1)) ||u|| D,  Kphat = sup ||Qtilde u||_inf / ||u||
    # (ii) needs no derivative of Qtilde and is usually the better of the two.
    req_grad = lambda b0: (2.0 * b0 / (LAMBDA_1 * c4) if c4 > 0 else np.inf)
    req_sup = lambda b0: (2.0 * b0 / (np.sqrt(LAMBDA_1) * c4) if c4 > 0 else np.inf)
    req = req_grad

    return {
        "Re_target": re_target, "safety": safety, "Lx": lx, "Lz": lz,
        "g_min_L2": g_min, "g_norm_L2": g_norm,
        "Re_E_sigma": re_e_sigma, "c4": c4,
        "norm_W": W, "norm_Wp": W1, "norm_Wpp": W2,
        "b0_cauchy_schwarz": b0_cs, "b0_regrouped": b0_rg,
        "required_c2_over_K_cs": req(b0_cs),
        "required_c2_over_K_regrouped": req(b0_rg),
        "required_c2_over_Kprime_regrouped": req_sup(b0_rg),
    }


def shift_norm_ratios(y, g):
    """||g'||/||g||, ||g''||/||g|| and ||g'||_inf/||g|| for a sampled profile."""
    d1 = np.gradient(g, y, edge_order=2)
    d2 = np.gradient(d1, y, edge_order=2)
    n0 = np.sqrt(np.trapezoid(g ** 2, y))
    return {"d1": float(np.sqrt(np.trapezoid(d1 ** 2, y)) / n0),
            "d2": float(np.sqrt(np.trapezoid(d2 ** 2, y)) / n0),
            "d1inf": float(np.max(np.abs(d1)) / n0)}


def main() -> None:
    y, g_hat, eff, sol = ss.optimal_shift_L2(0.0, BETA_C)
    ratios = shift_norm_ratios(y, g_hat)

    lz = 2.0 * np.pi / BETA_C
    print("=" * 78)
    print("Feasibility of the master inequality  b^2 < 4 c2 c4")
    print("=" * 78)
    print(f"\n  efficiency ||Phi'||        = {eff:.8f}")
    print(f"  ||g'||/||g||   (optimal g)  = {ratios['d1']:.6f}")
    print(f"  ||g''||/||g||  (optimal g)  = {ratios['d2']:.6f}")
    print(f"  ||g'||_inf/||g||            = {ratios['d1inf']:.6f}")
    print(f"  lambda_1 = pi^2/4           = {LAMBDA_1:.8f}")
    print(f"  box: Lx = Lz = 2 pi/beta_c  = {lz:.6f}")

    print("\n  Required ratio  chat2 / Khat  for the construction to close")
    print("  (smaller is easier; chat2 = certified linear decay rate of Qhat,")
    print("   Khat = certified nonlinear constant of Qhat):")
    print()
    print(f"     {'Re':>7} {'safety':>7} {'||g||_L2':>11} {'Re_E[Sig]':>11} "
          f"{'c4':>11} {'req(3.8)':>10} {'req(3.9)':>10} {'gain':>6}")
    rows = []
    for re_t in (20.70, 20.75, 21.00, 22.00):
        for safety in (1.5, 2.0, 4.0, 10.0):
            r = feasibility(re_t, safety, lz, lz, eff, ratios)
            rows.append(r)
            gain = r["required_c2_over_K_cs"] / r["required_c2_over_K_regrouped"]
            print(f"     {re_t:7.2f} {safety:7.1f} {r['g_norm_L2']:11.4e} "
                  f"{r['Re_E_sigma']:11.5f} {r['c4']:11.4e} "
                  f"{r['required_c2_over_K_cs']:10.1f} "
                  f"{r['required_c2_over_K_regrouped']:10.2f} {gain:6.2f}x")

    best = min(rows, key=lambda r: r["required_c2_over_K_regrouped"])
    print(f"\n  easiest case in this sweep: Re = {best['Re_target']}, "
          f"safety = {best['safety']}")
    print(f"     ->  chat2/Khat > "
          f"{best['required_c2_over_K_regrouped']:.2f}  (regrouped bound)")

    print("\n  NOTE.  The surviving first-order term still carries a factor")
    print("  sqrt(Lx Lz), because ||ubar'|| <= ||grad u||/sqrt(Lx Lz) is")
    print("  saturated only by xz-uniform fields -- pure mean streaks, for which")
    print("  Edot = -D/Re is strongly negative anyway.  Treating the mean-streak")
    print("  component separately should remove the box dependence entirely.")

    (RESULTS / "04_feasibility.json").write_text(json.dumps(
        {"efficiency": eff, "ratios": ratios, "lambda_1": LAMBDA_1,
         "rows": rows}, indent=2))
    print(f"\n  written: {RESULTS / '04_feasibility.json'}")


if __name__ == "__main__":
    main()
