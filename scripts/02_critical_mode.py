"""Structure of the critical Reynolds-Orr mode, and the symmetries it inherits.

The critical mode is what every later choice (the finite-rank part of Q, the
shape of W) has to be adapted to, so this script extracts it carefully:

  * the wall-normal profiles u(y), v(y), w(y) and their parities;
  * the physical interpretation (streamwise rolls + streak);
  * the spectral gap to the next Reynolds-Orr eigenvalue, which measures how
    isolated the dangerous direction is;
  * the width of the "dangerous cone" in wavenumber space, i.e. how quickly
    Re_E(alpha, beta) rises as one leaves the critical wavenumbers.

Writes results/02_critical_mode.json and results/02_mode_profiles.csv.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import reynolds_orr as ro       # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

BETA_C = 1.5581617774
N_MAX = 48


def parity(f: np.ndarray, y: np.ndarray) -> str:
    """Classify a profile sampled on a symmetric grid as even / odd / neither."""
    flipped = f[::-1]
    scale = np.max(np.abs(f))
    if scale == 0:
        return "zero"
    if np.max(np.abs(f - flipped)) / scale < 1e-8:
        return "even"
    if np.max(np.abs(f + flipped)) / scale < 1e-8:
        return "odd"
    return "neither"


def main() -> None:
    out: dict = {}
    y = np.linspace(-1.0, 1.0, 801)

    sol = ro.solve_mode(0.0, BETA_C, n_max=N_MAX)
    u, v, w = ro.normalise_mode(sol, y)

    print("=" * 74)
    print("Critical Reynolds-Orr mode of plane Couette flow")
    print("=" * 74)
    print(f"\n  (alpha_c, beta_c) = (0, {BETA_C})     Re_E = {sol.re_e:.12f}")

    # At alpha = 0 the eigenvector can be rotated so that u, v are real and
    # w = i v'/beta is purely imaginary; report the physically meaningful parts.
    ur, vr, wi = u.real, v.real, w.imag
    print("\n  amplitude check (imaginary residues should be ~0):")
    print(f"     max|Im u| / max|Re u| = {np.max(np.abs(u.imag)) / np.max(np.abs(ur)):.2e}")
    print(f"     max|Im v| / max|Re v| = {np.max(np.abs(v.imag)) / np.max(np.abs(vr)):.2e}")
    print(f"     max|Re w| / max|Im w| = {np.max(np.abs(w.real)) / np.max(np.abs(wi)):.2e}")

    print("\n  parities about the channel centreline y = 0:")
    for name, prof in (("u (streak)", ur), ("v (roll, wall-normal)", vr),
                       ("w (roll, spanwise)", wi)):
        print(f"     {name:24s} : {parity(prof, y)}")
    out["parity"] = {"u": parity(ur, y), "v": parity(vr, y), "w": parity(wi, y)}

    print("\n  amplitude ratios (unit total energy):")
    e_u = np.trapezoid(np.abs(u) ** 2, y)
    e_v = np.trapezoid(np.abs(v) ** 2, y)
    e_w = np.trapezoid(np.abs(w) ** 2, y)
    tot = e_u + e_v + e_w
    print(f"     streak energy  E_u / E = {e_u / tot:.6f}")
    print(f"     roll energy   (E_v + E_w) / E = {(e_v + e_w) / tot:.6f}")
    print(f"        of which   E_v / E = {e_v / tot:.6f},  E_w / E = {e_w / tot:.6f}")
    out["energy_split"] = {"u": e_u / tot, "v": e_v / tot, "w": e_w / tot,
                           "roll_total": (e_v + e_w) / tot}

    print(f"\n     max|u| / max|v| = {np.max(np.abs(u)) / np.max(np.abs(v)):.6f}")
    print("     (the streak is much larger than the roll that drives it -- the")
    print("      signature of the lift-up mechanism)")

    print("\n  Reynolds-Orr spectrum at the critical wavenumbers (top 6):")
    for i, lam in enumerate(sol.spectrum[:6]):
        tag = "  <- critical" if i == 0 else ""
        re_i = np.inf if lam <= 0 else 1.0 / lam
        print(f"     lambda_{i} = {lam:+.10f}   1/lambda = {re_i:12.6f}{tag}")
    gap = float(sol.spectrum[0] - sol.spectrum[1])
    print(f"\n     spectral gap lambda_0 - lambda_1 = {gap:.10f}")
    print(f"     relative gap = {gap / sol.spectrum[0]:.6f}")
    out["spectrum_top6"] = [float(x) for x in sol.spectrum[:6]]
    out["gap"] = gap

    print("\n  width of the dangerous cone in wavenumber space:")
    print("     (Re_E(alpha,beta) - Re_E) / Re_E, in percent")
    print("        beta:      " + "".join(f"{b:>10.3f}" for b in
                                          (1.25, 1.4, 1.5582, 1.7, 1.9)))
    cone = []
    for a in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
        row = f"     alpha={a:4.2f} "
        for b in (1.25, 1.4, 1.5582, 1.7, 1.9):
            r = ro.solve_mode(a, b, n_max=32).re_e
            row += f"{100 * (r - sol.re_e) / sol.re_e:>10.4f}"
            cone.append({"alpha": a, "beta": b, "Re_E": r})
        print(row)
    out["cone"] = cone

    hdr = "y,u,v,w_imag"
    data = np.column_stack([y, ur, vr, wi])
    np.savetxt(RESULTS / "02_mode_profiles.csv", data, delimiter=",",
               header=hdr, comments="")

    out["Re_E"] = sol.re_e
    out["beta_c"] = BETA_C
    out["n_max"] = N_MAX
    (RESULTS / "02_critical_mode.json").write_text(json.dumps(out, indent=2))
    print(f"\n  written: {RESULTS / '02_critical_mode.json'}")
    print(f"  written: {RESULTS / '02_mode_profiles.csv'}")


if __name__ == "__main__":
    main()
