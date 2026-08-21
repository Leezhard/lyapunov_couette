"""How large can the ratio chat2/Khat of Sec. 3.8 actually be made?

The feasibility criterion (3.18') asks for a quadratic form Q = tau I + Qtilde
whose certified linear decay rate chat2 beats its certified nonlinear constant
Khat by a factor of order 5.  This script computes both, per Fourier block.

In the Galerkin coordinates x = [v-coeffs, eta-coeffs] the L2 inner product is
x^H M y / k^2, the dissipation is x^H B x / k^2, and an operator Q acting on
fields is a matrix Q with M Q = Q^H M (self-adjointness).  The condition
Gamma_2 = 2<Q u, F1 u> <= -c2 D[u] is then the matrix inequality

    Q^H M L + L^H M Q  <=  - c2 B .                                     (LMI)

Substituting Q = tau I and using the identity M L + L^H M = 2(A - B/Re) turns
the tau-part into 2 tau (A - B/Re): the energy functional alone, which fails
exactly on the band of wavenumbers where Re_E(alpha,beta) < Re.

Best possible c2 per unit operator norm.  Whitening by M (y = M^{1/2} x) turns
(LMI) into Qw Lw + Lw^H Qw <= -c2 Bw with Qw Hermitian, and the equality case is
the Lyapunov equation whose solution is

    Qw = c2 * integral_0^inf exp(Lw^H t) Bw exp(Lw t) dt   (positive definite).

Setting c2 = 1 and reading off ||Qw|| gives the best achievable ratio
c2 / ||Q|| = 1 / ||Qw||.

The nonlinear constant.  Khat = sup ||grad(Qtilde u)||_inf / ||u||_2 and
Kphat = sup ||Qtilde u||_inf / ||u||_2 are computed by mapping the coefficient
matrix to velocity profiles and taking pointwise maxima, with Qtilde = Q - tau I
and tau chosen mid-spectrum to make ||Qtilde|| as small as possible.

Writes results/05_quadratic_form.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from couette import linearised as lin        # noqa: E402
from couette import reynolds_orr as ro       # noqa: E402
from couette.basis import make_bases         # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

BETA_C = 1.5581617774
RE = 20.7
N_MAX = 28


def whiten(alpha, beta, re, n_max):
    """Return (Lw, Bw, M, Mhalf, Minvhalf) in the M-whitened coordinates."""
    L, M = lin.build_generator(alpha, beta, re, n_max)
    A, B, _, _ = ro.build_matrices(alpha, beta, n_max)
    M = 0.5 * (M + M.conj().T)
    w, V = np.linalg.eigh(M)
    Mhalf = V @ np.diag(np.sqrt(w)) @ V.conj().T
    Minvhalf = V @ np.diag(1.0 / np.sqrt(w)) @ V.conj().T
    Lw = Mhalf @ L @ Minvhalf
    Bw = Minvhalf @ B @ Minvhalf
    return Lw, 0.5 * (Bw + Bw.conj().T), M, Mhalf, Minvhalf


def best_c2_per_norm(alpha, beta, re, n_max):
    """1 / ||Qw||, the largest c2 attainable per unit operator norm of Q.

    Also returns Qw itself (normalised to c2 = 1) for the pointwise estimates.
    """
    Lw, Bw, M, Mhalf, Minvhalf = whiten(alpha, beta, re, n_max)
    # Solve Qw Lw + Lw^H Qw = -Bw   (continuous Lyapunov, stable Lw)
    Qw = sla.solve_lyapunov(Lw.conj().T, -Bw)
    Qw = 0.5 * (Qw + Qw.conj().T)
    ev = np.linalg.eigvalsh(Qw)
    return float(1.0 / ev.max()), Qw, Mhalf, Minvhalf, ev


def rank_truncation(alpha, beta, re, n_max, lx, lz, ranks):
    """Trade-off between the certified rate c2 and the nonlinear constant K.

    The Lyapunov-equation solution Q is bounded on L2 but NOT smoothing, so
    sup ||grad(Qtilde u)||_inf / ||u||_2 diverges with resolution and the
    certificate of Sec. 3.8 cannot use it directly.  Truncating Qtilde to rank r
    makes the pointwise constants finite and resolution-independent, at the cost
    of a smaller certified rate.  Both are computed exactly here.

    For Qtilde = sum_j mu_j e_j (x) e_j with the e_j orthonormal in L2(Omega),
        ||Qtilde u||_inf   <= ( sum_j mu_j^2 ||e_j||_inf^2 )^{1/2} ||u||_2
        ||grad Qtilde u||_inf <= ( sum_j mu_j^2 ||grad e_j||_inf^2 )^{1/2} ||u||_2
    by Cauchy-Schwarz over j, which is what Kphat and Khat below evaluate.
    """
    Lw, Bw, M, Mhalf, Minvhalf = whiten(alpha, beta, re, n_max)
    Qw = sla.solve_lyapunov(Lw.conj().T, -Bw)
    Qw = 0.5 * (Qw + Qw.conj().T)
    ev_all, evec = np.linalg.eigh(Qw)
    Qw = Qw / ev_all.max()                       # normalise ||Q|| = 1
    ev_all = ev_all / ev_all.max()

    k = np.hypot(alpha, beta)
    nv = make_bases(n_max, max_deriv=2)[0].size
    ygrid = np.linspace(-1.0, 1.0, 2001)
    box = np.sqrt(2.0 / (lx * lz))               # ||e||_inf(Omega) factor

    def mode_norms(vec_w):
        """(||e||_inf, ||grad e||_inf) for a whitened unit vector, in L2(Omega)."""
        xc = Minvhalf @ vec_w
        xc = xc * k / np.linalg.norm(vec_w)      # so that ||e||_{L2(Omega)} = 1
        sol = ro.ROSolution(alpha=alpha, beta=beta, n_max=n_max, growth_rate=0.0,
                            re_e=np.inf, v_coeffs=xc[:nv], eta_coeffs=xc[nv:],
                            spectrum=np.zeros(1))
        u, du, v, dv, w, dw = ro.mode_profiles_and_derivatives(sol, ygrid)
        amp = np.abs(u) ** 2 + np.abs(v) ** 2 + np.abs(w) ** 2
        gr = (np.abs(du) ** 2 + np.abs(dv) ** 2 + np.abs(dw) ** 2
              + k ** 2 * amp)
        return box * np.sqrt(amp.max()), box * np.sqrt(gr.max())

    Bw_ih = np.linalg.inv(sla.sqrtm(Bw).real)
    order = np.argsort(-np.abs(ev_all - 0.5 * (ev_all.max() + ev_all.min())))
    out = []
    for r in ranks:
        tau = 0.5 * (ev_all.max() + ev_all.min())
        keep = order[:r]
        Qt = np.zeros_like(Qw)
        for j in keep:
            e = evec[:, j:j + 1]
            Qt += (ev_all[j] - tau) * (e @ e.conj().T)
        Qr = tau * np.eye(Qw.shape[0]) + Qt
        S = Qr @ Lw + Lw.conj().T @ Qr
        c2 = -float(np.linalg.eigvalsh(Bw_ih @ S @ Bw_ih).max())
        Kg2 = Ks2 = 0.0
        for j in keep:
            si, sg = mode_norms(evec[:, j])
            mu = abs(ev_all[j] - tau)
            Ks2 += (mu * si) ** 2
            Kg2 += (mu * sg) ** 2
        out.append({"rank": int(r), "c2": c2, "Khat": float(np.sqrt(Kg2)),
                    "Kphat": float(np.sqrt(Ks2)),
                    "ratio_grad": c2 / np.sqrt(Kg2) if Kg2 > 0 else np.inf,
                    "ratio_sup": c2 / np.sqrt(Ks2) if Ks2 > 0 else np.inf})
    return out


def main() -> None:
    out = {"Re": RE, "n_max": N_MAX}
    print("=" * 78)
    print(f"Best achievable c2 / ||Q|| per Fourier block, Re = {RE}")
    print("=" * 78)
    print("\n  c2/||Q|| is the largest dissipation-strength decay rate a quadratic")
    print("  form of unit operator norm can certify for the linearised dynamics.")
    print(f"\n  {'alpha':>6} {'beta':>8} {'Re_E(a,b)':>11} {'c2/||Q||':>11} "
          f"{'cond(Qw)':>11}")
    rows = []
    for alpha, beta in [(0.0, BETA_C), (0.0, 1.0), (0.0, 2.5), (0.0, 4.0),
                        (0.3, BETA_C), (0.8, 1.6), (1.5, 1.6)]:
        r, Qw, Mh, Mih, ev = best_c2_per_norm(alpha, beta, RE, N_MAX)
        ree = ro.solve_mode(alpha, beta, n_max=N_MAX).re_e
        rows.append({"alpha": alpha, "beta": beta, "Re_E": ree, "c2_per_norm": r,
                     "cond": float(ev.max() / ev.min())})
        print(f"  {alpha:6.2f} {beta:8.4f} {ree:11.4f} {r:11.6f} "
              f"{ev.max()/ev.min():11.3e}")
    out["blocks"] = rows

    print()
    print("=" * 78)
    print("Rank-truncation trade-off at the critical block")
    print("=" * 78)
    print("\n  The Lyapunov solution Q is bounded on L2 but NOT smoothing: its")
    print("  pointwise constants diverge with resolution, so the certificate")
    print("  cannot use it directly.  Truncating Qtilde to rank r fixes that, at")
    print("  the cost of a smaller certified rate c2.")
    lz = 2.0 * np.pi / BETA_C
    print(f"\n  box Lx = Lz = {lz:.4f};  required ratio at Re=20.7 is ~5 (docs/03)")
    print(f"\n  {'rank':>5} {'c2':>11} {'Khat(grad)':>12} {'Kphat(sup)':>12} "
          f"{'c2/Khat':>10} {'c2/Kphat':>10}")
    ranks = [1, 2, 3, 4, 6, 8, 12, 16]
    tr = rank_truncation(0.0, BETA_C, RE, N_MAX, lz, lz, ranks)
    for row in tr:
        print(f"  {row['rank']:5d} {row['c2']:11.6f} {row['Khat']:12.4f} "
              f"{row['Kphat']:12.5f} {row['ratio_grad']:10.5f} "
              f"{row['ratio_sup']:10.5f}")
    out["rank_truncation"] = tr
    best = max(tr, key=lambda r: r["ratio_sup"])
    print(f"\n  best c2/Kphat in this sweep = {best['ratio_sup']:.4f} "
          f"at rank {best['rank']}")
    print("  (a single Fourier block; the ratio scales like sqrt(Lx Lz), and so")
    print("   does the requirement, so the comparison is box-independent)")

    (RESULTS / "05_quadratic_form.json").write_text(json.dumps(out, indent=2))
    print(f"\n  written: {RESULTS / '05_quadratic_form.json'}")


if __name__ == "__main__":
    main()
