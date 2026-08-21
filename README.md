# Beating the energy-stability limit of plane Couette flow

Groundwork for an attempt to prove global asymptotic stability of full 3D plane
Couette flow at a Reynolds number strictly above the classical energy-stability
threshold `Re_E ≈ 20.66`, and so to narrow the open gap `20.66 < Re < 127.7`.

Convention throughout: channel `y ∈ [−1,1]` (half-gap `h = 1`), base flow
`U(y) = y`, `Re = U h / ν`.

## What is in here

| | |
|---|---|
| [`docs/00_prior_art.md`](docs/00_prior_art.md) | What is already known: the SOS / quartic-Lyapunov line (all 2-D), the generalised-energy line, and what remains open |
| [`docs/01_reynolds_orr.md`](docs/01_reynolds_orr.md) | Full derivation of the Reynolds–Orr variational/eigenvalue problem, the `(v, η)` reduction, the Chebyshev–Galerkin discretisation, and the `α = 0` ↔ Rayleigh–Bénard equivalence |
| [`docs/02_critical_mode.md`](docs/02_critical_mode.md) | Symmetries, the critical roll–streak mode, and a proof that the `α = 0` subspace is globally stable at **every** `Re` |
| [`docs/03_quartic_lyapunov.md`](docs/03_quartic_lyapunov.md) | The minimal quartic functional `V = E² + 2E⟨W,u⟩ + Q`, the shear-shift theorem, and an exact, rigorously controllable form of `dV/dt` |

Code lives in `src/couette/`, drivers in `scripts/`, numerical output in
`results/`, tests in `tests/`.

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 scripts/01_reproduce_ReE.py     # Re_E and its validation
PYTHONPATH=src python3 scripts/02_critical_mode.py     # critical mode structure
PYTHONPATH=src python3 scripts/03_shear_shift.py       # the shear-shifted problem
PYTHONPATH=src python3 scripts/04_feasibility.py       # the master inequality
PYTHONPATH=src python3 scripts/05_quadratic_form.py    # how large c2/K can be made
PYTHONPATH=src python3 -m pytest tests/ -q
```

## Results so far

**1. `Re_E` reproduced to 13 significant digits**, by three independent routes
(Galerkin in `(v,η)` variables; an analytic 3×3 boundary determinant in mpmath;
and the implied Rayleigh–Bénard values, which come out as `Ra_c = 1707.7618`,
`a_c = 3.1163`, matching the textbook rigid–rigid numbers):

```
Re_E = 20.6625372177809      at   (α_c, β_c) = (0, 1.5581617774)
```

The literature's "20.65"/"20.7" is this number rounded. A fourth cross-check:
restricting to two-dimensional disturbances (`β = 0`) gives
`Re_E(2D) = 44.3035467` at `α_c = 1.8934`, reproducing Orr's 1907 value. The critical mode is a
streamwise-independent roll–streak: `u` even, `v` even, `w` odd, with 66% of the
energy in the streak, and a large Reynolds–Orr spectral gap `λ₁/λ₀ = 0.26`.

**2. The energy method fails inside a provably stable subspace.**
Streamwise-independent perturbations of plane Couette flow are globally stable at
*every* `Re`: the rolls `(v,w)` obey the unforced 2D Navier–Stokes equations and
decay, after which the streak decays too. So the `Re_E` obstruction is an
artefact of the functional — the energy method is detecting transient lift-up
growth, not an instability.

**3. The shear-shift theorem.** For `V = E² + 2E⟨W,u⟩ + Q[u]` the derivative
`dV/dt` has **no degree-5 term**, and its degree-4 part is exactly `2E` times a
Reynolds–Orr functional in which the base-flow strain `S^U` is replaced by
`S^U − S^W`. Adding a linear term to a quartic functional is therefore precisely
a *shift of the effective shear*.

**4. Admissible shifts are mean-preserving — but that is not a limitation.**
For `W = (g(y),0,0)` the weight is `Σ = 1 − g'`, and controlling the viscous term
without `H²` bounds forces `g(±1) = 0` — equivalently `W ∈ D(A)`, the domain of
the Stokes operator — hence `∫g' dy = 0`. Since `Σ = Ũ'` for the shifted profile
`Ũ = y − g`, this says exactly that `Ũ` must still reach `±1` at the walls: the
mean shear is pinned by no-slip, as for a background profile in the
Doering–Constantin method. It constrains the *shape* of the shift, not the size
of the gain: concentrating `Σ` in wall layers of thickness `δ` gives
`Re_E[Σ_δ] ≃ 29.0713/δ → ∞`, verified numerically to six digits.

**5. Redistribution alone is very effective, and the optimal shift is explicit.**
With `Φ(y) = −Re[û conj(v̂)]` the Reynolds-stress density of the critical mode,
first-order perturbation theory gives `δλ = ∫ g Φ' dy`, so the cheapest shift is
`g ∝ −Φ'` — which vanishes at the walls and is odd automatically, i.e. it is
admissible without any constraint being imposed by hand. Efficiency
`‖Φ'‖ = 0.09135`, validated against nonlinear solves to `O(amplitude²)`.
Certifying `Re = 20.7` needs only `‖g‖_{L²} ≥ 9.59 × 10⁻⁴`.

**6. An exact negativity criterion and a master inequality.** Writing
`u = s û`, `dV/dt = a₂s² + a₃s³ + a₄s⁴`, so `dV/dt < 0` for all `s > 0` iff
`a₂ < 0`, `a₄ ≤ 0` and (`a₃ ≤ 0` or `a₃² < 4a₂a₄`). With explicit constants this
collapses to `b² < 4c₂c₄`, and feasibility reduces to a single ratio that the
still-to-be-built quadratic form `Q` must achieve.

**7. A regrouping that improves the constants ~4.7×.** Because `Ė` and the
quartic term share the factor `𝒟_W`, one has the exact identity
`2Ė⟨W,u⟩ + Γ₄ = −2𝒟_W(E + ⟨W,u⟩) − 2⟨W,u⟩⟨W,F₂(u)⟩`, which turns the dominant
`O(‖W‖)` cubic term into `O(‖W‖²)`. For the first target `Re = 20.7` the
requirement on `Q` drops from `ĉ₂/K̂ ≳ 20` to `ĉ₂/K̂ ≳ 5`. The criterion is
box-independent: both sides scale like `√(L_xL_z)`.

**8. Where the construction currently stands.** The linearised
Orr–Sommerfeld/Squire operator (validated to `2×10⁻¹⁵` against the energy
identity, and to `2×10⁻¹²` against exact eigenvalues at `α = 0`) turns condition
(C2) into a matrix inequality. Solving it shows that the natural Lyapunov-equation
`Q` is bounded on `L²` but **not smoothing** — its pointwise constants diverge
with resolution — so a finite-rank truncation is mandatory. **Rank two is
optimal**, recovering 99.4% of the full-rank certified rate: the dangerous
subspace really is the two-dimensional roll → streak pair. It gives

```
best achievable  ĉ₂/K̂' ≈ 0.183       required (Re = 20.7)  ≈ 4.9
```

## What is *not* established

**No improved bound on `Re` is proved.** The sufficient conditions fall short by
a factor of about 27, and that comparison already flatters the construction (the
achievable figure is for one Fourier block; the certificate needs one `c₂` across
all of them). This is a failure of the present chain of inequalities, not a
disproof of the ansatz — §3.10 identifies the three lossy steps, the crudest by
far being the `L^∞` cubic estimate, which discards the requirement that the
nonlinear term actually correlate with `𝒬̃u`. Nothing has been certified in
interval arithmetic.

**Novelty is not claimed.** Quartic Lyapunov functionals built by polynomial
optimisation are the established state of the art
(Fuentes–Goluskin–Chernyshenko 2022; Iligaray–Aballay–Fuentes 2026;
Darrow–Carlson–Goluskin 2026), and §2.3 is due to Kaiser–Tilgner–von Wahl. What
*is* still open is the 3-D case: every result in that line is two-dimensional,
and 3-D is named as future work. See [`docs/00_prior_art.md`](docs/00_prior_art.md)
and §3.11 of `docs/03_quartic_lyapunov.md`.
