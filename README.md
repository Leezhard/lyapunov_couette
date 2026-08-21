# Beating the energy-stability limit of plane Couette flow

Groundwork for an attempt to prove global asymptotic stability of full 3D plane
Couette flow at a Reynolds number strictly above the classical energy-stability
threshold `Re_E ≈ 20.66`, and so to narrow the open gap `20.66 < Re < 127.7`.

Convention throughout: channel `y ∈ [−1,1]` (half-gap `h = 1`), base flow
`U(y) = y`, `Re = U h / ν`.

## What is in here

| | |
|---|---|
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

The literature's "20.65"/"20.7" is this number rounded. The critical mode is a
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

**4. Admissible shifts are mean-preserving.** For `W = (g(y),0,0)` the weight is
`Σ = 1 − g'`, and controlling the viscous term without `H²` bounds forces
`g(±1) = 0`, hence `∫g' dy = 0`. The shear can only be redistributed across the
channel, never weakened on average.

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
collapses to `b² < 4c₂c₄`, and feasibility reduces to a single ratio
`ĉ₂/K̂ ≳ 10²` that the still-to-be-built quadratic form `Q` must achieve.

## What is *not* established

**No improved bound on `Re` is proved.** The above is the framework plus the
quartic ingredient; `Q` has not been constructed, the constants in the cubic
bound are lossy Cauchy–Schwarz/Poincaré estimates that should be sharpened
before drawing conclusions, and nothing has been certified in interval
arithmetic. Prior art (Kaiser–Tilgner–von Wahl; Mulone and co-workers;
Fuentes–Goluskin–Chernyshenko) has **not** been checked — the literature search
for this session did not complete — so no novelty is claimed. See §3.9 of
`docs/03_quartic_lyapunov.md` for the full accounting.
