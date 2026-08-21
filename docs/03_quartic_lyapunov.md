# 3. A minimal quartic Lyapunov functional and a rigorously controllable `dV/dt`

## 3.1 Why the functional must be quartic, and why *this* quartic

Let `V` be a functional on the divergence-free phase space and expand
`dV/dt = ⟨∇V, F₁u + F₂(u)⟩` by homogeneity degree in `u`. Two constraints
immediately narrow the search.

**(i) A homogeneous quadratic `V` cannot work (unless it is `E`).** Then
`dV/dt = Γ₂ + Γ₃` with `Γ₃` cubic. Along a ray `u = s û`, `dV/dt = a₂s² + a₃s³`,
which is negative for all `s > 0` only if `a₃ = 0` for every `û`. §2.4 showed
`Γ₃ = (1−μ)∫p₂ ∂_x u dV ≢ 0` for any component reweighting. The energy norm is
the unique quadratic form for which `Γ₃` vanishes identically, and it fails at
`Re_E`.

**(ii) A generic quartic `V` cannot work either**, because it produces a
*quintic* `⟨∇V₄, F₂(u)⟩` which dominates everything at large amplitude and
cannot be signed. The quartic terms whose quintic contribution vanishes
identically are exactly those built from quantities conserved by the
nonlinearity. Since `⟨u, F₂(u)⟩ = 0` (the advective nonlinearity is
energy-neutral), `E²` is such a term:

```
⟨∇(E²), F₂(u)⟩ = 2E ⟨u, F₂(u)⟩ = 0.                                      (3.1)
```

Remarkably, so is the *cubic* term `2E⟨W,u⟩`:

```
⟨∇(2E⟨W,u⟩), F₂⟩ = 2⟨W,u⟩⟨u,F₂⟩ + 2E⟨W,F₂⟩ = 0 + 2E⟨W,F₂(u)⟩,           (3.2)
```

which is only *quartic*. This is the whole reason the ansatz below is the
minimal viable one.

**The ansatz.**

```
V[u] = E² + 2E⟨W,u⟩ + Q[u],      Q[u] = ⟨𝒬u, u⟩,                        (3.3)
```

with `W` a fixed divergence-free field with `n·W = 0` on the walls, and `𝒬` a
bounded symmetric positive operator. Completing the square,

```
V = ( E + ⟨W,u⟩ )² + ( Q[u] − ⟨W,u⟩² ),                                  (3.4)
```

so **`V > 0` for all `u ≠ 0` provided `Q[u] > ⟨W,u⟩²`**, which holds whenever
`𝒬 ⪰ q₋ I` with `q₋ > ‖W‖²`.

## 3.2 The exact derivative

Differentiating (3.3) and using `Ė = ⟨u, F₁u⟩` (exactly quadratic, by (1.1)):

```
dV/dt = Γ₂ + Γ₃ + Γ₄,     with no degree-5 term, where

Γ₂ = 2⟨𝒬u, F₁u⟩                                                          (3.5a)
Γ₃ = 2Ė⟨W,u⟩ + 2E⟨W, F₁u⟩ + 2⟨𝒬u, F₂(u)⟩                                 (3.5b)
Γ₄ = 2E( Ė + ⟨W, F₂(u)⟩ )                                                (3.5c)
```

Each degree has a clean interpretation: `Γ₂` is a Lyapunov condition for the
*linearised* operator, `Γ₄` is a modified *energy-stability* condition, and `Γ₃`
is the cross term that must be squeezed between them.

## 3.3 The shear-shift theorem

This is the structural payoff of the ansatz.

> **Theorem.** Let `W` be divergence-free with `n·W = 0` on `∂Ω`, and let
> `S^W_ij = ½(∂_j W_i + ∂_i W_j)`. Then
> ```
> ⟨W, F₂(u)⟩ = ∫ u_i u_j S^W_ij dV                                       (3.6)
> ```
> and consequently
> ```
> Γ₄ = 2E [ − ∫ u_i u_j ( S^U_ij − S^W_ij ) dV − Re⁻¹ ∫|∇u|² dV ].       (3.7)
> ```
> That is: **adding `2E⟨W,u⟩` to `V` is exactly equivalent, at quartic order, to
> replacing the base-flow strain `S^U` by the shifted strain `S^U − S^W` in the
> Reynolds–Orr problem.** Hence
> ```
> Γ₄ ≤ 0  for all u   ⟺   Re ≤ Re_E[ S^U − S^W ].                        (3.8)
> ```

*Proof of (3.6).* `P` is an orthogonal projection and `W = PW`, so
`⟨W, F₂(u)⟩ = −⟨W, (u·∇)u⟩ = −∫ W_i u_j ∂_j u_i dV`. Integrating by parts,
using `∇·u = 0` and `u = 0` on the walls to kill the boundary term,
`= ∫ u_i u_j ∂_j W_i dV`, and since `u_i u_j` is symmetric in `(i,j)` only the
symmetric part of `∂_j W_i` survives, giving `∫ u_i u_j S^W_ij dV`. Combining
with (1.3) gives (3.7). ∎

So the quartic term of `dV/dt` is governed by *exactly the eigenvalue problem
already solved in §1*, with a modified weight — which is why the solver in
`src/couette/reynolds_orr.py` carries a general `Σ(y)` throughout.

## 3.4 Which shifts are admissible: the mean-preserving constraint

Take `W = (g(y), 0, 0)`, which is automatically divergence-free with `n·W = 0`.
Then `S^W_12 = S^W_21 = g'(y)/2`, so `∫u_iu_jS^W_ij dV = ∫ g'(y) u v dV`, and

```
Σ(y) = 1 − g'(y).                                                        (3.9)
```

Now examine the `W`-terms of `Γ₃`, using `⟨W, Pf⟩ = ⟨W, f⟩`:

```
⟨W, F₁u⟩ = ⟨W, −y∂_x u⟩ − ⟨W, v e_x⟩ + Re⁻¹⟨W, Δu⟩.
```

* `⟨W, −y∂_x u⟩ = −∫ g(y) y ∂_x u dV = ∫ u ∂_x(g y) dV = 0`, since `g(y)y` does
  not depend on `x`.
* `⟨W, v e_x⟩ = ∫ g(y) v dV = L_xL_z ∫ g(y) v̄(y) dy` where `v̄` is the `xz`-mean
  of `v`. But continuity plus `v(±1) = 0` forces the `(0,0)` Fourier mode of `v`
  to vanish identically, so **this term is identically zero.**
* `Re⁻¹⟨W, Δu⟩`: by Green's identity,
  `∫ (g Δu − u Δg) dV = ∮ (g ∂_n u − u ∂_n g) dS`. The second boundary piece
  vanishes because `u = 0` on the walls; the first is `∮ g ∂_n u dS`, which
  involves the **wall shear stress** of the perturbation.

Controlling `∮ g ∂_n u dS` would require `H²` control of `u`, which `dV/dt` does
not supply. Demanding instead the clean estimate
`|⟨W, Δu⟩| ≤ ‖g''‖ ‖u‖` therefore **forces**

```
g(±1) = 0     ⟹     ∫₋₁¹ g'(y) dy = g(1) − g(−1) = 0.                   (3.10)
```

> **The mean of `Σ` is pinned at 1.** The admissible shifts are exactly the
> **mean-preserving redistributions** of the shear across the channel. One
> cannot simply weaken the shear everywhere; one can only move it.

With (3.10) in force,

```
⟨W, u⟩   = L_xL_z ∫₋₁¹ g(y) ū(y) dy,
⟨W, F₁u⟩ = Re⁻¹ L_xL_z ∫₋₁¹ g''(y) ū(y) dy,                              (3.11)
```

so both `W`-functionals see **only the mean streak profile** `ū(y)` — a
substantial structural simplification.

## 3.5 Mean-preserving redistribution is nevertheless very effective

This is the central numerical finding (`scripts/03_shear_shift.py`). It is not
obvious a priori that a mean-preserving `Σ` can raise `Re_E[Σ]` at all. It can,
strongly — because the no-slip condition suppresses the critical mode's Reynolds
stress near the walls, so shear moved *out of the core and into the wall layers*
is largely wasted on the disturbance.

For the one-parameter family `g = d·y(1−y²)`, i.e. `Σ = 1 − d(1−3y²)`:

| `d` | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| `Re_E[Σ]` | 20.663 | 21.397 | 22.184 | 23.935 | 25.967 | 31.157 | 43.328 | 55.423 |

### The optimal shift, derived from the critical mode

Let `u_c` be the critical mode normalised to unit dissipation and define the
**Reynolds-stress density**

```
Φ(y) = − Re[ û_c(y) conj(v̂_c(y)) ],        λ₀ = ∫ Φ dy = 1/Re_E.        (3.12)
```

By Hellmann–Feynman, perturbing the weight by `δΣ = −g'` gives

```
δλ = − ∫ g'(y) Φ(y) dy = + ∫ g(y) Φ'(y) dy,                              (3.13)
```

the integration by parts being legitimate precisely because of (3.10). Hence

> **the `L²`-cheapest shift direction is `g ∝ −Φ'`**, and the minimum cost of
> raising the threshold from `Re_E` to a target `Re` is, to leading order,
> ```
> ‖g‖_{L²(−1,1)} ≥ ( 1/Re_E − 1/Re ) / ‖Φ'‖_{L²}.                        (3.14)
> ```

Note `Φ(±1) = Φ'(±1) = 0` automatically (both `u` and `v` vanish at the walls),
and `Φ` is even, so `g ∝ −Φ'` is odd and satisfies (3.10) with nothing imposed
by hand — the optimal shift is *automatically admissible*. The efficiency is

```
‖Φ'‖_{L²} = 0.0913455.
```

**Validation.** The first-order prediction against the full nonlinear solve:

| `‖g‖_{L²}` | predicted `Re_E[Σ]` | actual `Re_E[Σ]` | rel. err |
|---|---|---|---|
| 1e−4 | 20.66643785 | 20.66643774 | 5.7e−9 |
| 1e−3 | 20.70160995 | 20.70159837 | 5.6e−7 |
| 1e−2 | 21.06002945 | 21.05880927 | 5.8e−5 |
| 1e−1 | 25.46976378 | 25.25196916 | 8.6e−3 |

The error scales as the square of the amplitude, as a first-order theory should.
`α = 0` remains the critical wavenumber over this whole range.

**Cost of the first target.** For `Re = 20.7` — the modest first goal — (3.14)
gives

```
‖g‖_{L²} ≥ 9.5887 × 10⁻⁴ .
```

The required shift is *tiny*. This is the quantitative sense in which the
proposed functional is "minimal viable".

## 3.6 Convexity of the search over `Σ`

For fixed normalised `u` the Rayleigh quotient of (1.2) is **affine** in `Σ`.
Therefore `λ_max(Σ) = max_u (affine in Σ)` is a **convex** function of `Σ`, and
so is the maximum over wavenumbers. Minimising `λ_max` over a convex set of
admissible shifts (e.g. `‖g‖ ≤ ρ`) is thus a convex program with no spurious
local minima — it can be solved reliably by cutting planes, or cast as an SDP in
the epigraph form `min t s.t. A(Σ) ⪯ t B`. Convexity also justifies restricting
to *even* `Σ` (§2.1): symmetrising a shift cannot make it worse.

## 3.7 The exact negativity criterion

Write `u = s û` with `‖û‖ = 1` and `s > 0` the amplitude, and let
`D̂ = ∫|∇û|² dV`. Then

```
dV/dt = a₂(û) s² + a₃(û) s³ + a₄(û) s⁴,
```

and `dV/dt < 0` for all `s > 0` is **equivalent** to `a₂ + a₃ s + a₄ s² < 0` for
all `s > 0`, i.e. to

```
a₂ < 0,   a₄ ≤ 0,   and   ( a₃ ≤ 0   or   a₃² < 4 a₂ a₄ ).               (3.15)
```

(If `a₄ = 0` the last clause reduces to `a₃ ≤ 0`; `a₄ > 0` is impossible;
equality `a₃² = 4a₂a₄` gives a tangency `dV/dt = 0` and is excluded.)

This is exact — no inequalities have been spent yet. It is also the natural
target for an SOS/SDP formulation, since it is a condition on the coefficients
of a one-variable polynomial whose coefficients are quadratic-form-valued in `û`.

## 3.8 Sufficient conditions with explicit constants

Assume the three structural bounds

```
(C2)   Γ₂ ≤ − c₂ D[u]                          (linear certificate)
(C3)   |Γ₃| ≤ b ‖u‖ D[u]                       (cubic control)
(C4)   Γ₄ ≤ − c₄ ‖u‖² D[u],  c₄ = 1/Re − 1/Re_E[Σ] > 0   (quartic margin)
```

Then `a₂ ≤ −c₂D̂`, `|a₃| ≤ bD̂`, `a₄ ≤ −c₄D̂`, the factors of `D̂` match on both
sides of `a₃² < 4a₂a₄`, and (3.15) reduces to the single **master inequality**

```
                          b² < 4 c₂ c₄ .                                 (3.16)
```

The powers of `D̂` cancelling is not an accident — it is why (C2) must be stated
at *dissipation* strength rather than at `‖u‖²` strength.

### (C2) is attainable, by a two-regime argument

Write `𝒬 = τI + 𝒬̃`. Then `Γ₂ = 2τĖ + 2⟨𝒬̃u, F₁u⟩`. Let `θ = D[u]/‖u‖² ≥ λ₁`.
Two sharp bounds on the production are available,
`−∫uv ≤ D/Re_E` and `−∫uv ≤ ½‖u‖²`, so

```
Ė/D ≤ min( 1/Re_E , 1/(2θ) ) − 1/Re .
```

* **High-dissipation regime** `θ ≥ θ* ≈ Re/2`: the second bound dominates and
  `2τĖ ≤ −cD` on its own — the tail is handled by the energy term alone.
* **Low-dissipation regime** `θ < θ*`: here `D < θ*‖u‖²`, so a bound of the form
  `−c‖u‖²` *implies* `−(c/θ*)D`. A **bounded, finite-rank** `𝒬̃` therefore
  suffices, and it only has to act on the finitely many low-dissipation modes.

At `Re = 20.7`, `θ* ≈ 10.35`, while `λ₁ = π²/4 = 2.4674` (confirmed
numerically: `min D/‖u‖² = π²/4 + k²`, approached by the streamwise-uniform
streak `u = cos(πy/2)` as `k → 0`). So the "hard" region is the finite band
`2.467 ≤ θ ≤ 10.35` — precisely where the critical roll–streak lives.

This is exactly the resolved/tail split the programme calls for: **`𝒬̃`
finite-rank and smooth on the low-dissipation modes, the infinite-dimensional
tail carried by `τE`**, with `θ*` the rigorous, explicit separation threshold.

### (C3): the cubic constant

With the three pieces of (3.5b), using `|∫uv| ≤ D/Re_E` (valid for both signs, by
the symmetry `(u,v,w)(x,y,z) → (u,−v,w)(x,−y,z)`, which preserves
divergence-freedom and `D` while flipping the sign of `∫uv`), and (3.11):

```
|2Ė⟨W,u⟩|          ≤ 2( 1/Re_E + 1/Re ) ‖W‖ ‖u‖ D
|2E⟨W,F₁u⟩|        ≤ Re⁻¹ ‖W''‖ ‖u‖³      ≤ (‖W''‖ / (Re λ₁)) ‖u‖ D
|2⟨𝒬u,F₂(u)⟩|      ≤ 2K ‖u‖³              ≤ (2K/λ₁) ‖u‖ D
```

where the last line uses `⟨u,(u·∇)u⟩ = 0` to replace `𝒬` by `𝒬̃`, then
`⟨𝒬̃u,(u·∇)u⟩ = −⟨(u·∇)𝒬̃u, u⟩`, and

```
K := sup_{u≠0}  ‖∇(𝒬̃u)‖_{L^∞} / ‖u‖_{L²}   <  ∞  for finite-rank smooth 𝒬̃.
```

Hence

```
b = 2( 1/Re_E + 1/Re ) ‖W‖ + ( ‖W''‖/Re + 2K ) / λ₁ .                    (3.17)
```

### The feasibility ratio

`c₂` and `K` both scale linearly with the size of `𝒬̃`; writing `𝒬̃ = ρ𝒬̂` with
`𝒬̂` normalised, `c₂ = ρĉ₂`, `K = ρK̂`, and `b = b₀ + ρB̂` with
`B̂ = 2K̂/λ₁`. Maximising `4c₂c₄ − b²` over `ρ` gives a positive value **iff**

```
                 4 c₄ ĉ₂  >  16 B̂ b₀ ,    i.e.    ĉ₂ / K̂  >  8 b₀ / (λ₁ c₄).   (3.18)
```

`scripts/04_feasibility.py` evaluates (3.18) over a sweep of targets and shift
amplitudes, with the box `L_x = L_z = 2π/β_c`. Writing `‖g‖ = safety × (3.14)`:

| `Re` | safety | `‖g‖_{L²}` | `Re_E[Σ]` | `c₄` | `b₀` | required `ĉ₂/K̂` |
|---|---|---|---|---|---|---|
| 20.70 | 1.5 | 1.44e−3 | 20.7188 | 4.38e−5 | 2.88e−3 | 213 |
| 20.70 | 2.0 | 1.92e−3 | 20.7376 | 8.76e−5 | 3.83e−3 | 142 |
| 20.70 | 4.0 | 3.84e−3 | 20.8132 | 2.63e−4 | 7.67e−3 | 94.6 |
| 20.70 | 10.0 | 9.59e−3 | 21.0434 | 7.88e−4 | 1.92e−2 | 78.9 |
| 21.00 | 4.0 | 3.41e−2 | 22.0819 | 2.33e−3 | 6.73e−2 | 93.6 |
| 22.00 | 10.0 | 3.22e−1 | 52.7022 | 2.65e−2 | 6.13e−1 | 75.1 |

Two features are worth noting. The required ratio is **almost independent of the
target `Re`** — pushing from 20.7 to 22.0 barely changes it — because `b₀` and
`c₄` both scale linearly with `‖g‖`. And it *decreases* with the safety factor,
saturating near 75: overshooting the quartic margin is cheap, so there is no
reason to run close to the minimum of (3.14).

**This is the concrete target for the next stage of the work**: construct a
finite-rank `𝒬̂` whose certified linear decay rate `ĉ₂` exceeds its certified
nonlinear constant `K̂` by a factor of order 10², using the constants as they
stand — or, better, first sharpen them (see §3.9) and lower the target.

## 3.9 Honest accounting: what is proved, and what is not

**Established here, rigorously:**

1. `Re_E = 20.6625372177809` at `(α,β) = (0, 1.5581617774)`, by three
   independent routes agreeing to 13 digits (§1.7).
2. The energy method fails first inside a subspace — streamwise-independent
   perturbations — on which plane Couette flow is **globally stable at every
   `Re`** (§2.3). The `Re_E` obstruction is an artefact of the functional.
3. The shear-shift theorem (3.6)–(3.8): the quartic term of `dV/dt` for
   `V = E² + 2E⟨W,u⟩ + Q` is exactly `2E` times a Reynolds–Orr functional with
   strain `S^U − S^W`. No degree-5 term is generated.
4. Admissible shifts are mean-preserving (3.10), and both `W`-functionals see
   only the mean streak (3.11).
5. Mean-preserving redistribution raises `Re_E[Σ]` efficiently; the optimal
   direction is `g ∝ −Φ'` with efficiency `‖Φ'‖ = 0.0913`, validated against
   nonlinear solves (§3.5).
6. The exact negativity criterion (3.15) and the master inequality (3.16) with
   explicit constants (3.17), plus the two-regime argument showing (C2) is
   attainable with a finite-rank `𝒬̃` and an explicit threshold `θ* ≈ Re/2`.

**Not established — the open work:**

* **No improved bound on `Re` is proved yet.** Everything above is the
  *framework* plus the quartic ingredient. Points 1–6 do not by themselves
  certify global stability at any `Re > Re_E`.
* `𝒬` has not been constructed. Producing a finite-rank `𝒬̂` with certified
  `ĉ₂` and `K̂` satisfying (3.18) is the next concrete task, and it is where the
  construction may still fail.
* The constants in (3.17) come from Cauchy–Schwarz and Poincaré and are
  certainly not sharp. In particular the term `2Ė⟨W,u⟩` is better handled by the
  regrouping `2Ė(E + ⟨W,u⟩) = 2ĖΨ`, which keeps the sign information that
  Cauchy–Schwarz throws away; the crude route makes `b₀` grow like `√(L_xL_z)`
  and so makes (3.18) box-dependent, which the regrouped form need not be. This
  should be redone before concluding that any particular target is out of reach.
* No interval-arithmetic / rational-reconstruction certification has been done.
  All numbers above are double-precision (plus one mpmath cross-check) and are
  *evidence*, not proof.
* **Prior art has not been checked.** The literature searches for this session
  did not complete. Before claiming novelty, the work of Kaiser–Tilgner–von Wahl
  on generalised energy functionals for plane Couette, the Mulone–Rionero and
  Falsaperla–Giacobbe–Mulone line on tilted perturbations, and
  Fuentes–Goluskin–Chernyshenko on SOS-based global stability beyond the energy
  limit must all be read. It is entirely possible that parts of §3.3–§3.5 are
  known.
