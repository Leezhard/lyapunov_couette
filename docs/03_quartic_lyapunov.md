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

The cubic term `2E⟨W,u⟩` cannot produce a quintic by mere degree counting
(`∇V₃` is quadratic, `F₂` is quadratic, so at most quartic). What energy
neutrality buys there is different, and it is what makes the ansatz clean:

```
⟨∇(2E⟨W,u⟩), F₂⟩ = 2⟨W,u⟩⟨u,F₂(u)⟩ + 2E⟨W,F₂(u)⟩ = 0 + 2E⟨W,F₂(u)⟩,     (3.2)
```

i.e. the degree-4 **cross term** `2⟨W,u⟩⟨u,F₂(u)⟩` vanishes, leaving `Γ₄` as a
single clean object. Without that cancellation `Γ₄` would be
`2E·Ė + 2⟨W,u⟩⟨u,F₂(u)⟩ + 2E⟨W,F₂(u)⟩` and the shear-shift theorem below would
not hold.

Both cancellations use `⟨u, F₂(u)⟩ = 0` in the *unweighted* `L²` inner product.
This is special to `E²`, not to quartic functionals in general: replacing `E` by
a weighted energy `⟨Mu,u⟩` with `M ≠ cI` — the reweighting of §2.4 — brings the
quintic term straight back.

**The ansatz.**

```
V[u] = E² + 2E⟨W,u⟩ + Q[u],      Q[u] = ⟨𝒬u, u⟩,                        (3.3)
```

with `W` a fixed divergence-free field with `n·W = 0` on the walls, and `𝒬` a
bounded symmetric positive operator. Completing the square,

```
V = ( E + ⟨W,u⟩ )² + ( Q[u] − ⟨W,u⟩² ),                                  (3.4)
```

so **`V > 0` for all `u ≠ 0` if and only if `Q[u] > ⟨W,u⟩²` for all `u ≠ 0`**,
i.e. iff `𝒬 ≻ W ⊗ W`. Sufficiency is immediate from (3.4). Necessity: along a
ray `u = s û` with `⟨W,û⟩ < 0`, the choice `s = −⟨W,û⟩/E[û] > 0` annihilates the
square exactly, leaving `V = s²(Q[û] − ⟨W,û⟩²)`; and replacing `û` by `−û`
covers the rays with `⟨W,û⟩ > 0`. A convenient sufficient surrogate is
`𝒬 ⪰ q₋ I` with `q₋ > ‖W‖²` (Cauchy–Schwarz).

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

*Proof of (3.6).* `P` is an orthogonal projection and `W = PW` — this is where
both `∇·W = 0` and `W·n = 0` are load-bearing, since together they kill
`∫W·∇q = −∫q ∇·W + ∮ q W·n` for the nonlinear pressure `q`. Hence
`⟨W, F₂(u)⟩ = −⟨W, (u·∇)u⟩ = −∫ W_i u_j ∂_j u_i dV`. Integrating by parts and
using `∇·u = 0`, the boundary term is `−∮ (W·u)(u·n) dS`, which vanishes under
`u·n = 0` alone (full no-slip is not needed here; it is needed only for the
viscous term in §3.4). So `⟨W,F₂(u)⟩ = ∫ u_i u_j ∂_j W_i dV`, and since
`u_i u_j` is symmetric in `(i,j)` only the symmetric part of `∂_j W_i` survives.
Combining with (1.3) gives (3.7). ∎

Note `W` is **not** required to vanish at the walls for (3.6); that constraint
arrives separately in §3.4. This is verified numerically in
`tests/test_shear_shift_theorem.py`, which checks (3.6) against a directly
evaluated advective nonlinearity using shift profiles that are deliberately
non-zero at `y = ±1`.

> **Warning: the quartic order alone certifies nothing.** Under the hypotheses
> of the theorem, `W = U = (y,0,0)` is admissible — it is divergence-free with
> `W·n = 0`. Then `S^W = S^U` and `Γ₄ = −2E Re⁻¹∫|∇u|² < 0` at *every* `Re`,
> which would "prove" global stability of plane Couette flow outright. The
> freedom in `W` must therefore be paid for elsewhere in `dV/dt`, and §3.4
> identifies exactly where: the viscous part of `Γ₃` forces `g(±1) = 0`, which
> excludes `W = U` and confines the shift to mean-preserving redistributions.

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
dV/dt = a₂(û) s² + a₃(û) s³ + a₄(û) s⁴ = s² q(s),   q(s) = a₂ + a₃s + a₄s².
```

The condition we actually want is not bare negativity but a **uniform margin per
direction**, `sup_{s>0} (dV/dt)/s² < 0`, equivalently `q < 0` on the *closed*
half-line `[0,∞)`, equivalently: there is `c(û) > 0` with `dV/dt ≤ −c(û)‖u‖²`
for all `s`. That is exactly

```
a₂ < 0,   a₄ ≤ 0,   and   ( a₃ ≤ 0   or   a₃² < 4 a₂ a₄ ).               (3.15)
```

(If `a₄ = 0` the last clause reduces to `a₃ ≤ 0`; `a₄ > 0` is impossible;
equality `a₃² = 4a₂a₄` gives a tangency `q = 0` at a finite amplitude and is
excluded.) Bare negativity on the *open* half-line is strictly weaker — it only
requires `a₂ ≤ 0`, `a₄ ≤ 0`, `(a₃ ≤ 0 or a₃² < 4a₂a₄)` with
`(a₂,a₃,a₄) ≠ (0,0,0)` — and it is not what a Lyapunov argument can use. Note
also that `a₃` is odd under `û → −û`, so requiring (3.15) for *every* direction
collapses the `a₃ ≤ 0` branch to `a₃ = 0`.

**Why a margin, and not just `dV/dt < 0`.** The unit sphere of an
infinite-dimensional phase space is not compact, and `a₂(û)` is unbounded below
(it contains `−Re⁻¹‖∇û‖²`), so pointwise strict negativity on the sphere does
*not* by itself give a rate uniform in `û`. This is precisely why the sufficient
conditions of §3.8 are stated with constants measured against the **dissipation**
`D[u]` rather than against `‖u‖²`: that is what makes the margin uniform.

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
`𝒬̂` normalised, `c₂ = ρĉ₂`, `K = ρK̂`, and `b = b₀ + ρB̂` with `B̂ = 2K̂/λ₁`.
Maximising `F(ρ) = 4ρĉ₂c₄ − (b₀ + ρB̂)²` over `ρ > 0` gives
`F_max = (4ĉ₂c₄/B̂)(ĉ₂c₄/B̂ − b₀)`, so a feasible `ρ` exists **iff**

```
                 ĉ₂ c₄  >  b₀ B̂ ,     i.e.     ĉ₂ / K̂  >  2 b₀ / (λ₁ c₄).   (3.18ʹ)
```

(This is verified symbolically and numerically in
`tests/test_feasibility_algebra.py`. An earlier version of this section carried
a spurious factor of 4 here, making every target four times harder than it is.)

**A sharper choice of the cubic bound.** The estimate
`|⟨𝒬̃u,(u·∇)u⟩| ≤ ‖∇(𝒬̃u)‖_∞ ‖u‖²` is not the only one available. Using instead
`|⟨𝒬̃u,(u·∇)u⟩| ≤ ‖𝒬̃u‖_∞ ‖u‖ ‖∇u‖` gives

```
|2⟨𝒬u,F₂(u)⟩| ≤ (2K' / √λ₁) ‖u‖ D,        K' := sup ‖𝒬̃u‖_∞ / ‖u‖,
```

which needs no derivative of `𝒬̃` at all. Since `‖∇(𝒬̃u)‖_∞ ≳ 2‖𝒬̃u‖_∞` for
modes at the critical wavenumber, while `√λ₁ = 1.571 < λ₁ = 2.467`, this is
roughly a further 1.3× gain. Both bounds are reported by
`scripts/04_feasibility.py`; the better one should be used.

`scripts/04_feasibility.py` evaluates (3.18ʹ) over a sweep of targets and shift
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

## 3.9 Sharpening: the regrouped form of `dV/dt`

The estimate (3.17) is wasteful in one specific place. The term `2Ė⟨W,u⟩` was
bounded by Cauchy–Schwarz, which throws away the fact that `Ė` and the quartic
term `Γ₄` share the *same* factor `𝒟_W`. Keeping it gives an exact identity.

Write `𝒟_W := −( Ė + ⟨W, F₂(u)⟩ ) = ∫ Σ u v dV + Re⁻¹ D`, so that
`Γ₄ = −2E 𝒟_W` and the quartic condition (C4) reads `𝒟_W ≥ c₄ D`. Substituting
`Ė = −𝒟_W − ⟨W, F₂(u)⟩` into `2Ė⟨W,u⟩` and adding `Γ₄`:

> **Regrouping identity.**
> ```
> 2Ė⟨W,u⟩ + Γ₄ = − 2 𝒟_W Ψ − 2⟨W,u⟩⟨W, F₂(u)⟩,      Ψ := E + ⟨W,u⟩.       (3.18)
> ```
> Hence, exactly,
> ```
> dV/dt = − 2 𝒟_W Ψ − 2⟨W,u⟩⟨W,F₂(u)⟩ + 2E⟨W,F₁u⟩ + Γ₂ + 2⟨𝒬u,F₂(u)⟩.     (3.19)
> ```

Every `W`-dependent term in (3.19) is now either **the leading negative term**
`−2𝒟_W Ψ`, or **quadratic in `W`**. Since `Ψ = E + ⟨W,u⟩ ≥ ½‖u‖² − ‖W‖‖u‖ > 0`
as soon as `‖u‖ > 2‖W‖`, the leading term is negative except in a ball of radius
`2‖W‖`, where it is bounded by `4C_D‖W‖²D` with
`C_D = ‖Σ‖_∞/(2λ₁) + 1/Re`. Explicitly,

```
−2𝒟_W Ψ ≤ − c₄ D ‖u‖² + 2c₄‖W‖‖u‖D + 4(C_D + c₄)‖W‖² D.
```

Two further estimates improve on §3.8:

* `|2⟨W,u⟩⟨W,F₂(u)⟩| ≤ (‖W‖ ‖g'‖_∞ / λ₁) ‖u‖ D` — **quadratic in the shift
  amplitude**, where the Cauchy–Schwarz route gave a term linear in it.
* For `2E⟨W,F₁u⟩`, integrate by parts *once* rather than twice: using
  `ū(±1) = 0`, `∫g''ū dy = −∫g'ū' dy`, and `‖ū'‖²_{L²(−1,1)} ≤ ‖∇u‖²/(L_xL_z)`,
  ```
  |2E⟨W,F₁u⟩| ≤ (‖W'‖ / Re) ‖u‖² D^{1/2} ≤ ( ‖W'‖ / (Re √λ₁) ) ‖u‖ D,
  ```
  which replaces `‖W''‖/(Re λ₁)` by the smaller `‖W'‖/(Re √λ₁)`.

Collecting, with `‖u‖ = s` and `D = s²D̂`, `dV/dt ≤ D̂[ −c₄s⁴ + b s³ − c₂^eff s² ]`
where

```
b       = ‖W'‖/(Re√λ₁) + 2K/λ₁ + 2c₄‖W‖ + ‖W‖‖g'‖_∞/λ₁                   (3.20)
c₂^eff  = c₂ − 4(C_D + c₄)‖W‖²                                           (3.21)
```

and the master inequality becomes `b² < 4 c₂^eff c₄`. Only the first two terms of
(3.20) are first order in the shift amplitude; the last two are second order, and
the correction in (3.21) is second order too.

**The gain.** For the optimal shift `g ∝ −Φ'` the relevant norm ratios are

```
‖g'‖/‖g‖ = 3.4226,     ‖g''‖/‖g‖ = 15.4509,     ‖g'‖_∞/‖g‖ = 4.1685.
```

At `Re = 20.7` the first-order `‖W‖`-coefficient of `b₀` drops from `0.4959`
(§3.8) to `0.1053` — a factor **4.71**. Including the second-order terms of
(3.20), `scripts/04_feasibility.py` gives:

| `Re` | safety | required `ĉ₂/K̂`, §3.8 | required `ĉ₂/K̂`, §3.9 | gain |
|---|---|---|---|---|
| 20.70 | 1.5 | 53.2 | 11.6 | 4.60× |
| 20.70 | 2.0 | 35.5 | 7.8 | 4.56× |
| 20.70 | 4.0 | 23.7 | 5.4 | 4.42× |
| 20.70 | 10.0 | 19.7 | **4.9** | 4.03× |
| 21.00 | 2.0 | 35.1 | 9.6 | 3.66× |
| 22.00 | 2.0 | 33.8 | 15.3 | 2.21× |
| 22.00 | 10.0 | 18.8 | 27.7 | 0.68× |

So for the first target `Re = 20.7` the requirement on `𝒬` is a ratio of order
**5**, not 10² — a materially different proposition.

The last row is instructive: the regrouping is only an improvement while the
shift is small, because the terms it leaves behind are `O(‖W‖²)` and eventually
overtake the `O(‖W‖)` term it removed. That is exactly the right behaviour for
this programme, whose whole strategy is to sit just above `Re_E` with a tiny
shift and then continue upward in small steps. It also says the §3.8 and §3.9
bounds should both be evaluated and the better one taken.

**The box dependence cancels.** `‖W'‖ = √(L_xL_z)‖g'‖_{L²(−1,1)}`, so `b₀`, and
with it the *required* ratio in (3.18ʹ), grows like `√(L_xL_z)`. But so does the
*available* ratio: for `𝒬̃` built from modes normalised in `L²(Ω)`, `K̂` (a
pointwise norm) scales like `1/√(L_xL_z)` while `ĉ₂` (a ratio of integrals) does
not, so `ĉ₂/K̂ ∝ √(L_xL_z)` as well. **The feasibility criterion is therefore
box-independent**, which is what one wants — a construction that only worked in
small boxes would be of little interest. Sharpening the mean-streak estimate
(`‖ū'‖ ≤ ‖∇u‖/√(L_xL_z)` is saturated only by `xz`-uniform fields, for which
`v = w = 0` and `Ė = −Re⁻¹D` is strongly negative) would lower both sides
further, but is not needed to make the criterion well-posed.

## 3.10 Honest accounting: what is proved, and what is not

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
* **`𝒬` has not been constructed.** Producing a finite-rank `𝒬̂` with certified
  `ĉ₂` and `K̂` satisfying (3.18ʹ) is the immediate next task, and it is where
  the construction may still fail. Doing it needs the linearised Orr–Sommerfeld
  /Squire operator, which is not implemented here. A crude scaling estimate for
  a rank-two `𝒬̃` built from the critical mode suggests the achievable ratio is
  of order unity against a requirement of order 5 — i.e. short, but by a factor
  of a few rather than by orders of magnitude, and computed with bounds
  (particularly the `L^∞` cubic estimate) that are the crudest in the chain.
  That estimate is quoted only to indicate scale; it has not been computed
  carefully and should not be relied on either way.
* The constants are still not sharp. §3.9 already removes the worst loss (a
  factor 4.71), but the surviving first-order term keeps a `√(L_xL_z)` box
  dependence that a separate treatment of the mean-streak component should
  remove. Nothing here should be read as showing a particular target is out of
  reach.
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
