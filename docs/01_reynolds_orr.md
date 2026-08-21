# 1. The Reynolds–Orr problem for plane Couette flow

## 1.1 Setting and conventions

Channel `y ∈ [−1, 1]` (half-gap `h = 1`), periodic in the streamwise direction
`x` (period `L_x`) and the spanwise direction `z` (period `L_z`). The walls move
at `∓U`, so the base flow is

```
U(y) = (y, 0, 0),        U'(y) = 1,
```

and the Reynolds number is `Re = U h / ν = 1/ν`. This is the convention in which
the classical answer is `Re_E ≈ 20.66` and the lowest known nontrivial exact
solution appears near `Re ≈ 127.7`.

A perturbation `u = (u, v, w)` about the base flow satisfies

```
∂_t u + y ∂_x u + v e_x + (u·∇)u = −∇p + Re⁻¹ Δu,     ∇·u = 0,
u|_{y=±1} = 0,     u periodic in x, z.
```

Write `E = ½ ∫ |u|² dV` and let `⟨a, b⟩ = ∫ a·b dV` over the periodic box.
It is convenient to write the dynamics as

```
∂_t u = F₁u + F₂(u),
F₁u = P[ −y ∂_x u − v e_x + Re⁻¹ Δu ],      F₂(u) = −P[ (u·∇)u ],
```

with `P` the Leray projector onto divergence-free fields with `n·u = 0` on the
walls. `F₁` is linear, `F₂` is quadratic. Note `P` is an orthogonal projection,
so `⟨W, P f⟩ = ⟨W, f⟩` for any `W` in the divergence-free space.

## 1.2 The energy identity

Take the inner product of the momentum equation with `u` and integrate.

* **Base-flow advection.** `∫ u·(y ∂_x u) dV = ∫ y ∂_x(|u|²/2) dV = 0` by
  periodicity in `x` (the coefficient `y` does not depend on `x`).
* **Production.** `∫ u·(v e_x) dV = ∫ u v dV`. This is the only term that
  couples the perturbation to the base flow.
* **Nonlinearity.** `∫ u·(u·∇)u dV = ∫ (u·∇)(|u|²/2) dV = 0` using `∇·u = 0`
  and `u = 0` on the walls.
* **Pressure.** `∫ u·∇p dV = −∫ p ∇·u dV = 0`, again using the boundary
  conditions.
* **Dissipation.** `Re⁻¹ ∫ u·Δu dV = −Re⁻¹ ∫ |∇u|² dV`.

Hence the **exact** identity

```
dE/dt = − ∫ u v dV − Re⁻¹ ∫ |∇u|² dV.                                     (1.1)
```

The manipulations above assume enough regularity to integrate by parts (strong
solutions suffice) and that the perturbation pressure is single-valued and
periodic, so that `∫ ∇p·u dV` really integrates to zero rather than picking up a
mean pressure-gradient contribution.

Two features of (1.1) drive everything that follows.

1. **There is no cubic term.** The advective nonlinearity is invisible in the
   energy norm. Therefore `dE/dt` is *exactly homogeneous of degree 2* in `u`,
   and the set `{dE/dt > 0}` is a **cone**: if the energy method fails at some
   `u`, it fails equally at `s u` for every amplitude `s > 0`. So the failure of
   the *energy method* at `Re_E` is a statement about direction, not amplitude.

   This must not be over-read as a statement about the *flow*. Writing
   `u = a q` with `‖q‖ = 1`, energy neutrality `⟨q, F₂(q)⟩ = 0` makes the
   amplitude equation `ȧ = a⟨q, F₁q⟩` cubic-free, but the direction equation
   `q̇ = Π_{q⊥}[F₁q] + a Π_{q⊥}[F₂(q)]` carries an explicitly
   amplitude-proportional term. The cone is **not flow-invariant**, and the
   semiflow is not homogeneous: the nonlinearity does zero net work on `E` while
   still steering the state into and out of the cone at a rate set by the
   amplitude. That is precisely the opening a non-quadratic functional exploits
   — the quartic `V` of §3 is amplitude-dependent by design (§3.1).
2. **The pressure drops out.** This is special to the energy norm. Any other
   quadratic weighting of the components reintroduces pressure work (§3.2),
   and with it a genuinely cubic term.

## 1.3 The variational problem

From (1.1), `dE/dt < 0` for every non-zero admissible `u` if and only if
`Re < Re_E`, where

```
       1        − ∫ Σ(y) u v dV
     ───── = max ───────────────── ,        Σ(y) = U'(y) = 1,             (1.2)
      Re_E   u    ∫ |∇u|² dV
```

the maximum being over `u ∈ H¹₀,σ`, i.e. divergence-free fields vanishing on the
walls; it is attained, the production form being compact relative to the
dissipation form by Rellich. We carry a general real weight `Σ(y)` through the
whole computation, because the quartic Lyapunov functional of §3 requires
exactly the same eigenvalue problem with a *modified* `Σ`.

**Dependence on the box.** Strictly, `Re_E = Re_E(L_x, L_z)`: a finite periodic
box admits only the discrete wavenumbers `α ∈ (2π/L_x)ℤ`, `β ∈ (2π/L_z)ℤ`, so
its threshold is at least as large as the infimum over *all* real `(α, β)`. The
number reported below is that infimum — the worst case over all box sizes, and
the relevant value for boxes large enough to contain the critical wavelength.
Any finite box does at least as well.

Writing the production in terms of the rate-of-strain tensor
`S^U_ij = ½(∂_j U_i + ∂_i U_j)` — for `U = (y,0,0)` the only nonzero entries are
`S^U_12 = S^U_21 = ½` — gives the equivalent form

```
     − ∫ u v dV = − ∫ u_i u_j S^U_ij dV,                                  (1.3)
```

which is the form that generalises in §3.

### Euler–Lagrange equations

Maximising `−∫Σ u v` subject to `∫|∇u|² = 1` and `∇·u = 0`, with multipliers
`Λ = 1/Re_E` and `p`, gives

```
2Λ Δu − ∇p = (Σ v, Σ u, 0)ᵀ,      ∇·u = 0,      u|_{y=±1} = 0.           (1.4)
```

## 1.4 Fourier reduction and the divergence-free variables

The quadratic forms in (1.2) are invariant under translations in `x` and `z`,
and the base flow depends only on `y`; the problem therefore decouples over
Fourier modes `exp(i(αx + βz))`. For `u = Re[ û(y) e^{i(αx+βz)} ]` the `x,z`
average of a product of two such fields is `½ Re[â conj(b̂)]`, and this factor
`½` cancels between numerator and denominator of (1.2). So for each `(α, β)`

```
       1          − ∫₋₁¹ Σ(y) Re[ û conj(v̂) ] dy
   ──────────  = max ───────────────────────────────────── .              (1.5)
   Re_E(α,β)      ∫₋₁¹ ( |Dû|² + k²|û|² + … ) dy
```

with `k² = α² + β²`, and `Re_E = min_{α,β} Re_E(α,β)`.

The divergence constraint is eliminated *exactly* by the Orr–Sommerfeld/Squire
variables: the wall-normal velocity `v̂` and the wall-normal vorticity
`η̂ = iβ û − iα ŵ`. For `k² > 0`,

```
û = (i/k²)(α D v̂ − β η̂),        ŵ = (i/k²)(β D v̂ + α η̂).                 (1.6)
```

A one-line computation confirms `iα û + D v̂ + iβ ŵ = 0` for *any* `(v̂, η̂)`, so
the pair is unconstrained; the boundary conditions become `v̂ = D v̂ = 0` and
`η̂ = 0` at `y = ±1`. In these variables, writing `𝒟[û] = Σ_i ∫ (|Dû_i|² + k²|û_i|²) dy` for the modal
dissipation integral (so that the physical `∫|∇u|²dV` equals `(L_xL_z/2)𝒟[û]`,
and likewise `−∫ Σ u v dV = (L_xL_z/2)·(−∫Σ Re[û conj(v̂)] dy)` — the same factor
on both, so it cancels in the Rayleigh quotient):

```
k² 𝒟[û] = ∫ ( |D²v̂|² + 2k²|Dv̂|² + k⁴|v̂|² + |Dη̂|² + k²|η̂|² ) dy          (1.7)

k² ( −∫ Σ Re[û conj(v̂)] dy ) = ∫ Σ Im[ (α D v̂ − β η̂) conj(v̂) ] dy       (1.8)
```

Both (1.6)–(1.8) are **pointwise algebraic identities in `y`**, valid for
arbitrary `(v̂, η̂)`: no integration by parts and no boundary condition is used in
deriving them. (Boundary conditions enter elsewhere — in `⟨u,Δu⟩ = −∫|∇u|²`, and
in the §1.6 collapse to `‖(D²−β²)v‖²`.) Both sides of (1.7)–(1.8) carry the same
factor `1/k²`, so the Rayleigh quotient of (1.5) is the ratio of the right-hand
sides of (1.8) and (1.7). The mode `k = 0` need not
be considered: continuity plus `v̂(±1) = 0` forces `v̂ ≡ 0` there, hence zero
production.

## 1.5 Discretisation

`src/couette/basis.py` builds Galerkin bases in which every function satisfies
the boundary conditions exactly:

```
v-basis:   φ_n(y) = (1 − y²)² T_n(y)      (φ = φ' = 0 at y = ±1)
η-basis:   ψ_n(y) = (1 − y²)  T_n(y)      (ψ      = 0 at y = ±1)
```

All inner products are Gauss–Legendre quadratures with enough nodes to be exact
for the polynomial integrands, so the assembled matrices are Hermitian to
round-off. `(1.8)` becomes a Hermitian matrix `A` and `(1.7)` a Hermitian
positive-definite `B`, and

```
        1/Re_E(α, β) = λ_max(A, B)
```

is obtained from a Hermitian-definite generalised eigensolve. Two details:

* `Im[Dv̂ conj(v̂)]` contributes `(α/2i)(G − Gᵀ)` with `G_mn = ∫ Σ φ_m Dφ_n`;
  for `Σ ≡ 1` the matrix `G` is antisymmetric and this reduces to `−iαG`.
* `−β Im[η̂ conj(v̂)]` contributes the off-diagonal blocks `+(iβ/2)M` and
  `−(iβ/2)Mᵀ` with `M_mn = ∫ Σ φ_m ψ_n`.

## 1.6 The reduced α = 0 problem, and its Rayleigh–Bénard twin

At `α = 0` **and `β ≠ 0`** continuity gives `ŵ = i D v̂ / β`, so all the
cross-stream motion is carried by `v̂` alone and the `(v, w)` dissipation
collapses to

```
∫ ( |Dv|² + β²|v|² + |Dw|² + β²|w|² ) dy = β⁻² ∫ | (D² − β²) v |² dy,
```

using `v = Dv = 0` at the walls. Choosing phases so that `û` and `v̂` are real,

```
    1            − ∫ u v dy
 ─────── = max ──────────────────────────────────────────────────────── .  (1.9)
 Re_E(β)  u,v   ∫ (u'² + β²u²) dy + β⁻² ∫ (v'' − β²v)² dy
```

over real `(u, v)` with `u(±1) = 0` and `v(±1) = v'(±1) = 0` — a class that is
not an extra assumption but is forced by `u = 0` at the walls together with
continuity (`v' = iβw` and `w(±1) = 0`). The `β⁻²` makes the degeneracy explicit:
`β = 0` is excluded, and indeed `Re_E(β) ~ 17.79/β → ∞` as `β → 0`, so nothing is
lost.

With `R := Re_E/2` the Euler–Lagrange equations of (1.9) are

```
(D² − β²) u = R v,        (D² − β²)² v = − R β² u,
```

so `u` satisfies the sixth-order problem

```
(D² − β²)³ u + R² β² u = 0,
u = (D² − β²)u = D(D² − β²)u = 0   at y = ±1.                            (1.10)
```

**This is the Rayleigh–Bénard rigid–rigid problem.** Mapping `y ∈ [−1,1]` onto a
unit-depth layer via `y = 2ηc − 1` sends `D_y = ½ D_η`, and (1.10) becomes
`(D_η² − a²)³ W + Ra a² W = 0` with

```
        a = 2β,        Ra = 16 R² = 4 Re_E².                             (1.11)
```

Hence the textbook values `Ra_c = 1707.762`, `a_c = 3.117` predict

```
Re_E = √(Ra_c)/2 ≈ 20.6625,        β_c = a_c/2 ≈ 1.5585.
```

(The rounded `a_c = 3.117` gives `β_c = 1.5585`; the sharper
`a_c = 3.1163236` gives `β_c = 1.5581618`, which is the value computed in §1.7.
`Ra_c` is far less sensitive than `a_c`, because `Ra(a)` is stationary at its
minimum while `a_c` itself is not.)

The critical mode is even in `y`, so setting `u = Σ_j A_j cosh(s_j y)` with
`s_j² = β² + q_j` and `q_j³ = −R²β²` reduces (1.10) to a 3×3 boundary
determinant. One subtlety: the three roots `{−m, m e^{iπ/3}, m e^{−iπ/3}}` are
closed under conjugation, so conjugating the determinant swaps two columns and
flips its sign — **the determinant is purely imaginary** for real `(R, β)`, and
it is its imaginary part that must be driven to zero.

## 1.7 Results

`scripts/01_reproduce_ReE.py` runs three mutually independent computations.

| route | `Re_E` | `β_c` |
|---|---|---|
| Galerkin, full 3D, `(v, η)` variables | 20.662537217781 | 1.5581617672 |
| analytic 3×3 determinant, mpmath | 20.6625372177808694 | 1.55816177741056 |
| implied Rayleigh–Bénard | `Ra_c = 1707.7617771` | `a_c = 3.11632355` |

The two computations of `Re_E` agree to `5×10⁻¹⁴`, and the implied
Rayleigh–Bénard values reproduce the independently-known `1707.762` and `3.117`.

```
Re_E = 20.6625372177809      at   (α_c, β_c) = (0, 1.5581617774)
```

The value quoted in the literature as "20.65" or "20.7" is this number rounded.

**Convergence.** The Galerkin eigenvalue is already converged to 13 digits at
`N = 10` and stays flat to `N = 80` — expected, since the exact eigenfunctions
are entire and the bases are polynomial.

**`α = 0` is optimal.** `Re_E(α, β_opt(α))` increases monotonically with `α`:

| α | 0 | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 | 1.2 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| `Re_E` | 20.6625 | 20.6711 | 20.6970 | 20.8002 | 21.2122 | 22.8484 | 25.5425 | 33.9839 |

So the energy method first fails on a **streamwise-independent** disturbance.
Section 2 shows why that is the most favourable possible news for the programme
of beating `Re_E`.

**A fourth cross-check: Orr's two-dimensional value.** Restricting to `β = 0`
(purely two-dimensional disturbances) the same solver gives

```
Re_E(2D) = 44.3035467005     at   α_c = 1.8933674203,
```

reproducing the value ≈ 44.3 that Orr obtained in 1907 — an independent
literature number that shares no machinery with the Rayleigh–Bénard check of
§1.6, since it lives on the opposite edge of the wavenumber plane. It is also
more than twice the 3D threshold, which is the classical statement that the
energy method's worst case is genuinely three-dimensional.
