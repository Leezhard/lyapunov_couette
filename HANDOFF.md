# Handoff — continuing this work locally

## Where things stand

`README.md` has the full result summary; `docs/00_prior_art.md` has the literature
position. In one paragraph: the Reynolds–Orr machinery is built and validated to
13 digits by four independent routes; the quartic framework
`V = E² + 2E⟨W,u⟩ + Q` is derived with an exact shear-shift theorem and an exact
negativity criterion; **no `Re > Re_E` is certified yet** — the sufficient
conditions fall short by a factor of ~27, traced to three identified lossy
inequalities.

```bash
git clone https://github.com/Leezhard/lyapunov_couette
cd lyapunov_couette
git checkout claude/couette-stability-gap-dr0mvt
pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest tests/ -q          # expect: 59 passed
```

Scripts write to `results/`:

```bash
PYTHONPATH=src python3 scripts/01_reproduce_ReE.py      # Re_E and its validation
PYTHONPATH=src python3 scripts/02_critical_mode.py      # critical mode structure
PYTHONPATH=src python3 scripts/03_shear_shift.py        # the shear-shifted problem
PYTHONPATH=src python3 scripts/04_feasibility.py        # the master inequality
PYTHONPATH=src python3 scripts/05_quadratic_form.py     # how large c2/K can be made
```

## First thing to do locally: fetch the papers

This was impossible from the cloud container — its egress proxy blocks every
scholarly host. Locally it is trivial for the arXiv ones (all open access):

```bash
mkdir -p papers && cd papers
curl -L -o darrow2026.pdf    https://arxiv.org/pdf/2606.18232   # Tier 1
curl -L -o fuentes2022.pdf   https://arxiv.org/pdf/1911.09079   # Tier 1
curl -L -o iligaray2026.pdf  https://arxiv.org/pdf/2604.23915   # Tier 2
curl -L -o mulone2023.pdf    https://arxiv.org/pdf/2304.11416   # Tier 3
# LaTeX source is better than PDF for the equation-heavy ones:
curl -L -o darrow2026.tar.gz  https://arxiv.org/e-print/2606.18232
curl -L -o fuentes2022.tar.gz https://arxiv.org/e-print/1911.09079
```

Paywalled, needs Cambridge access (IP-based auth often works from `curl` on
campus; SSO/cookie-based does not — download in the browser then):

* Kaiser, Tilgner & von Wahl, *A Generalized Energy Functional for Plane Couette
  Flow*, SIAM J. Math. Anal., doi 10.1137/S0036141004442604
* Goulart & Chernyshenko, *Global stability analysis of fluid flows using
  sum-of-squares*, Physica D (2012)
* Chernyshenko et al., *Polynomial sum of squares in fluid dynamics: a review
  with a look ahead*, Phil. Trans. R. Soc. A (2013), doi 10.1098/rsta.2013.0350

`papers/` is gitignored — do not commit publisher PDFs.

## The three questions the papers must answer

1. **Is the shear-shift theorem (§3.3) already known?** Darrow–Carlson–Goluskin
   2026 identify "the simplest class of non-quadratic Lyapunov functions for 2-D
   parallel shear flows: a three-parameter family of quartic functions". Compare
   that family against `V = E² + 2E⟨W,u⟩ + Q` and against the statement that
   adding `2E⟨W,u⟩` shifts the base-flow strain `S^U → S^U − S^W`. **This is the
   single cheapest, highest-value check — one equation settles it.**
2. **What are their refined inequalities?** They advertise "refines key
   inequalities in previous works". The ~27× shortfall in §3.10 is a chain of
   lossy inequalities, the crudest being the `L^∞` cubic estimate. If their
   refinement applies, it may close most of the gap directly.
3. **How is the infinite-dimensional tail handled rigorously?** (Fuentes et al.
   2022.) This is what makes a certificate apply to the PDE rather than to a
   Galerkin truncation, and it is the part this project has *not* done.

## Next technical steps, in order

1. **(Blocked on the papers.)** Sharpen the cubic bound. If their refinement does
   not apply, the fallback derived here is: the sharp constant is itself a
   Reynolds–Orr problem. For `𝒬̃ = Σ_j μ_j e_j ⊗ e_j`,
   `⟨e_j,(u·∇)u⟩ = −∫ u_i u_k S^{e_j}_{ik} dV` by the same identity (3.6), so
   ```
   κ_j = max_u |∫ u_i u_k S^{e_j}_{ik}| / ∫|∇u|²  =  1 / Re_E[S^{e_j}]
   ```
   giving `|2⟨𝒬̃u,F₂(u)⟩| ≤ 2(Σ_j μ_j² κ_j²)^{1/2} ‖u‖ D` — the right powers, with
   no `L^∞` norm and no Poincaré factor. Expected gain ≈ 8× (2× from strain vs
   full gradient, 4.2× from Reynolds–Orr sharpness vs Poincaré). Since `e_j` sits
   at `(0, β_c)`, `S^{e_j}` depends on `y` *and* `z`, so this is no longer
   wavenumber-diagonal — it becomes a Bloch/Floquet problem in `z` with triad
   coupling. That triad structure is also the natural home for the rigorous tail
   bound.
2. **(Not blocked.)** Global `c₂` across *all* Fourier blocks. The 0.183 in §3.10
   is the critical block alone; the certificate needs one `c₂` valid everywhere,
   which can only be lower. Cheap with the existing code, and it calibrates the
   target.
3. Replace the direction-by-direction Young inequality (3.15) by a **joint
   SDP/SOS** condition on the Gram matrices — removes the loss from worst-casing
   `a₂, a₃, a₄` separately. Needs cvxpy + a solver (MOSEK if available).
4. Continuation in `Re` (20.7 → 21 → 22 …). Note §3.9's table: the regrouping
   helps only while the shift is small, so small steps are the right strategy.
5. Only once a candidate closes: rigorous certification — interval arithmetic or
   rational reconstruction.

## Things to be careful about

* `Re_E = 20.6625372178` is the **3-D** value; `44.3035467` is the **2-D** (β = 0)
  value and is what the Fuentes/Darrow line improves upon. Do not conflate them,
  and do not confuse either with the Falsaperla–Giacobbe–Mulone conjecture that
  also lands on 44.3 by restricting the admissible class.
* §2.3 is due to Kaiser–Tilgner–von Wahl, not to this work.
* An earlier version of (3.18ʹ) carried a spurious factor of 4;
  `tests/test_feasibility_algebra.py` now pins the correct threshold
  `ĉ₂ c₄ > b₀ B̂`. Keep that test.
* Every number here is double precision (plus one mpmath cross-check). Nothing is
  certified in interval arithmetic.
