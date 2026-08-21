# 0. Prior art — what is already known

Compiled from a literature search; **arXiv and most scholarly hosts are blocked
by this environment's egress policy**, so the entries below come from search
abstracts and metadata, not from reading the papers. Everything here should be
verified against the actual PDFs before being relied on. Where a claim could not
be confirmed, it says so.

## 0.1 The SOS / non-quadratic-Lyapunov line

| | |
|---|---|
| **Goulart & Chernyshenko (2012)**, *Global stability analysis of fluid flows using sum-of-squares*, Physica D | origin of the SOS approach to global fluid stability |
| **Fuentes, Goluskin & Chernyshenko (2022)**, PRL **128**, 204502, [arXiv:1911.09079](https://arxiv.org/abs/1911.09079) | polynomial optimisation → non-quadratic Lyapunov functions. Verifies global stability of **2-D plane Couette** above the energy threshold Orr found in 1907. Billed as **"the first global stability result for any flow that surpasses the energy method"** |
| **Iligaray, Aballay & Fuentes (2026)**, [arXiv:2604.23915](https://arxiv.org/abs/2604.23915) | *Improved global stability bounds for 2-D plane Poiseuille flow*. Quartic Lyapunov functionals via SOS. At the critical energy-stable streamwise length, `Re_E ≈ 87.59` → global stability to `Re ≈ 106.8` (**+22%**). Simplest successful mode set: **five modes** |
| **Darrow, Carlson & Goluskin (June 2026)**, [arXiv:2606.18232](https://arxiv.org/abs/2606.18232) | *Quartic Lyapunov functions for global fluid stability*. Identifies **the simplest class of non-quadratic Lyapunov functions for 2-D parallel shear flows: a three-parameter family of quartic functions.** Exploits shear-flow symmetries via complex-variable representations to shrink the problem; **refines key inequalities in previous works**; replaces expensive computational steps with analytical alternatives; proves stability over a *continuous range* of `Re`. Verifies 2-D plane Couette and 2-D Poiseuille |

**Everything in this line is two-dimensional.** The 3-D case is explicitly named
as future work — "similar statements are expected for analogous
three-dimensional flows, which are of obvious interest for future research" —
with computational bottlenecks (mode-set size) given as the barrier.

## 0.2 The generalised-energy line

**Kaiser, Tilgner & von Wahl**, *A Generalized Energy Functional for Plane
Couette Flow*, SIAM J. Math. Anal.
([doi](https://epubs.siam.org/doi/10.1137/S0036141004442604)). A generalised
energy functional giving **conditional** (finite-amplitude) nonlinear stability
for `Re < 177.2`. Note *conditional*, not global — it does not close the gap in
the sense this project means.

**They also proved that streamwise-independent perturbations of plane Couette
flow are `L²`-energy stable at any Reynolds number.** That is exactly the
statement derived in §2.3 of `docs/02_critical_mode.md`. **§2.3 is therefore not
new and must be attributed to them.** (The derivation there is independent and
still worth keeping as motivation, but the credit belongs to Kaiser–Tilgner–von
Wahl.)

## 0.3 The Joseph-vs-Orr threshold dispute

**Falsaperla, Giacobbe & Mulone**, incl. *Nonlinear monotone energy stability of
plane shear flows: Joseph or Orr critical thresholds?*, SIAM J. Appl. Math.
([arXiv:2304.11416](https://arxiv.org/abs/2304.11416)). They conjecture that the
Reynolds–Orr maximisation should be restricted to a subspace of kinematically
admissible perturbations, which moves the critical value from Joseph's
streamwise `20.65` to Orr's spanwise `44.3` (a "Squire theorem for nonlinear
monotone energy stability"), and connect this to their earlier work on tilted
rolls.

This is a **conjecture about restricting the admissible class**, and it is
contested. It does not change the classical full-space Reynolds–Orr value: over
all of `H¹₀,σ` the maximum is unambiguously `Re_E = 20.6625372178`, which §1.7
confirms by three independent routes. Treat `44.3` as the *two-dimensional*
threshold (§1.7 computes `Re_E(2D) = 44.3035467`), not as a replacement for
`20.66`.

## 0.4 Consequences for this project

**Good news — the target is genuinely open.** No published result certifies
global stability of **full 3-D** plane Couette flow at any `Re > Re_E = 20.66`.
Every success in the SOS line is 2-D, and 3-D is stated as future work.

**Sobering news — the method is not new.** Quartic Lyapunov functionals
constructed by polynomial optimisation are the established state of the art, and
a June 2026 paper has already identified "the simplest class" of them for 2-D
parallel shear flows as a three-parameter quartic family. Whether the
**shear-shift theorem** of §3.3 — that adding `2E⟨W,u⟩` is exactly a shift of the
base-flow strain in the quartic term — appears there could not be determined,
because the paper is unreadable from this environment. **Check this before
claiming any of §3.3–§3.5 as new.**

**Directly actionable.** Darrow–Carlson–Goluskin advertise exactly the two things
this project is currently short of:

1. *"refines key inequalities in previous works"* — the shortfall in §3.10 is a
   factor of ~27 coming from a chain of lossy inequalities, above all the `L^∞`
   cubic estimate. Their refined inequalities may be directly reusable and
   should be read **before** re-deriving anything.
2. *"exploits symmetries of shear flows via complex variable representations,
   greatly reducing the problem size"* — the stated barrier to 3-D is mode-set
   size, so this is the lever that makes 3-D reachable.

Their result that only **three parameters** (2-D) and Fuentes et al.'s that only
**five modes** suffice is also encouraging for §3.10's finding that **rank two**
is optimal here: small mode sets are apparently enough, so the 3-D problem may
not be as computationally hopeless as "3-D" suggests.

## 0.5 Reading list, in priority order

1. Darrow, Carlson & Goluskin 2026 (2606.18232) — the refined inequalities and
   the three-parameter family. **Read first.**
2. Fuentes, Goluskin & Chernyshenko 2022 (1911.09079) — the original framework
   and how the infinite-dimensional tail is handled rigorously.
3. Iligaray, Aballay & Fuentes 2026 (2604.23915) — mode-set selection, and the
   continuation-in-`Re` machinery.
4. Kaiser, Tilgner & von Wahl — the generalised energy functional, and their
   streamwise-independent result (§2.3 credit).
5. Goulart & Chernyshenko 2012 — origins.
