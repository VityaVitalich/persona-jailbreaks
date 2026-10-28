# A1 — Projecting `w_raw` onto the persona span: the math, precisely

*(Notation: column vectors in ℝ⁴⁰⁹⁶ = activation space at layer 16, last prompt token, pre-generation. All data below is the held-out HarmBench test split — behaviors AND jailbreak templates disjoint from training, seed 41, n = 252.)*

## 1. The two objects

**The detector direction `w_raw`.** `classify_jailbreak_raw_activations.py` fits a logistic
regression on standardized raw activations and maps the weights back to original activation
space (`w_orig = w_scaled / scaler.scale_`, then unit-normed). A linear detector scores a
response by the dot product

    s(x) = w_raw · x

and AUC measures how well this score ranks jailbroken above non-jailbroken responses.
Because standardization is affine and AUC only depends on the *ranking* of scores,
`AUC(w_raw · x)` is exactly the fitted classifier's test AUC: **0.8387**.

**The persona vectors.** 229 trait vectors v₁…v₂₂₉ (difference-of-means directions, same
layer), stacked as rows of

    V ∈ ℝ^{229×4096}

Their span is every linear combination of personas:

    span(V) = { Σᵢ cᵢ vᵢ : c ∈ ℝ²²⁹ }

— a 229-dimensional subspace of the 4096-dimensional activation space.

## 2. The projection = the reconstruction

We want the part of `w_raw` that personas can express. Three steps: build an
orthonormal basis of the persona span (2.1), turn it into a projector (2.2), and see
why the projection is exactly the least-squares reconstruction (2.3).

### 2.1 Why Q is an orthonormal basis of span(V)

The persona vectors v₁,…,v₂₂₉ span the subspace but are an awkward basis: they are
mutually correlated (cosines up to 0.97) and not orthogonal, so you cannot read off
"the component of w inside the span" from dot products with them directly — each
dot product double-counts what other personas already cover. Gram–Schmidt
(numerically: the QR decomposition of Vᵀ) fixes this. Process the personas in order,
keeping only what is genuinely new about each:

    q₁ = v₁ / ‖v₁‖
    q̃₂ = v₂ − (q₁·v₂) q₁          ← subtract what q₁ already covers
    q₂ = q̃₂ / ‖q̃₂‖
    ⋮
    q̃ₖ = vₖ − Σ_{j<k} (qⱼ·vₖ) qⱼ   ← subtract what q₁…q_{k−1} already cover
    qₖ = q̃ₖ / ‖q̃ₖ‖

By construction the qⱼ are orthonormal (qᵢ·qⱼ = 1 if i=j, else 0); every qₖ is a
combination of v₁…vₖ, and — inverting the recursion — every vₖ is a combination of
q₁…qₖ. Each set is expressible in the other, so they span the *same* subspace:

    span(q₁,…,q₂₂₉) = span(v₁,…,v₂₂₉) = span(V)

Stacking the qⱼ as columns gives Q ∈ ℝ^{4096×229} with QᵀQ = I₂₂₉; the bookkeeping
coefficients form the upper-triangular R in Vᵀ = QR. (This needs V to have full rank
229 so no q̃ₖ collapses to zero — true here, confirmed by `lstsq`'s reported rank.)

### 2.2 Why P = QQᵀ is the orthogonal projector

For any w ∈ ℝ⁴⁰⁹⁶,

    P w = Q (Qᵀ w) = Σⱼ (qⱼ · w) qⱼ

Read right to left: Qᵀw computes the 229 coordinates of w along the orthonormal
basis directions, and multiplying by Q rebuilds the vector having exactly those
coordinates and nothing else. (Rank-1 sanity check: for a single unit vector q,
P = qqᵀ and Pw = (q·w)q — the familiar "component of w along q". QQᵀ = Σⱼ qⱼqⱼᵀ just
sums this over an orthonormal set, which is legitimate *only because* the qⱼ don't
overlap — that's what 2.1 bought us.)

The three properties that make P "the projection onto span(V)":

- **Lands in the span:** Pw is a combination of the qⱼ, hence of the personas.
- **Idempotent:** P² = Q(QᵀQ)Qᵀ = QQᵀ = P. Projecting twice = projecting once, and
  any vector already in the span is left exactly where it is.
- **The remainder is orthogonal to the span:** Qᵀ(w − Pw) = Qᵀw − (QᵀQ)Qᵀw = 0, so
  r = w − Pw has zero dot product with every qⱼ — hence with every persona vector,
  hence with everything in the span.

### 2.3 Why Pw is the optimal least-squares reconstruction

The reconstruction problem asks for coefficients c ∈ ℝ²²⁹ minimizing

    ‖w_raw − Σᵢ cᵢ vᵢ‖²  =  ‖w_raw − Vᵀc‖²

As c ranges over all of ℝ²²⁹, the candidate u = Vᵀc ranges over exactly span(V), so
the problem is geometric: *find the point of the span closest to w_raw*. Claim: it is
ŵ = P w_raw, uniquely.

Proof by Pythagoras. Take any u ∈ span(V) and split the error:

    w_raw − u = (w_raw − ŵ) + (ŵ − u)
                └── = r, ⊥ span ──┘   └─ ∈ span ─┘

The first piece is orthogonal to the span (§2.2, third bullet); the second lies
inside it; so they are orthogonal to each other and

    ‖w_raw − u‖² = ‖r‖² + ‖ŵ − u‖²  ≥  ‖r‖²

with equality iff u = ŵ. No point of the span beats the projection.
(Calculus route to the same place: setting the gradient of ‖w − Vᵀc‖² to zero gives
the normal equations V(w − Vᵀc) = 0, i.e. "residual ⊥ every persona vector" — the
identical orthogonality condition. `np.linalg.lstsq` solves exactly this, which is
why the April reconstruction script and the projector formulation produce the same ŵ;
the explicit recipe is c* = R⁻¹Qᵀw_raw.)

So "projection" and "reconstruction" are one operation with two descriptions, and we
get the orthogonal split ŵ ⊥ r with (Pythagoras again)

       ‖w_raw‖² = ‖ŵ‖² + ‖r‖²

The old April analysis computed exactly this split and reported the *geometric* size of
the reconstruction:

    R² = ‖ŵ‖² / ‖w_raw‖² = 0.087        (so ‖r‖²/‖w_raw‖² = 0.913)

and concluded "reconstruction failed: personas explain 8.7% of w_raw."
(Equivalently cos∠(w_raw, ŵ) = √R² ≈ 0.30.)

## 3. What A1 measures instead

The score of *any* response decomposes along the same split:

    w_raw · x  =  ŵ · x  +  r · x
                 (persona-expressible part)   (persona-orthogonal part)

A1 asks: which part carries the detection? Score the held-out test set with each
direction separately (AUC is scale-invariant, so renormalizing ŵ changes nothing):

| direction | fraction of ‖w_raw‖² | test AUC |
|---|---|---|
| `w_raw` (full) | 100% | 0.839 |
| **ŵ = P w_raw (pure persona combination)** | **8.7%** | **0.841** |
| r = w_raw − ŵ (orthogonal to every persona) | 91.3% | 0.784 |
| random directions (20 draws) | — | 0.51 ± 0.09 |

**Result:** the persona-expressible 8.7% of `w_raw` detects jailbreaks *as well as the
whole vector*. There exists an explicit weighted sum of the 229 named persona vectors
whose dot-product with activations matches the raw-activation detector on data it has
never seen.

## 4. Why R² is tiny while AUC is preserved (the resolution of the paradox)

AUC depends only on differences of scores between test points:
w · (x_i − x_j). So only the component of w lying in the subspace where **the data
actually varies** affects AUC. Components of w along directions with (near-)zero data
variance contribute (near-)zero to every score difference — but they still count fully
toward ‖w‖².

With n ≈ 1.6k training samples in d = 4096 dimensions, logistic regression is massively
underdetermined: the fit places large weight mass along low-variance / noise directions
(it costs the training loss nothing). That mass dominates the Euclidean norm of `w_raw`
and therefore dominates R² — but it is prediction-irrelevant.

**Toy example.** Let data vary only along e₁ (signal: class means differ) and e₂
(noise variance), and let the fitted weight be w = e₁ + 10·e₃ where e₃ is a direction of
zero data variance. Take the persona span = span(e₁). Then

    R² = ‖P w‖²/‖w‖² = 1/101 ≈ 1%      (reconstruction "fails")

yet for every data point x (which has no e₃ component), w·x = (P w)·x exactly: the
projected direction makes **identical predictions**. Geometric R² on a weight vector can
be arbitrarily small while the functional content is fully preserved.

**Empirical proof this is what happens here:** the repo contains two independent fits of
the same detector — plain `w_raw` (AUC 0.839) and the 10-seed-averaged
`stable_hyperplane` (AUC 0.833). They agree almost perfectly in function and yet
cos(plain, stable) = **0.093** — nearly orthogonal as vectors. If `w_raw` cannot even be
"reconstructed" from a refit of itself, Euclidean R² against personas was never a valid
test. (This is the filter-vs-pattern distinction of Haufe et al. 2014: weight vectors of
linear models are *filters*, not interpretable directions; only their action on the data
manifold is meaningful.)

## 5. Why is the residual itself still at AUC 0.784?

At first glance r should be dead: it is orthogonal to every persona vector, and we
just argued the persona span carries the detection signal. The resolution:
**the jailbreak signal in activation space is redundant — spread across many
correlated directions — and some of it lies outside the persona span.**

Make "signal" concrete: let Δμ = mean activation of jailbroken responses minus mean
activation of non-jailbroken ones. For any scoring direction u, the between-class gap
in mean scores is u·Δμ; to first order (two-Gaussian picture), a direction with
u·Δμ = 0 separates the classes at chance. The residual scores 0.784 ≫ 0.5, so
necessarily r·Δμ ≠ 0 — i.e. **Δμ itself sticks partly out of the persona span**.
Out-of-span jailbreak signal exists.

Why doesn't that contradict A1? Because *standalone performance* and *marginal
contribution* are different questions — exactly like two strongly correlated features
in a regression: each predicts well alone, either is redundant given the other. The
signal r sees is largely a **copy** of signal ŵ already sees. The proof is in the
table: adding r back to ŵ reconstitutes the full detector (w_raw = ŵ + r), and that
sum scores 0.839 — no better than ŵ's 0.841 alone. A1's claim is precisely this
marginal statement: given the persona-span component, the other 91% of the vector
contributes nothing further.

This redundancy is independently documented in the repo: the top individual traits
correlate 0.77–0.97 with one another, and a single trait direction (egalitarian)
matches the full 229-trait detector in-distribution. When class signal is smeared
across that many correlated directions, any large-mass residual will overlap some
of it.

Note what r's 0.784 forecasts: the necessity test (C) will **not** come out as
"the signal lives exclusively in the persona span" — deleting the span from the data
will leave a detectable residue (we already know the fixed detector only drops to
0.784). The honest target claim is "concentrated and sufficient", not "exclusive",
and A3/C exist to put numbers on exactly that.

## 6. What A1 proves — and what it deliberately does not

**Proven (sufficiency of expression):** every bit of jailbreak-detection ability in the
raw-activation detector can be rewritten as a linear combination of persona vectors:
∃ c ∈ ℝ²²⁹ with AUC(Σᵢ cᵢ vᵢ · x) = 0.841 ≥ AUC(w_raw · x) = 0.839. The persona-orthogonal
residual — 91% of the vector's mass — adds nothing on top.

**Not proven (by design):**
- *Specificity* — that persona vectors are special. A 229-dim subspace is large; random or
  otherwise-constructed subspaces might also suffice (Johnson–Lindenstrauss makes this a
  real threat at k=229). → tests B1/B2: matched random subspaces, and dose–response at
  small k, where personas and random genuinely separate. B2 runs in two bases: top-k
  **pure personas** (greedy selection — keeps names) and top-k **persona-PCs** (PCA of the
  229 persona vectors — potentially more compact, loses individual names).
- *Necessity* — that the signal lives *only* there. → test C: ablate span(V) from the
  activations (x → (I−P)x; fixed detector drops 0.839 → 0.784) vs. matched random
  ablation, plus the stringent retrain-after-ablation variant.

## 7. Where A1 sits in the overall proof

The full argument for "jailbreaks live in the persona subspace" has three pillars:

1. **Sufficient** — the persona subspace contains the full detection signal (A1 above;
   A3 = the same statement for the class-mean-difference pattern Δμ, the noise-free
   geometric version).
2. **Specific** — matched-size random subspaces don't do this, and few named persona
   dimensions suffice (B1, B2).
3. **Causal** — steering constrained to the persona span reproduces jailbreaks;
   orthogonal-complement steering doesn't (D — parked for now, needs GPU).

Parked caveat (noted in the paper): even if specific vs. *random* subspaces, a semantic
control — same extraction pipeline on non-persona contrasts (topics, styles) — is needed
to claim personas beat any *meaningful* basis (the abandoned Null 2 / B3).

---
*Scripts: `wraw_reconstruction_grid.py`, `wraw_projection_auc.py` (session scratchpad);
inputs mirrored under `full_trait_output/` (hyperplanes, trait sets, HarmBench
activations). Test-set replication reproduces the saved AUC 0.8387 exactly.*
