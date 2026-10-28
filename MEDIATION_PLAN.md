# Does persona space mediate jailbreaks?

This plan turns the one-page note, `~/Downloads/Does persona space mediate jailbreaks_.pdf`, into an experiment in our setup. We use Llama-3.1-8B-Instruct, layer 16, and the filtered trait vectors in `full_trait_output/trait_matrix_layer16.npy`. Nothing has been run yet.

## 1. The question

Strong and weak jailbreak prompts put the model in different internal states. Part of that difference lies in persona space, the subspace spanned by our trait vectors. We ask how much of the jailbreak works through that part.

We run the model on a weak jailbreak prompt. At one point in the forward pass, we replace only the persona part of its hidden state. We take the replacement from the hidden state produced by a strong jailbreak prompt. Everything else stays the same. Then we count how often the model complies.

We measure how much of the gap between weak and strong prompts the patch reproduces. If it reproduces most of the gap, persona space carries most of the jailbreak, even without flipping every prompt. If it reproduces only a small share, persona space plays a minor role. Our main measure is the share of the effect that goes through persona space, defined in §3.

The note defines three variables:

- **X** is the treatment. A weak jailbreak has X=0, and a strong jailbreak has X=1. We label whole jailbreak types from a preprocessing ASR eval, never single prompts.
- **M** is the mediator. It consists of the hidden state's persona coordinates, its projections onto the trait vectors.
- **Y** is the outcome. It records whether the model was jailbroken.

The note says 240 trait vectors. The paper uses the 229 that pass the quality filter, and so do we.

### Relation to activation patching

We use activation patching restricted to a subspace. Standard activation patching has a base run and a counterfactual or source run. Here, the weak prompt $x_0$ provides the base run, and the strong prompt $x_1$ provides the source run. Activation patching copies an activation from the source run into the base run and checks whether the output changes.

We copy only the activation component in persona space and keep the rest from the base run. In the interpretability literature, Geiger et al. call this a subspace interchange intervention. Distributed Alignment Search uses the same operation with a learned subspace instead of a given one. Copying the whole activation is ordinary activation patching. We also run it as a reference, as described in §3.

## 2. The intervention

Let $h_0$ be the hidden state on the weak prompt $x_0$ and $h_1$ the state on the strong prompt $x_1$. Both states come from the same layer and token position. Stack the trait vectors as the rows of a matrix $T$. Then $Th$ is the vector of persona coordinates of $h$. The note asks for the state closest to $h_0$ whose persona coordinates equal those of $h_1$:

$$
h^* = \arg\min_h \|h - h_0\|^2 \quad \text{subject to} \quad T h = T h_1 .
$$

Let $P$ be the orthogonal projection onto the span of the trait vectors. As in the paper's `sec:directions`, we get it from a QR decomposition of $T^\top$: $T^\top = QR$, then $P = QQ^\top$. The solution is

$$
h^* = h_0 + P\,(h_1 - h_0) = (I - P)\,h_0 + P\,h_1 .
$$

The state $h^*$ keeps the part of $h_0$ outside persona space. It takes the part inside persona space from $h_1$.

### Why this is the optimum

Write $h = h_0 + d$. The constraint becomes $T d = T(h_1 - h_0)$. We want the shortest $d$ that satisfies it.

Split $d$ into its persona-space and orthogonal parts: $d = P d + (I-P) d$. The constraint only sees the first part because $T$ annihilates everything orthogonal to its rows. This fixes the first part: $P d = P(h_1 - h_0)$. The second part is free. By Pythagoras, $\|d\|^2 = \|Pd\|^2 + \|(I-P)d\|^2$. The length is smallest when the free part is zero, which gives $d = P(h_1 - h_0)$.

### Choice of $T$

Only the span of the trait vectors enters $h^*$, through $P$. Normalising the vectors, rescaling them, or replacing them with all their principal components gives exactly the same $h^*$.

Two variants change $h^*$, and both are useful extensions:

- A smaller persona subspace keeps only the top $k$ principal components of $T$. This tests how few persona dimensions can carry the effect.
- A searched subspace maximises the effect through learning, as Distributed Alignment Search does, but stays inside the trait-vector span to preserve the persona interpretation. A learned subspace fits its training data, so we must learn it on calibration behaviours and evaluate it on held-out behaviours, as described in §5.

### Patch site

Every Llama-3.1 chat prompt ends with the same five tokens, `<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`. The trait vectors and the paper's detector are defined at the last of these tokens. These tokens are identical in every prompt. Position $j$ in $x_0$ therefore corresponds exactly to position $j$ in $x_1$, despite their different prompt lengths.

We replace the state there with $h^*$ during the prompt pass. The model recomputes everything after that point, and generation proceeds normally from $x_0$.

We consider three versions of the patch site:

1. Only the last prompt token. This is closest to the note because the trait vectors are defined there.
2. All five closing tokens. This assumes the trait vectors also mean "persona" at the header tokens, which we have not checked.
3. The five tokens, plus a constant shift on every generated token. This combines patching with steering in case a prompt-only patch fades during generation.

We pick one before looking at any persona result, as described in §6.

### Layer

The main experiment uses layer 16, where the paper's detector and steering operate. We can easily vary the layer because trait vectors are already stored for every layer. A layer sweep with the same patch then shows where along depth the persona part starts to carry the jailbreak.

## 3. What we measure

### Notation

For a prompt $x$, let $f(x)$ be the model's greedy response. Let $f_{h}(x)$ be the response when we replace the state at the patch site with $h$. A judge $J$ returns 1 if a response is jailbroken and 0 otherwise. For a set of prompts or pairs, ASR is the average of $J$. We use the pair notation of §2: weak $x_0$ with state $h_0$, strong $x_1$ with state $h_1$.

### Mediation terms

Causal mediation analysis splits a treatment's effect into a part that passes through a mediator and a part that does not. Its basic object is the outcome under mixed conditions, $Y(x, M(x'))$. The treatment is set to $x$. The mediator takes the value it would naturally have under treatment $x'$.

Usually, we cannot observe this mix because a person cannot take and not take a drug at the same time. In a network, we can build it directly. We run prompt $x$ and patch in the persona coordinates from prompt $x'$.

| Quantity | Causal definition | In our setup | What it answers |
|---|---|---|---|
| Total effect (TE) | $Y(1, M(1)) - Y(0, M(0))$ | $\text{ASR}(f(x_1)) - \text{ASR}(f(x_0))$ | How much more often strong prompts jailbreak. This is the gap to explain. |
| Natural indirect effect (NIE) | $Y(0, M(1)) - Y(0, M(0))$ | $\text{ASR}(f_{h^*}(x_0)) - \text{ASR}(f(x_0))$ | Does giving the weak prompt the strong prompt's persona coordinates alone raise the ASR? This tests sufficiency. |
| Natural direct effect (NDE) | $Y(1, M(0)) - Y(0, M(0))$ | $\text{ASR}(f_{\bar h}(x_1)) - \text{ASR}(f(x_0))$, with $\bar h = h_1 + P(h_0 - h_1)$ | Does the strong prompt still work when we reset its persona coordinates to the weak prompt's? If not, persona coordinates are necessary. |
| Proportion mediated | $\text{NIE} / \text{TE}$ | | The share of the gap that goes through persona space. This is our main measure. |

This is why the note defines the NIE as $y^* - y_0$. "Indirect" means the effect that reaches the outcome only through the mediator. To isolate it, we hold the treatment at its baseline, the weak prompt. We switch only the mediator to the value the strong prompt produces.

"Natural" means we set the mediator to a value a real prompt produces, rather than an arbitrary one. Steering instead sets the persona coordinates to an arbitrary point chosen by $\alpha$ and a direction. That corresponds to a different quantity, the controlled direct effect.

NIE and NDE need not add up to TE. When persona and non-persona parts only work together, both can be small while TE is large. We therefore report both. The difference $\text{ASR}(f(x_1)) - \text{ASR}(f_{\bar h}(x_1)) = \text{TE} - \text{NDE}$ is the reverse counterpart of the NIE. It is the subspace-level version of the single-direction ablation in Appendix H.

### Full-state patch

Before measuring mediation, we check that the site matters. We replace the whole state with $h_1$, using ordinary activation patching: $\text{ASR}(f_{h_1}(x_0)) - \text{ASR}(f(x_0))$. If this barely moves the ASR, the jailbreak does not act through this site. A persona patch there cannot tell us anything. This is a precondition, not a result.

### Same-behaviour and different-behaviour pairs

When $x_0$ and $x_1$ wrap the same harmful request, the quantities above are the natural effects, defined per request. When they wrap different requests, the persona coordinates come from another request's prompt, as in the note's random pairing. The estimate is then a closely related population-level version, the interventional indirect effect. We report the two separately.

## 4. Risks and controls

A large indirect effect alone is not enough. Each risk below could produce one without persona space mattering. Our stored activations already show that several of these risks are real.

| Risk | Control |
|---|---|
| Any patch that moves this much of the hidden state flips the model, regardless of direction. | We use matched-size random subspaces. They have the same dimension and capture the same share of $h_1 - h_0$ as persona space. Plain random subspaces are too easy to beat because persona space captures far more of the difference. |
| Two templates always differ in persona coordinates. In stored activations, weak-to-weak pairs differ in persona space about as much as weak-to-strong pairs. | We use a placebo that patches in another weak prompt's persona coordinates. This shift has a similar size but the wrong direction, so it should not jailbreak. |
| Persona space contains much of the refusal direction, so a persona patch also moves refusal. | We repeat the patch after removing the refusal-aligned part from the persona subspace. We also run that part alone. This tests whether the effect comes from persona or from refusal inside persona space. |
| $h^*$ combines pieces of two states, which can produce an unnatural state the model never sees. Subspace patches are known to give misleading results this way. | We use an on-manifold patch. It reaches the same persona coordinates with the smallest change that stays within the directions natural activations actually use. We also log how unnatural each patched state is. |
| The patch may encode only which template was used, rather than anything about the specific prompt. | We use a template-average patch. It replaces the pair-specific persona shift with the average shift between the two templates. If it works equally well, we describe the effect as a template-level persona shift. |
| Our weak templates are not persona prompts, while the strong ones are amoral personas. "Persona" could then mean only "has a persona frame". | We run a second experiment with a persona-matched contrast. The weak side also uses persona templates, ones that mostly fail. |
| The patch could carry over $x_1$'s content, making the model answer $x_1$'s request instead of $x_0$'s. This is the note's sanity check. | We check fidelity by pairing prompts about different harmful behaviours. A judge identifies which request the patched response answers. For same-behaviour pairs, we instead check whether the response adopts $x_1$'s persona format. |

## 5. Choosing and pairing jailbreaks

The note asks for jailbreak methods. Most of our Llama attack methods sit in the middle of the ASR range. They do not split clearly into "weak" and "strong". The clearest method-level pair is weak PEZ against strong GPTFuzz. We use it as a replication.

The HarmBench human-jailbreak templates split much better. Some almost always fail, and others almost always succeed. Each template is already crossed with all HarmBench behaviours, and the responses are judged. We can therefore label whole templates as weak or strong. This follows the note's rule of labelling types, not prompts.

These templates are the main contrast. We do not use direct requests as the weak side because they are not a jailbreak method.

The main estimate uses same-behaviour pairs. One weak and one strong template wrap the same harmful request. Only the jailbreak wrapper differs, which gives the cleanest effect. We run the note's random pairs, with different behaviours, as the fidelity check from §4.

We split behaviours into a small calibration set and a larger estimation set. We make every choice on the calibration set:

- The patch site.
- The size of the matched random subspaces.
- How we validate the cheap proxy described in §6.
- Any learned subspace described in §2.

We use the estimation set once, with those choices locked. This keeps us from tuning the final numbers to the data we report them on.

## 6. Running and judging the experiment

We use the official Llama-3.1-8B-Instruct chat template that produced our stored labels and activations. We do not use the mirror, whose template differs. We use greedy decoding, as in all earlier runs. We regenerate unpatched baselines in the same harness, so patched and unpatched runs differ only by the patch.

We use the ablation experiment's label judge, which sees the plain harmful request and the response. The HarmBench classifier serves as a second judge for cross-checking. Gibberish is its own category, never a jailbreak or a refusal. We count judge parse failures separately.

The cheap proxy checks a short generation for refusal phrases with a regex. We need it to screen many random subspaces without paying for a judge. We use it only for screening, never for a reported number. Checking only the first token does not work here. Several templates make the model open with formatting or a persona tag rather than "I can't" or "Sure".

On the calibration set, we run only the unpatched and full-state patches to choose the site. We choose the narrowest site from §2 where the full-state patch recovers a substantial part of the total effect. If none does, we stop. The jailbreak does not act there, and this design cannot test persona mediation.

Before the main run, we record the main measure and its comparisons. We report the proportion mediated on same-behaviour pairs, with the NIE and NDE alongside. We compare it against matched random subspaces, the placebo, refusal removed, and the on-manifold patch. We compute uncertainty per behaviour because every behaviour appears in many pairs.

## 7. Implementation

New code goes in `~/persona-vector-jbing` and follows our existing directional-ablation pipeline. We do not touch existing direction bundles.

| Component | What it does |
|---|---|
| `build_mediation_design.py`, Mac | Labels templates, creates the calibration/estimation split and all pair lists, and writes them to `design.json` with seeds and the locked choices. |
| `cache_site_states.py`, GPU | Runs one forward pass per prompt and stores the state at the five closing tokens. These states must match our stored activations. |
| `build_mediation_subspaces.py`, Mac | Builds every subspace the patches need: persona, refusal-removed, matched random, on-manifold, and template averages. |
| `mediate_patch.py`, GPU | Runs a hook that swaps in the patched state during the prompt pass only, then generates and judges. It also has a proxy mode. It follows `ablate_fixed_direction.py`, including resume after preemption. |
| `judge_mediation_fidelity.py` | Judges which request the response answered and checks the persona format. |
| `extract_mediation_cells.py`, plotting and table scripts, Mac | Computes TE, NIE, NDE and the proportion mediated with confidence intervals, and produces the figure and table. |

The hook must patch exactly the five closing positions of every row in a padded batch. It must patch only on the prompt pass. Before any real run, an identity test patches each state with itself and must reproduce the unpatched output. GPU jobs go to RunAI from the existing work directory on scratch, as the ablation run did.

## 8. Stages

1. Build the design on the Mac, including template labels, splits, and pairs.
2. Prepare the pipeline by caching states, building subspaces, and passing the identity test.
3. Calibrate on the calibration set by choosing the site, validating the proxy, and sizing the matched random subspaces. Stop here if no site carries the effect.
4. Run the judged core patches on the estimation set. These include baseline, full-state, persona patch for NIE, reverse patch for NDE, placebo, refusal removed, on-manifold, and template average.
5. Run controls and replications. These include matched random subspaces, the cross-behaviour fidelity check, the persona-matched contrast, and the PEZ/GPTFuzz method contrast.
6. Run extensions if stages 4–5 leave a specific question open. These include the layer sweep, a dose sweep scaling the persona patch up and down, the top-$k$ persona subspace, and a learned subspace inside persona space.

Stages 4 and 5 need judge calls. They start only after approval.

We leave Gemma out for now. Our human templates barely jailbreak it, so there is no clear strong side. Persona space also captures little of the template differences there. Gemma would need the method-level contrast instead.

## 9. Interpreting the outcome

| If we see | What we can say for these templates, Llama, layer 16 |
|---|---|
| High proportion mediated that beats every control, and a small NDE | Persona space carries most of the jailbreak and is needed for it. |
| High proportion mediated, but the NDE is also large | Persona space is sufficient but redundant. The jailbreak can also go around it. This matches the detection finding that the signal is not only in persona space. |
| Moderate proportion mediated that beats the controls | Persona space partially mediates the jailbreak. It carries a measurable share. |
| Effect no better than matched random subspaces or the placebo | Moving this much of the state in a template-like direction is enough. We make no persona-specific claim. |
| Effect disappears once refusal is removed | The effect runs through the refusal-aligned part of persona space. |
| Pair-specific patch no better than the template average | The effect is a template-level persona shift. We should word the claim that way. |
| Effect gone with the on-manifold patch | It was an artefact of an unnatural state. We make no claim. |
| Persona-matched contrast works too | The effect is not just "persona frame vs no frame". |
| Patched responses answer $x_1$'s request | The patch carries content, not just persona. As the note says, the setup needs rethinking. |

A positive result would add a paragraph after the ablation paragraph in the paper's steering section, plus a new appendix. A null result also goes in the appendix. We would then soften the "causal" wording in the intro and abstract.

## 10. Decisions

1. Should templates be the main contrast and the PEZ/GPTFuzz methods the replication, or the other way round?
2. Should same-behaviour pairs provide the main estimate, with the note's random pairs as the fidelity check?
3. Should we choose the patch site by the calibration rule, or always patch only the last token, as in the note?
4. What proportion mediated should count as "persona space carries the jailbreak"? We will propose a threshold before the main run.
5. Should the layer sweep and the learned subspace move up from extensions into the main plan?
