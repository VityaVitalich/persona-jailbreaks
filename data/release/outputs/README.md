# Persona jailbreaks: data

Model outputs, judge scores, trait vectors and activations behind "Jailbreaks Live in Persona Space", in the layout
the code writes (`outputs/<model>/`). Place this folder at `outputs/` in the code repository; `scripts/4_detect.py`
and `scripts/5_analysis.py` then run without a GPU.

**Content warning.** Responses include harmful content that the models produced under jailbreak attacks and steering.

## Layout (per model: `llama`, `gemma`)

| Path | Content |
|---|---|
| `vectors/generations/<trait>.jsonl` | 400 answers per trait: `label` (`positive_p{prompt}_q{question}`), `system`, `question`, `answer`, judge `score` (0-100, null if unparsed) |
| `vectors/vectors.pt` | raw trait vectors of all 240 traits at the detection layer (Llama 16, Gemma 30) |
| `vectors/T.npy`, `trait_names.json` | unit-norm vectors of the kept traits (Llama 229, Gemma 240) |
| `vectors/persona_axis.npy`, `matched_pairs.json` | Principal Persona Axis; per-trait count of matched pairs with a score gap of at least 50 |
| `<dataset>/rows.jsonl` | `id`, `behavior_id`, `behavior`, `prompt`, `response`, `jailbroken` (judge label) for `harmbench`, `gcg`, `pair`, `pap`, `gptfuzz`, `pez`, `wildjailbreak` |
| `<dataset>/acts.pt` | `last`: activation at the detection layer and the last prompt token, aligned with `rows.jsonl` |
| `refusal_direction.pt` | mean activation on harmful minus harmless instructions |
| `steer/<benchmark>/<arm>.jsonl` | steering on refused prompts: `alpha` (first α ≤ 0.5 that jailbroke, else null), final `response`, `degenerate`, `jailbroken` |
| `ablate/<benchmark>/<arm>.jsonl` | directional ablation on complied prompts (Llama): `response`, `degenerate`, `jailbroken` |

## Differences from the code in the repository

The data was produced before the code was consolidated. Where the protocol differs:

- **Judges:** labels and trait scores used GPT-4.1-mini at temperature 1. Steering used a judge that sees the jailbreak prompt rather than the behavior.
- **Trait answers:** Llama answers were sampled with vLLM, Gemma answers were capped at 256 tokens.
- **Llama refusal direction:** taken at the input of block 16, from the filtered instructions of Arditi et al. and their chat template without the date header. The code uses the output of block 16.
- **Steering:**
  - The Llama PersonaJB direction was fit with C = 1.
  - The Llama single-trait HarmBench arms (`trait_*_raw_behaviors`) ran on 400 HarmBench behaviors without jailbreak templates.
  - Controls ran on a 300-prompt subset.
  - JBB arms cover all 100 goals.
  - `degenerate` is recomputed with the rule in `scripts/6_steer.py`, and degenerate responses count as failures.
- **Not included:** the multi-token activations for the readout analysis, the JLT and Mahalanobis inputs, and the text-baseline scores. `scripts/2_collect.py` and `scripts/3_baselines.py` regenerate them.
