# Jailbreaks Live in the Persona Subspace

Code, numeric results, model generations and an interactive results dashboard for the persona-jailbreaks project:
detecting jailbreaks from the persona directions of a model's residual stream, steering along those directions, and
measuring how much of the refusal and jailbreak directions lies inside the persona subspace.

**Dashboard: https://vityavitalich.github.io/persona-jailbreaks/**

## Credits

This repository builds on work by **Dominic Bazina** (original research repository `dbazinag/persona-vector-jbing`),
and on the **Assistant Axis** repository [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis),
whose trait-elicitation and persona-axis code the pipeline here extends. The `transcripts/`, `notebooks/` and
`assistant_axis/` directories come from that upstream work.

This repository starts with fresh history: it is a snapshot, not a fork of the original repository's commits.

## Content warning

`full_trait_output/`, `data/release/` and `data/gemma_traits40_backup/` contain model responses produced under
jailbreak attacks and activation steering, together with judge labels. They are published for safety research.

Responses for HarmBench behaviours in the `chemical_biological` category are **withheld** here: the response text is
replaced by `[withheld: chem/bio, see HF dataset]`, while prompts, behaviour ids, labels and scores are kept. 57,496
response fields across 1,559 files are withheld. The full text is in the gated Hugging Face dataset below.

## Layout

| Path | Content |
|---|---|
| `dashboard/` | the results dashboard: `build_data.py` → `data.json`, `index.html`, `style.css`, `deploy.sh`. See `dashboard/README.md` for what each tab shows and which file feeds it. |
| `full_trait_output/` | analysis scripts and their outputs: result JSONs, per-prompt score files under 50 MB, generations and judge labels (`*.jsonl.gz`), logs, figures. The sweeps live in `q1_detection/`, `norm_share_controls/` (+ `gemma/`), `trait_pc_sweep/`, `pca_k_sweep/`, `olmo3_7b/detect/`, `qwen3_32b/`, `roleplay_check/`, `cosine_nulls/`. |
| `data/release/` | the paper's data release in the layout the clean code expects: `outputs/` (per-model rows, steering and ablation arms, trait answers), `raw/` (the original run outputs), `derived/` (detection and analysis results). See `data/release/outputs/README.md`. |
| `data/gemma_traits40_backup/` | Gemma trait-elicitation generations and judge scores (tar.gz). |
| `results/` | paper-level result tables, cell CIs and summary JSONs / CSVs. |
| `full_trait_tools/`, `jailbreaking_tools/`, `assistant_axis/`, `trait_tools/`, `pipeline/`, `scripts in the root` | the pipeline: trait elicitation, vector extraction, attacks, activation capture, judging, steering, ablation, plotting. |
| `transcripts/`, `notebooks/`, `presentation/` | upstream transcripts, exploratory notebooks and slides. |

### Rebuilding the dashboard

```bash
python3 dashboard/build_data.py     # reads this repo's result files, writes dashboard/data.json
bash dashboard/deploy.sh            # rebuild, scan, and push the gh-pages branch
```

`build_data.py` only reads result files that are already saved; it never recomputes a score. Every AUC it reports is
the folded AUC, max(a, 1−a).

## Data that is not in git

Activations, trait vectors, steering bundles, the large per-prompt score archives and the free-text attack logs are
too big for GitHub. They are in the gated Hugging Face dataset
**[VityaVitalich/persona-jailbreaks-data](https://huggingface.co/datasets/VityaVitalich/persona-jailbreaks-data)**
(about 40 GB), which also carries the un-withheld chem/bio responses. The data card maps each file to the script in
this repository that produced or consumes it.

Some activation sets are still only on the compute clusters and are listed as "to add" in that data card.

## Licences

- **Code: MIT** (`LICENSE`).
- **Data and results: CC BY 4.0** (`LICENSE-DATA`) — covers `results/`, `full_trait_output/`, `data/` and
  `dashboard/data.json`. Third-party prompt sets and attack artifacts keep their original licences.
