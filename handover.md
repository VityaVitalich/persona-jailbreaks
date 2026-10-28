# Handover — assistant-axis-llama3.1-8B

**Written:** 2026-05-28 · **Updated:** 2026-07-21 | **Author:** Claude Code session

---

## 1. Project Overview

This project develops **trait-vector-based jailbreak detection** for LLMs. The core idea: extract a set of persona "trait" vectors from the model's hidden states by running the model on personas of different psychological traits, then project layer activations onto those vectors as features for a linear classifier. The classifier is trained on human-labelled HarmBench responses (jailbroken vs not), then evaluated on its ability to detect attacks it was never trained on (PAIR, PAP, GPTFuzz, PEZ, GCG).

The primary target model is **Llama-3.1-8B-Instruct**. A parallel track targets **OLMo-3-7B-Instruct** to test generality. There is also a jailbreak attack component: steering the model's trait activations toward a computed "jailbreak direction" during generation to achieve high attack success rates.

**Current status:** The paper's experiments are complete. All paper-used data is organized under `full_trait_output/paper_data/` on the cluster (see §8), and the repo has been cleaned up. No jobs are currently running.

---

## 2. Repository & Environment

### Local (your WSL2 machine)
- **Project root:** `/home/dbazinag/projects/assistant-axis-llama3.1-8B`
- Edits are made here, committed, and pushed to the private repo (see **Git & sync** below).

### Cluster (EPFL RCP scratch)
- **Cluster path:** `/dlabscratch1/bazina/assistant-axis-llama3.1-8B`
- Inside RunAI pods the PVC is mounted at `/mnt`, so the path becomes `/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B`
- These two paths refer to the same files: cluster jobs run scripts from `/mnt/...`, SSH from outside uses `/dlabscratch1/...`

### Git & sync
- **Remote (private):** `github.com/dbazinag/persona-vector-jbing` — moved off a public fork; the local checkout's `origin` points here.
- **Local and cluster are two independent checkouts, not auto-synced.** The cluster is where jobs run; the local machine is where you commit + push.
- **The cluster cannot authenticate to the private repo** (no GitHub creds there), so `git pull`/`push` on the cluster fail until you add a PAT (`git config --global credential.helper store`, then fetch once) or switch its remote to SSH. Until then, sync the cluster by copying changed files over SSH, or — once creds exist — `git fetch && git reset --hard origin/master` (the `full_trait_output/` data is untracked/gitignored, so a reset never touches it).
- **gitignored** (on disk only, not in git): `full_trait_output/` (outputs), `*.sh` (run scripts), `*.tar.gz` (backups), `.claude/`.

### Key subdirectories

| Dir | Purpose |
|-----|---------|
| `assistant_axis/` | Core library: axis computation, steering, PCA, generation |
| `full_trait_pipeline/` | 5-step pipeline: generate traits → activations → judge → vectors → PC1 axis |
| `full_trait_tools/` | All sweep scripts, baseline runners, attack evaluation, data collection |
| `jailbreaking_tools/` | Trait-steering attack scripts (`trait_steering_attack_jbb.py` is the main one) |
| `data/` | Static data: trait definitions, role prompts, extraction questions |
| `full_trait_output/` | **All outputs live here** (on cluster only, ~hundreds of GB) |
| `results/` | Small CSV/JSON summary files committed to git |
| `run_*.sh` | One script per experiment; these are the entry points submitted to RunAI |

### Virtual environment
- **uv** is the package manager: `/dlabscratch1/bazina/.local/bin/uv`
- Venv location: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.venv`
- Python ≥3.10, PyTorch 2.9.1+cu128, vllm 0.15.0 (pinned via `uv.lock`)
- To install from scratch on cluster: `cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B && uv sync`
- To run a script: `uv run python full_trait_tools/SCRIPT.py [args]`

### Secrets / config files
- `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.env` — contains `OPENAI_API_KEY` (needed by any script that calls GPT-4.1-mini for judging). Source it with `set -a; source .env; set +a` before running judge scripts.
- HuggingFace token: set via `HF_TOKEN` env var in scripts that pull gated models (e.g. Llama-3.1-8B). Some scripts hardcode it inline; check `run_wjb_collect.sh` for the pattern.
- HF cache: `/dlabscratch1/bazina/.cache/huggingface`

---

## 3. Infrastructure & Remote Access

### RunAI cluster
- **CLI:** `runai-rcp-prod` (installed locally, not over SSH)
- **Project:** `dlab-bazina`
- **Suppress deprecation warnings:** always prefix with `SUPPRESS_DEPRECATION_MESSAGE=true` — see Known Issues below
- **List jobs:** `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina`
- **Submit job:** `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit <name> -p dlab-bazina --image ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt [--gpu N] -- bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_SCRIPT.sh`
- **Pull logs:** `SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs <jobname> -p dlab-bazina`

### SSH access via file-bridge
Cluster nodes are not directly accessible. You SSH through a permanent interactive pod called `file-bridge62` that runs `sleep infinity`. The pod gets **suspended** by the cluster after idle time but can be **resumed** — do not submit a new one, just resume the existing job.

**Current bridge:** `file-bridge62` (use `runai resume` if Suspended)

**Resume if suspended:**
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume file-bridge62 -p dlab-bazina
```

**Wait for Running, then start port-forward:**
```bash
# Get pod name
kubectl get pods -n runai-dlab-bazina --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep file-bridge62

# Port-forward (background)
kubectl port-forward -n runai-dlab-bazina file-bridge62-0-<N> 2242:22 > /tmp/pf61.log 2>&1 &
sleep 5
```

**SSH in:**
```bash
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -l bazina localhost
```

Wait ~10s after resume for the GASPAR user to be created before SSH-ing. Poll:
```bash
until ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost "echo ok" 2>/dev/null; do sleep 5; done
```

**If file-bridge62 is ever permanently lost**, submit a new one (bump the number, e.g. `file-bridge63`) with `sleep infinity` (not `sleep 7200`):
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit file-bridge63 -p dlab-bazina \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --pvc dlab-scratch:/mnt --node-pools cpu \
  --interactive --cpu 4 --memory 16G -- sleep infinity
```
The image starts sshd automatically — do NOT add a custom `bash -c "useradd ... sshd"` block.

### Cluster paths
- Project root: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B` (via SSH) = `/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B` (inside pods)
- HarmBench repo: `/dlabscratch1/bazina/HarmBench` — used for GCG generation; results at `/dlabscratch1/bazina/HarmBench/results/`
- All experiment outputs: `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/`
- Model cache: `/dlabscratch1/bazina/.cache/huggingface/`

### VPN requirement
Both the RunAI API and SSH port-forward require the EPFL VPN to be active. The WSL2 virtual network adapter sometimes drops (gateway `172.27.80.1` becomes unreachable, breaking DNS for `caas-prod.rcp.epfl.ch`). Fix: run `wsl --shutdown` from Windows PowerShell, then reopen your WSL terminal and reconnect to VPN.

---

## 4. How to Run Things

### General pattern for submitting a training job
```bash
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod submit <jobname> \
  -p dlab-bazina \
  --image ghcr.io/jkminder/dlab-runai-images/pytorch:master \
  --pvc dlab-scratch:/mnt \
  --gpu 1 \
  -- bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_SCRIPT.sh
```

For CPU-only jobs (sweep scripts, judge scripts), omit `--gpu 1`.

### Running scripts locally on the cluster (via SSH)
```bash
ssh bazina@... "cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B && uv run python full_trait_tools/SCRIPT.py [args]"
```

### Key script entry points

| Script | What it does |
|--------|-------------|
| `run_steer_trait_jbb_strong.sh` | Trait-steering jailbreak attack, k=10/20/30 sweep, alpha 0.1–0.7 |
| `run_eval_gcg_jbb.sh` | Evaluate GCG attack results (submit after gcg-jbb-gen2 succeeds) |
| `run_all_traits_sweep_olmo3_all_traits.sh` | OLMo-3 trait feature sweep (classification) |
| `run_all_traits_sweep_olmo3_pca3.sh` | OLMo-3 PCA3 feature sweep (outperforms trait) |
| `run_baselines_self_exam_olmo3.sh` | OLMo-3 self-exam baseline (asks model to self-evaluate) |
| `run_harmbench_gcg_olmo3_merge.sh` | Merge 4 GCG chunks into one test_cases.json |
| `run_save_olmo3_clf.sh` | Save best OLMo-3 classifier to pkl |
| `run_save_wjb_clf.sh` | Save best WildJailbreak classifier to pkl |

### Output locations — use `full_trait_output/paper_data/` (canonical index)

**All paper data is indexed under `full_trait_output/paper_data/`** — a symlinked, browsable tree
(originals untouched, nothing duplicated; every entry was verified to back a paper number). Route
agents through this rather than the raw dirs:

| `paper_data/` path | Backs | Contents |
|---|---|---|
| `detection_llama/` | Table 1 | `trait__sweep_logreg/` (trait sweep), `raw_pca__fast_transfer/{all,pap,pez,gptfuzz}/`, `baselines/{mahalanobis,jbshield,jlt,gradsafe_llamaguard_selfexam,perplexity}/`, `layer_sweep/`, `consolidated_table.json` (final per-method AUCs), `trained_models/` |
| `detection_gemma/` | Table 2 (HB+WJB-trained) | `trait/ raw/ pca/ baselines/` |
| `steering/{harmbench,jbb,wildjb}/` | Table 4 | classic-attack + steering variants (persona_w, raw_w, w_aligned, steer_traits) |
| `analysis/` | Tables 5–8 + figs | `single_trait_auc`, `trait_weights.csv`, `pca_decomposition`, `w_alignment`, `jailbreak_pca_inputs/` |
| `building_blocks/` | inputs / vectors | `persona_trait_vectors_llama` (229), `trait_matrix_layer16.npy`, `classifier_w_hyperplanes` (steering **`w`** directions), `trait_generation`, `trait_judge`, `static_inputs` (repo `data/`) |
| `activations/` | eval data | `harmbench`, `gcg`, `pair`, `pap`, `gptfuzz`, `pez`, `wildjailbreak` |
| `models/` | model weights | `llama_3.1_8b_instruct`, `gemma_4_31b_it`, `llama_guard_3_8b` (symlinks into the HF cache) |

- Rebuild the consolidated Table 1 from the canonical dirs: `full_trait_tools/build_llama_detection_table.py`.
- Retrain the two paper detectors: `full_trait_tools/train_paper_detectors.py` → `paper_data/detection_llama/trained_models/`.
- Superseded / smoke runs live in `full_trait_output/_archive/` (kept intact, **not** in `paper_data/`).

### Reading results (all via `paper_data/`)

```bash
# Detection: read the final Table 1 (per-method AUCs, single-sourced)
ssh bazina@... "python3 -c \"
import json
d = json.load(open('full_trait_output/paper_data/detection_llama/consolidated_table.json'))
for name, m in d['methods'].items():
    print(f\\\"{name:32s} HB={m['HarmBench']} xfer={m['Avg_Transfer']}\\\")
\""
```

```bash
# Detection: rank models within the trait sweep (via paper_data)
ssh bazina@... "python3 -c \"
import json
d = json.load(open('full_trait_output/paper_data/detection_llama/trait__sweep_logreg/results.json'))
rows = [(v.get('human_test',{}).get('auc',{}).get('mean',0), v.get('val_auc',{}).get('mean',0), m,
         {k: v[k]['auc']['mean'] for k in ['PAIR','PAP','GPTFuzz','PEZ'] if k in v})
        for m, v in d['summary'].items()]
rows.sort(reverse=True)
for rank, (t, val, m, tr) in enumerate(rows[:5], 1):
    print(rank, m, f'val={val:.4f}', f'test={t:.4f}', tr)
\""
```

```bash
# Steering: read an attack summary (per-pair ASR, GPT-4.1-mini) via paper_data
ssh bazina@... "python3 -c \"
import json
d = json.load(open('full_trait_output/paper_data/steering/jbb/persona_w/harmbench/summary.json'))
print('per_pair_asr:', d['gpt41mini']['per_pair_asr'])
print('per_behavior_asr:', d['gpt41mini']['per_behavior_asr'])
print('jbb_rubric:', d['jbb_rubric']['per_pair_asr'])
\""
# (other steering cells: paper_data/steering/{harmbench,jbb,wildjb}/{persona_w,raw_w,w_aligned,steer_traits,classic}/)
```

---

## 6. Known Issues & Past Mistakes

### `--suppress-deprecation-message` as trailing arg — BREAKS JOB
**Symptom:** Job fails immediately with `sleep: unrecognized option '--suppress-deprecation-message'`  
**Root cause:** When passed as a trailing arg, RunAI forwards it into the container where it gets appended to the `sleep 7200` command.  
**Fix:** Always use `SUPPRESS_DEPRECATION_MESSAGE=true` as an env var *prefix* before `runai-rcp-prod`, not as a trailing `--suppress-deprecation-message` flag.

### Port-forward conflict
**Symptom:** `Failed to create port forwarding... unable to listen on [{2242 22}]`  
**Root cause:** A stale port-forward process is still holding port 2242.  
**Fix:** `pkill -f "port-forward"` before starting a new one.

### `results.json` key structure is non-obvious
When reading sweep results, the structure is:
```python
data['summary'][model_name]['val_auc']['mean']           # validation AUC
data['summary'][model_name]['human_test']['auc']['mean'] # human-labelled test AUC
data['summary'][model_name]['PAIR']['auc']['mean']       # transfer AUC for PAIR
```
Do NOT try `val_auc = data['summary'][model]['val_auc']` directly — it's a dict, not a float. This caused a `NameError` earlier in the session.

### OLMo-3 uses `trust_remote_code=True`
All scripts that load OLMo-3 need `--trust_remote_code` flag or `trust_remote_code=True` in the `AutoModel.from_pretrained` call. The full_trait_pipeline step 1 already has this, but if writing new OLMo-3 scripts, remember to add it.

### WJB label field inconsistency
`classified_responses.jsonl` from WildJailbreak uses `jailbroken` not `binary_jailbroken`. `save_best_wjb_clf.py` now falls back correctly (`.get("binary_jailbroken", r.get("jailbroken"))`), but if writing new scripts that consume WJB labels, check which field name the file actually uses.

---

## 7. Key Files & What They Do

| File | Purpose | Notes |
|------|---------|---------------------|
| `full_trait_tools/build_llama_detection_table.py` | **Regenerates the consolidated Table 1** (`paper_data/detection_llama/consolidated_table.json`) from the canonical result dirs; documents each row's source | Trait←sweep logreg, Raw/PCA←fast_transfer, baselines←their dirs |
| `full_trait_tools/train_paper_detectors.py` | **Trains the two paper detectors** (trait `logreg_l2_C10.0`, raw `logreg_C1.0`) on HarmBench, evaluates transfer, saves to `paper_data/detection_llama/trained_models/` | Reproduces trait xfer ≈0.76 / raw ≈0.71 without the full sweep |
| `full_trait_tools/run_all_traits_sweep_v2.py` | Main sweep: trains classifiers on trait-projected activations, reports val + test + transfer AUCs. Backs `paper_data/detection_llama/trait__sweep_logreg/` | Added balanced/raw split, PCA feature type |
| `jailbreaking_tools/trait_steering_attack_jbb.py` | Trait-steering jailbreak attack. Steers top-k trait axes during generation. | Sign fix at line 594 |
| `full_trait_tools/collect_harmbench_activations.py` | Collects Llama-3.1-8B activations + responses on HarmBench behaviours; supports OLMo-3 via `--model` + `--trust_remote_code` | OLMo-3 support added |
| `full_trait_tools/collect_harmbench_attack_activations_olmo3.py` | Collects OLMo-3 activations for attack families (PAIR/PAP/GPTFuzz/PEZ) on HarmBench | New file this session |
| `full_trait_tools/collect_wildjailbreak_activations.py` | Collects activations for WildJailbreak eval split (adversarial + direct harmful rows) | New file this session |
| `full_trait_tools/judge_harmbench_responses.py` | GPT-4.1-mini judge for classifying responses as jailbroken/not | Updated to accept `--responses_path` |
| `full_trait_tools/run_baselines_self_exam.py` | Self-exam baseline: asks the model to classify its own responses | No output caching — restarts from scratch on job restart |
| `full_trait_tools/run_baselines_llama_guard.py` | Llama Guard baseline AUC across attack families | Supports `--olmo3` flag |
| `full_trait_tools/save_best_wjb_clf.py` | Sweeps LogisticRegression variants on WJB data, saves best to pkl | Fixed WJB label field |
| `full_trait_tools/save_best_olmo3_clf.py` | Saves best OLMo-3 classifier (from all-traits sweep) to pkl | New file |
| `full_trait_tools/eval_olmo3_clf.py` | Evaluates a saved OLMo-3 clf on all attack families | New file |
| `run_steer_trait_jbb_strong.sh` | Runs steer-strong4 k=10/20/30 sweep (alpha 0.1–0.7) | This is what steer-strong4 job runs |
| `run_harmbench_gcg_olmo3_merge.sh` | Merges 4 GCG chunk outputs for OLMo-3 | Lives on cluster, not in local git |

---

## 8. Data & Checkpoints

### Full `full_trait_output/paper_data/` inventory

Everything the paper's tables/figures use, as `paper_data/` symlinks (canonical index; originals
resolve under `full_trait_output/`). Use `paper_data/` so a fresh agent traverses the organized tree.
The symlinks are **relative**, so they resolve whether you browse via `/dlabscratch1/…` or
`/mnt/dlabscratch1/…` (inside a pod) — if one looks odd, check you're on the cluster, not local.

**Inputs (shared)**

| What | `paper_data/…` | Description |
|---|---|---|
| Llama HarmBench activations | `activations/harmbench` | Layer-16 (+28) activations for all Llama human-request pairs |
| Attack-family activations | `activations/{gcg,pair,pap,gptfuzz,pez}` | The 5 unseen attack families (Table 1 transfer eval) |
| WildJailbreak activations | `activations/wildjailbreak` | Adversarial + direct harmful pairs (WJB eval split) |
| Persona/trait vectors (Llama) | `building_blocks/persona_trait_vectors_llama` | **229 × 4096** filtered set (`filter_matched_pairs_ge_50_count_ge_10_total`); NOT `all_traits_no_filter` (240) |
| Trait matrix (layer 16) | `building_blocks/trait_matrix_layer16.npy` | Normalised 229×4096 matrix used by classifiers |
| Trait generation / judge | `building_blocks/{trait_generation,trait_judge}` | Persona prompts+responses + judge scores used to build the vectors |
| Static inputs | `building_blocks/static_inputs` | Repo `data/`: trait definitions, role prompts, extraction questions |
| Classifier **`w`** / hyperplanes | `building_blocks/classifier_w_hyperplanes` | Classifier-normal `w`: `{hyperplane_normal,stable_hyperplane}_layer{16,28}.pt` — the **steering directions** for Table 4 (persona-w / raw-w). `w` is also inside each `detection_llama/trained_models/*_detector.pkl` (the LogReg `coef` + `trait_matrix` maps it back to activation space) |
| Assistant axis (PC1) | `analysis/jailbreak_pca_inputs/assistant_axis_pc1.pt` | The assistant-axis direction (PC1) |
| Base model weights | `models/{llama_3.1_8b_instruct, gemma_4_31b_it, llama_guard_3_8b}` | Symlinks into the HF cache `~/.cache/huggingface/hub` — already downloaded, no re-download needed |

**Detection — Llama (Table 1)**

| What | `paper_data/…` |
|---|---|
| Trait sweep (trait row source) | `detection_llama/trait__sweep_logreg` |
| Raw / PCA (fast_transfer) | `detection_llama/raw_pca__fast_transfer/{all,pap,pez,gptfuzz}` |
| Baselines | `detection_llama/baselines/{mahalanobis,jbshield,jlt,gradsafe_llamaguard_selfexam,perplexity}` |
| Layer sweep (figure) | `detection_llama/layer_sweep/{summary.csv, per_layer/L00..L31}` |
| Final table | `detection_llama/consolidated_table.json` |
| Trained detectors | `detection_llama/trained_models/{trait,raw}_detector.pkl` |

**Detection — Gemma (Table 2, HB+WJB-trained)**

| What | `paper_data/…` |
|---|---|
| Trait / Raw / PCA | `detection_gemma/{trait,raw,pca}` |
| Baselines | `detection_gemma/baselines/{mahalanobis,jbshield,jlt,gradsafe_llamaguard_selfexam,perplexity}` |

**Steering attacks (Table 4)**

| What | `paper_data/…` |
|---|---|
| HarmBench | `steering/harmbench/{steer_traits_and_raw_w, persona_w}` |
| JBB | `steering/jbb/{classic, persona_w, raw_w, w_aligned, steer_traits}` |
| WildJailbreak | `steering/wildjb/{classic, raw_w, w_aligned, steer_traits}` |

**Analysis (Tables 5–8 + figures)**

| What | `paper_data/…` |
|---|---|
| Single-trait AUC (Table 5) | `analysis/single_trait_auc` |
| Trait weights (Table 6) | `analysis/trait_weights.csv` |
| PCA decomposition (Table 7) | `analysis/pca_decomposition` |
| w-alignment cosines (Table 8) | `analysis/w_alignment` |
| Jailbreak-PCA figure inputs | `analysis/jailbreak_pca_inputs/` |

**Not in `paper_data/`** (still on the cluster): HarmBench behaviours
`/dlabscratch1/bazina/HarmBench/data/behavior_datasets/` (JBB in `jbb_behaviors.csv`); GCG test cases
`/dlabscratch1/bazina/HarmBench/results/JBB_GCG/llama3_1_8b/test_cases/`; superseded/smoke runs
`full_trait_output/_archive/`; legacy OLMo-3 `full_trait_output/*_olmo3*` (parallel track, not in the paper).

### Checkpoints
- **Paper Table 1** (per-method AUCs, single-sourced): `paper_data/detection_llama/consolidated_table.json`
- **The two paper detectors** (trait `logreg_l2_C10.0`, raw `logreg_C1.0`): `paper_data/detection_llama/trained_models/{trait,raw}_detector.pkl` — retrain with `train_paper_detectors.py` (Provided because doing another parameter sweep takes over 20 hours)
- Legacy: `full_trait_output/wildjailbreak_clf/best_model.pkl`, `full_trait_output/all_traits_sweep_v2_olmo3/best_model.pkl`

### Artefact pkl format
```python
{
    "pipeline":     sklearn.Pipeline (StandardScaler + LogisticRegression),
    "trait_matrix": np.ndarray [n_traits × 4096],
    "trait_names":  list[str],
    "layer":        int,  # 16
    "threshold":    float,
    "sign":         int,  # +1 or -1
    "meta":         dict  # model name, cv_auc, val_auc, etc.
}
```

---

## 9. External Dependencies & APIs

| Service | Purpose | Auth |
|---------|---------|------|
| OpenAI API (GPT-4.1-mini) | Judge for classifying model responses as jailbroken/not | `OPENAI_API_KEY` in `/dlabscratch1/bazina/assistant-axis-llama3.1-8B/.env` |
| HuggingFace Hub | Model weights: `meta-llama/Llama-3.1-8B-Instruct` (primary), `google/gemma-4-31B-it` (Table 2), `allenai/OLMo-3-7B-Instruct` (legacy), `meta-llama/Llama-Guard-3-8B` (baseline). Cached at `/dlabscratch1/bazina/.cache/huggingface/hub`; also symlinked in `paper_data/models/` | `HF_TOKEN` env var; token at `/dlabscratch1/bazina/.cache/huggingface/token` |
| EPFL RCP / RunAI | GPU compute; job scheduling | `runai-rcp-prod` CLI, EPFL VPN required, project `dlab-bazina` |
| Llama Guard 3 | Input-only / input+output jailbreak detection baseline | Loaded from HF (same HF_TOKEN) |
| WildJailbreak dataset | Adversarial jailbreak pairs for evaluation | `allenai/wildjailbreak` via HuggingFace `datasets` |

**GPT-4.1-mini cost note:** Judge scripts call GPT-4.1-mini per (behavior, response) pair. Large runs (HarmBench self-exam: 3180 rows, 5 families = ~15K calls) can be expensive. Always verify `.env` is loaded before submitting judge jobs.

---

## 10. Useful One-Liners

```bash
# List all jobs, filter to interesting ones
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina 2>&1 | \
  grep -E "Running|Failed|Pending" | grep -v "file-bridge[0-9]*  Succeeded"

# Resume the permanent file-bridge (do this instead of submitting a new one)
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod resume file-bridge62 -p dlab-bazina

# Start SSH port-forward (get pod name first)
POD=$(kubectl get pods -n runai-dlab-bazina --field-selector=status.phase=Running \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep file-bridge62)
kubectl port-forward -n runai-dlab-bazina $POD 2242:22 > /tmp/pf61.log 2>&1 &
sleep 5 && ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/id_ed25519 -p 2242 -l bazina localhost "echo ok"

# Tail job logs
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod logs steer-strong4 -p dlab-bazina --tail 20

# Check steer-strong4 k=20 progress (how many results written)
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
  "wc -l full_trait_output/trait_steering_attack_jbb_strong_k20/trait/results.jsonl"
# Total expected: 11200 (100 behaviors × 7 alphas × 16 pairs)

# Read sweep results ranked by test AUC
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
"python3 -c \"
import json
d = json.load(open('full_trait_output/all_traits_sweep_v2_olmo3_all_traits/results.json'))
rows = [(v.get('human_test',{}).get('auc',{}).get('mean',0), v.get('val_auc',{}).get('mean',0), m,
         {k: v[k]['auc']['mean'] for k in ['PAIR','PAP','GPTFuzz','PEZ'] if k in v})
        for m, v in d['summary'].items()]
rows.sort(reverse=True)
for rank, (t, val, m, tr) in enumerate(rows[:5], 1):
    print(rank, m, f'val={val:.4f}', f'test={t:.4f}', tr)
\""

# Read steering attack summary
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
"python3 -c \"
import json
d = json.load(open('full_trait_output/trait_steering_attack_jbb_strong_k10/trait/summary.json'))
pp = d['gpt41mini']['per_pair_asr']
pb = d['gpt41mini']['per_behavior_asr']
jb = d['jbb_rubric']['per_pair_asr']
print(f'per_pair={pp:.4f} per_behavior={pb:.4f} jbb={jb:.4f}')
\""

# Check if GCG completed (JBB_GCG dir appears only after all 100 behaviors done)
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
  "ls /dlabscratch1/bazina/HarmBench/results/JBB_GCG/ 2>/dev/null && echo 'GCG done'"

# Run harmbench GCG OLMo-3 merge (after all 4 chunks succeed)
ssh -i ~/.ssh/id_ed25519 -p 2242 -o StrictHostKeyChecking=no bazina@localhost \
  "bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_harmbench_gcg_olmo3_merge.sh"

# Check file-bridge62 status (resume if Suspended)
SUPPRESS_DEPRECATION_MESSAGE=true runai-rcp-prod list jobs -p dlab-bazina 2>&1 | grep "file-bridge62"
```

---

## 11. Outstanding runs to fully fill paper tables

**Status 2026-08-05 (evening): items 3–5 are DONE and filled into `tables/steering_llama.tex`
(uncommitted); items 1–2 are the Gemma work, with GCG generation in flight.**

| # | Cell | Result | How |
|---|---|---|---|
| 3 | steer-$w$ (persona) × WildJB | **0.77** | Already existed in `panelA_wjb_personaw/harmbench/summary.json` (per-pair 0.773, α 0.1–0.5, HB clf). It was never symlinked into `paper_data/`, which is why it looked unrun. No compute. |
| 4 | steer-$w$ (aligned) × HarmBench | **0.657** | Recovered from the completed `harmbench_waligned_tpl_a7` run (44,800 pairs) by joining to baseline labels on `(behavior_id, jailbreak_idx)`; reproduces the 1935-refused / 1245-jailbroken split exactly. Recompute with `full_trait_tools/waligned_refused_asr.py`. No compute. |
| 5 | GCG × JBB (vanilla) | **0.40** | New 100-prompt run. The JBB classic column reads per-pair `jailbroken` from `jbb_attack_activations/`, **not** from `comparison_attacks_jbb/*/summary.json` — the two pipelines disagree (PAIR 0.28 vs 0.09), so the pre-existing GCG 0.31 was not usable. Data: `paper_data/activations/jbb_gcg`. |

Every remaining `--` in the analysis / appendix tables (`pca_llama`,
`refusal_direction_llama`, `subspace_*`) is a structural N/A cell, **not** a missing run.
The two genuine gaps left are both Gemma:

### `tab:detection-gemma` (`tables/detection_gemma.tex`) — Gemma-4-31B detection
1. **GCG on Gemma** — the entire **GCG column is `--` for all 11 methods**. **IN FLIGHT.**
   HarmBench cannot do this: its `.venv` pins transformers 4.45.2, which rejects
   `model_type: gemma4` outright. Standalone replacement written instead (no HarmBench
   source touched):

   | Script | Role |
   |---|---|
   | `full_trait_tools/gcg_gemma.py` | GCG under `.venv_gemma`, HarmBench behaviours + optimizer targets, shardable, resumes from its own jsonl |
   | `full_trait_tools/gcg_gemma_to_harmbench.py` | Converts shard output into the per-behaviour test-case layout the Gemma collector reads |
   | `run_gcg_gemma_shard.sh` | One shard per GPU |
   | `run_gcg_gemma_finish.sh` | Chained onto shard 0: waits for all shards → convert → collect layer-30 activations → judge |

   Launched 2026-08-05 ~21:45 as `gcg-gemma-s0..s19` (400 behaviours, 500 steps, width 512).
   Measured **13.1 s/step** on an A100-80GB (batch 64 is only 9% faster than 16 — compute-bound),
   so ≈1.8 h/behaviour and **≈36 h wall clock**, ETA 2026-08-07 morning. Delete the workloads
   when done. Then run every detector on the new family to fill the column.
2. **GradSafe on Gemma** — the **GradSafe row is `--` across every column**. It was *attempted*,
   not skipped: `baselines_all_attacks_gemma/gradsafe_all_attacks.json` holds HarmBench only,
   n=288 with **n_jb=3**, AUC 1.0, and the checkpoint stalled at 458 rows. The cause is
   Gemma's HarmBench jailbreak rate of **2.0%** (63/3180), not a bug in the GradSafe port —
   `run_all_traits_sweep_v2.py` folds WildJailbreak into *training* only ("val/test/transfer
   eval stay pure HarmBench"), so every Gemma row shares that thin pool; the sweep rows just
   average 50 seeds where GradSafe scores one slice and `balanced_auc` collapses it to 3-vs-3.
   ~85 s/prompt → ~100 GPU-h unsharded. **Open decision:** run as-is for strict comparability
   (and accept an unreportable HarmBench cell), or enrich the slice with all 63 positives plus
   matched negatives (~27 GPU-h, far more stable, needs a footnote).

### `tab:attack` (`tables/steering_llama.tex`) — DONE
Cells 3–5 are filled (see the table at the top of this section). `paper_data/` now links
them: `steering/wildjb/persona_w`, `steering/harmbench/w_aligned`, `activations/jbb_gcg`.
Other steering outputs remain under
`paper_data/steering/{harmbench,jbb,wildjb}/{persona_w,w_aligned,classic}/`.

---


---

## 12. Provenance of paper numbers (verified subset)

For a coauthor who wants to re-plot or re-run a panel. **Every mapping in 12.3 was
checked by comparing the data on disk against the numbers printed in the `.tex`.**
Mappings that did not reproduce, or that could not be traced, are listed separately
in 12.5 — treat those as unknown, not as documented.

**Cluster root:** `/dlabscratch1/bazina/assistant-axis-llama3.1-8B`
(inside a RunAI pod: `/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B`).
Paths below are relative to it. `full_trait_output/` is gitignored — cluster only.

### 12.1 Curated index: `full_trait_output/paper_data/`

Relative symlinks into the raw output dirs, so the tree moves as a unit. Prefer
these over the raw dirs, which also contain superseded runs.

| Path | Contents |
|---|---|
| `paper_data/activations/{harmbench,pair,pap,gptfuzz,pez,gcg,wildjailbreak,jbb_gcg}` | Llama per-pair activations + `classified_responses.jsonl` |
| `paper_data/detection_llama/`, `paper_data/detection_gemma/{trait,raw,pca,baselines}` | detection sweeps |
| `paper_data/analysis/` | Llama mechanistic outputs |
| `paper_data/analysis_gemma/` | Gemma mechanistic outputs (added 2026-08-17) |
| `paper_data/steering/{harmbench,jbb,wildjb}/{persona_w,w_aligned,classic}` | steering-attack ASR runs |
| `paper_data/building_blocks/`, `paper_data/models/` | trait vectors, axes, saved classifiers |

### 12.2 Trait vectors — the input to every mechanistic analysis

| Model | Layer | Path |
|---|---|---|
| Llama-3.1-8B | 16 | `full_trait_output/traits40_vectors/pre_generation_last_token/<filter>/` |
| OLMo-3-7B | 16 | `full_trait_output/traits40_vectors_olmo3_7b/...` |
| Gemma-4-31B | 30 only | `gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter/` |

Gemma's `.pt` files store a **single row (index 0) that *is* layer 30**, not a
layer-indexed tensor, so every reader needs `--trait_row_index 0` with `--layer 30`.

### 12.3 Verified table → data → script

Checked value shown in brackets; it matches the `.tex`.

| Table | Data | Script | Check |
|---|---|---|---|
| `trait_auc_llama` | `paper_data/analysis/single_trait_auc/` (= `persona_individual_auc_report`) | `persona_individual_auc.py` | egalitarian 0.874, 101 seeds → 0.87 |
| `pca_llama` | `paper_data/analysis/pca_decomposition/` (= `pca_direction_decomposition`) | `pca_direction_decomposition.py` | PC1 var 17.65%, weight 0.67, AUC 0.80 |
| `cross_family_traits_llama` trait columns | `full_trait_output/cross_family_traits/results.json` | `cross_family_trait_auc.py` | egalitarian .87, universalist .86, progressive .85 |
| `cross_family_traits_llama` detector-AUC column | `paper_data/detection_llama/consolidated_table.json` | — | 0.854 / 0.842 / 0.817 / 0.790 / 0.708 / 0.641 exact |
| `subspace_ablation_llama` | `full_trait_output/harmbench_logreg/pc_ablation_results.json` | `pc_ablation_study.py` | all-four 0.827, rank 1 of 15 |
| `trait_jailbreak_llama` | `paper_data/analysis/trait_weights.csv` (= `all_traits_sweep_v2/trait_importance.csv`) | `run_all_traits_sweep_v2.py` | rank 1 = paranoid, safe side |
| `trait_auc_gemma` | `paper_data/analysis_gemma/single_trait_auc/` | `persona_individual_auc.py` | forgiving 0.836, 101 seeds |
| `pca_gemma` | `paper_data/analysis_gemma/pca_decomposition/` | `pca_direction_decomposition.py` | PC1 27.0%, weight 0.74 |
| `pca_poles_gemma` | `paper_data/analysis_gemma/pca_poles/` | `pca_component_interpretation.py` | PC1 impatient 0.60 |
| `cross_family_traits_gemma` | `paper_data/analysis_gemma/cross_family/` | `cross_family_trait_auc.py --gemma` | HarmBench forgiving .84 |
| `subspace_ablation_gemma` | `paper_data/analysis_gemma/pc_ablation/` | `pc_ablation_study.py` | PC1+PC2+PC3 0.711 rank 1 |
| `subspace_doseresponse_gemma` | `paper_data/analysis_gemma/subspace_doseresponse/` | `run_trait_subspace_sweep.py` | k=50 avg transfer 0.722 |

`consolidated_table.json` carries a `source` field per method — the authoritative
provenance record for the Llama detection numbers. Prefer reading it over inferring.

**Verified figure** (script writes that exact filename, and the file is on the cluster):

| Figure | Script | Writes |
|---|---|---|
| `persona_layer_dynamics.png` | `persona_layer_dynamics.py` | `full_trait_output/persona_layer_dynamics/` |

### 12.4 Gemma protocol — read before re-running anything Gemma

HarmBench jailbreaks Gemma on only **2.0% of pairs (63 of 3180)**, so every *fitted*
Gemma analysis folds WildJailbreak into the **training pool only**, via
`--augment_classified_path` / `--augment_activations_path` pointing at
`full_trait_output/wildjailbreak_activations_gemma/` (+184 positives, 220 total).
PCA and subspace **bases**, and all **evaluation** pools, stay pure HarmBench —
matching `run_all_traits_sweep_v2.py`'s own augmentation semantics.

Dirs carrying the correct protocol are suffixed **`_hbwjb`**. The un-suffixed Gemma
analysis dirs are HB-only and **superseded** — they produce materially different
conclusions (weight in $w$ moved from PC3/PC4 to PC1). Unfitted quantities
(single-trait AUC, PC variance, poles, per-family AUC) are protocol-independent.

Caveat no augmentation fixes: the pool split is disjoint in both behaviours and
templates, so a Gemma held-out split is ~288 pairs holding a **median of 4**
jailbroken examples (14 of 50 seeds have fewer than 3). Single held-out Gemma AUCs
are correspondingly noisy.

### 12.5 Provenance NOT established — confirm before relying on these

- **`detection_llama`** — the trait row prints 0.86 / 0.83 / 0.80 / 0.63 / 0.71 / 0.85,
  but `consolidated_table.json` holds 0.854 / 0.817 / 0.790 / 0.641 / 0.708 / 0.842.
  Five of six differ by 0.01, so the table came from a different run than the
  consolidated record. **The two tables also disagree with each other**:
  `cross_family_traits_llama`'s detector-AUC column matches the consolidated file
  exactly while `detection_llama` does not, though the caption says they are the
  same quantity. Worth reconciling before submission.
- **`detection_gemma`** — `paper_data/detection_gemma/{trait,raw,pca}` point at the
  `all_traits_sweep_v2_gemma_hbwjb_*` sweeps, but those `results.json` files expose
  only `val_auc`, so the printed per-family AUCs could not be reproduced from them.
- **`subspace_projection_llama`** — no data or script found for the share-of-norm and
  projected/residual AUC columns.
- **`subspace_doseresponse_llama`** — has four constructions (greedy traits, trait PCs,
  random, activation PCs). Only "trait PCs" plausibly corresponds to
  `trait_subspace_sweep/`, and even there k=50 gives 0.844 against 0.835 printed.
  Note the Gemma companion table reports the trait-PC construction only.
- **`refusal_direction_llama`** — no producing script in this repo.
- **All figures except `persona_layer_dynamics.png`** — 9 of 13 do not exist on the
  cluster under the filename the paper uses, and of those that do,
  `raw_vs_trait_classifier_alignment.py` writes no `.png` at all while
  `per_position_probe.py` and `plot_layer_sweep.py` write different filenames
  (`per_position_probe.png`, `all_layers_sweep_plot.png`). The figures were renamed
  or produced elsewhere; the chain is not documented.

### 12.6 Which venv

| Work | Interpreter |
|---|---|
| Anything loading Gemma-4-31B | `.venv_gemma/bin/python` (transformers 5.8) |
| Everything else, incl. CPU analysis of saved activations | `.venv/bin/python` |
| HarmBench attack generation | `/dlabscratch1/bazina/HarmBench/.venv` |

`uv` is not installed on the file-bridge pod — call the interpreters directly.
CPU-only analyses run fine there (56 cores, ~340 GB RAM) and need no GPU.
