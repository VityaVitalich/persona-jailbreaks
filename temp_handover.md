# TEMP HANDOVER — Persona-Subspace Jailbreaking paper (Llama-3.1-8B)

Handoff for a fresh agent continuing work on the **NeurIPS 2026 paper** "Jailbreaks Live in the Persona Subspace". You are working through reviewer-style `\Dominic{}` / `\Viktor{}` comments in the LaTeX, backed by real experiments on the EPFL RCP cluster. Date context: work done 2026-07-25/26.

Also read the persistent memory index first: `/home/dbazinag/.claude/projects/-home-dbazinag-projects-assistant-axis-llama3-1-8B/memory/MEMORY.md` (esp. `project_persona_readout.md`, `feedback_cluster_runai_submit.md`, `feedback_cluster_nfs_rootsquash.md`).

## 1. Repos & layout
- Main repo: `/home/dbazinag/projects/assistant-axis-llama3.1-8B` (branch `master`). Scripts live in `full_trait_tools/`, outputs in `full_trait_output/`. `full_trait_output/paper_data/` exists ONLY on the cluster, not locally.
- **Paper is a SEPARATE git repo** at `paper/` (branch `main`, Overleaf-synced). It is gitignored by the main repo. Sections in `paper/sections/*.tex`, tables in `paper/tables/*.tex`, figures in `paper/figures/`. Main file `paper/neurips_2026.tex`. Section body order: intro → rel_work → method → **interp → detection → steering** → discussion → conclusion → appendix.
- Paper repo state right now: **1 commit ahead of origin** (commit `7a24885 pca stuff` is local, NOT pushed). Working tree clean except 4 untracked `figures/trace_*.png` (pre-existing, not mine). **Do NOT push to origin / Overleaf** unless the user explicitly asks — the user manages Overleaf sync. Commit only when asked.

## 2. Cluster access (IMPORTANT — port-forward is flaky)
- `kubectl` context `rcp-caas-prod`, namespace `runai-dlab-bazina`. Always `export SUPPRESS_DEPRECATION_MESSAGE=true`.
- A persistent CPU pod `file-bridge62-0-0` is always up (72 CPUs, has the repo's `.venv` with torch/sklearn/matplotlib). SSH port-forward (2242) kept dropping after VPN reconnects this session — **prefer `kubectl exec` instead**:
  - Run as the `bazina` user (NFS root-squash): `kubectl exec file-bridge62-0-0 -- setpriv --reuid 315355 --regid 30154 --init-groups bash -c 'cd /dlabscratch1/bazina/assistant-axis-llama3.1-8B && ...'`
  - Pipe a script to the pod's python: `kubectl exec -i file-bridge62-0-0 -- setpriv --reuid 315355 --regid 30154 --init-groups bash -c 'cd /dlab.../assistant-axis-llama3.1-8B && .venv/bin/python' < local_script.py`
  - Copy a file to the pod: `kubectl exec -i file-bridge62-0-0 -- setpriv ... bash -c 'cat > /dlab.../path' < localfile`
  - Fetch a file: `kubectl exec file-bridge62-0-0 -- setpriv ... bash -c 'cat /dlab.../file' > localfile`
  - Long CPU jobs: launch detached `... bash -c 'cd ...; MPLCONFIGDIR=/tmp/mpl nohup .venv/bin/python script.py > /tmp/x.log 2>&1 & echo launched'` then poll `/tmp/x.log`. (kubectl exec holds open if the bg proc inherits fds — my tool times out at 120s and backgrounds it; the pod proc still runs. Verify with `pgrep -af script`.)
- **GPU jobs**: submit a RunAI `TrainingWorkload` via `kubectl apply -f /tmp/<name>.yaml`. Template (adapt name + run script path): image `ghcr.io/jkminder/dlab-runai-images/pytorch:master`, gpu 1, PVC `dlab-scratch` at `/mnt`, uid/gid 1000, `arguments: bash /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_<x>.sh`. A run script does `set -a; source .env; set +a; cd repo; .venv/bin/python full_trait_tools/<x>.py`. **Use `.venv/bin/python`, NOT `uv run`** — `uv run` fails building the project (missing README.md).
- GPU is a **heavily shared/slow** cluster right now — an 80-generation job took 15–20 min. Poll pod phase `kubectl get pod <name>-0-0 -o jsonpath='{.status.phase}'`; beware stale reads (I once caught a stale "Succeeded" and killed a still-running job). The script writes its JSON only at the END, so also check output-file mtime. Delete workloads when done: `kubectl delete trainingworkload.run.ai/<name>`.
- Model: `meta-llama/Llama-3.1-8B-Instruct`, layer 16 pre-generation last-token activations. 229 persona-trait vectors, difference-of-means, unit-normalized. Detector = logreg on 229 trait projections (`trait_matrix` 229×4096), threshold in pkl, sign +1.

## 3. Paper conventions
- `\claudewritten{...}` = green prose (Claude-written, humans must rewrite before submission). `\clauderesponse{...}` = reply to a `\Dominic`/`\Viktor` comment. `\Dominic{}`/`\Viktor{}` = reviewer comments (violet/cyan). When resolving pull conflicts where a human has **de-greened** (removed `\claudewritten`) prose, keep their de-greening and wrap only NEW unreviewed content in `\claudewritten` so it flags for review.
- Numbers must trace to result files. Don't push, don't commit unless asked. Be honest about weak/negative results.

## 4. Dominic/Viktor comment status (audited)
ANSWERED: interp:119 (per-layer persona dynamics), interp:143 (cross-family transfer), interp:166 (refusal direction in trait subspace, appendix), interp:168 (per-token persona/readout — the big one, see §5). Framing notes (interp:4, steering:4 Viktor) fulfilled; detection:4 Viktor "sell as practical" = tone check. OPEN: **interp:17 Dominic** — restyle `fig_jailbreak_pca` (`fig_jailbreak_pca.png`) to be "prettier like Assistant Axis" (aesthetic; there's an old `preview_style.py` with an Assistant-Axis palette in a stale scratchpad as a starting point). SCAFFOLDING to delete before submission: `neurips_2026.tex:115-117` (`\viktor`/`\dominic` legend examples).

## 5. Work done this session (all committed in paper `7a24885`)
### (a) L168 persona/readout localization — DONE, in paper
Finding: the per-token persona/detector signal is content-driven; it does NOT cleanly separate outcome along the prompt. The compliance decision **localizes to the pre-generation readout token**. Benign fires HIGHEST at readout (compliance-intent, not harm) — **deliberately kept OUT of the paper** (it made the detector look like "just a compliance classifier"; the user chose to drop benign). Instead the paper shows: (1) `fig:persona-readout` = trajectory plot, 3 harmful groups (direct/failed/worked), neutral "trait-subspace projection" y-axis; (2) `fig:persona-readout-probe` = per-position **fresh-probe** AUC (content ~0.6, readout 0.94) proving localization is a model property not a detector artifact; (3) appendix single-example `fig:persona-readout-example`. Scripts: `full_trait_tools/aggregate_readout_heldout.py`, `per_position_probe.py`, `failed_jb_trace.py`, `aggregate_readout*.py`, `benign_readout.py`. Figures in `paper/figures/fig_persona_readout_*.png`.
### (b) Per-layer all-traits classifier line — DONE, in paper
Added bold crimson "all-229 trait classifier (logreg)" line to `fig:persona-layers` (`persona_layer_dynamics.png`). CRITICAL: must use **behavior- AND template-disjoint** CV (a random or behavior-only split leaks template→success → inflated 0.97). Correct = **0.868 @ L16, peak 0.895 @ L17**, matches paper's ~0.86–0.88. Script `full_trait_tools/persona_layer_dynamics.py` (has `clf_auc` with disjoint split + joblib parallel over layers, runs on file-bridge CPU ~30s). Caption + text updated (interp.tex, new content wrapped in `\claudewritten`).
### (c) Table 5 moved — DONE
`\input{tables/cross_family_traits_llama}` moved from interp.tex to **detection.tex** (right after `\input{tables/detection_llama}`), because it's a classifier/transfer result. Its discussion stays in interp (cross-references `tab:cross-family`).
### (d) PC-steering qualitative table (Dominic pca_llama comment) — DONE, in appendix
`paper/tables/pc_steering_qualitative.tex` (`tab:pc-steering`) + appendix section `app:pc-steering`. Steers a benign prompt along ±each PC (add `coeff*||h||*PC` to layer-16 residual at GENERATED positions only — steering all positions breaks coherence past ±0.4). Honest quality after 3 tuned runs: **PC1 vivid** (disinhibition, ±1.0), **PC2 legible** (evil–principled, neighbor prompt ±1.3), **PC3 clear** (introspective–reserved, meaning prompt ±1.3), **PC4 subtle** (ritualistic–casual, morning prompt ±1.3 — flagged as weakest). Excluded all degenerate high-coefficient rows. Data: `full_trait_output/pc_qualitative/steer_pc_qualitative{,_v2,_v3}.json` (v1=±1.0 3 prompts, v2=±1.5, v3=±1.3 tailored). Script `full_trait_tools/steer_pc_qualitative.py` (recomputes PC1–4 from activation cloud, orients by trait poles edgy/principled/introspective/ritualistic).

## 6. Known inconsistency to fix
`paper/tables/pca_llama.tex` line 7 `\clauderesponse` was reverted by the user to the OLD text ("Done for PC1… two benign prompts… PC2 subtler; say if you'd like it included too") — but the table now covers ALL 4 PCs. This is stale; the user reverted it intentionally so DON'T touch unless they ask. The appendix prose + table itself already correctly describe all 4 PCs.

## 7. Likely next steps (ask user before doing)
- The open interp:17 aesthetic restyle of `fig_jailbreak_pca`.
- User may want PC4 dropped from `tab:pc-steering` (keep only PC1–3) — a one-line change.
- Human pass to de-green remaining `\claudewritten` prose before submission.
- Possibly push paper commit to origin/Overleaf (ONLY if user asks).

## 8. Cluster housekeeping
No GPU workloads currently running (all deleted). `file-bridge62-0-0` stays up. If you resubmit `steer-pc-qual` etc., delete when done.
