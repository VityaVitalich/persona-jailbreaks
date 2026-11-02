# Persona-jailbreaks results dashboard

Live at https://vityavitalich.github.io/persona-jailbreaks/.

`build_data.py` collects result files that are already saved into one `data.json`; `index.html` + `style.css` render
it (facet chips, tables, inline-SVG charts, no external scripts). Nothing is recomputed. The only arithmetic is
re-aggregation of saved numbers: means over saved per-draw or per-set AUCs, rates over saved per-prompt judge labels,
and differences of two saved point estimates. Every AUC is the folded AUC max(a, 1 − a).

## Run and deploy

```bash
PJB_ROOT=~/persona-vector-jbing python3 dashboard/build_data.py   # the research checkout: this repo withholds chem/bio response
                                         # text, which the Gemma looks_like_text rule would count as garbage
python3 -m http.server 8811 -d dashboard # then open http://127.0.0.1:8811/
PJB_ROOT=~/persona-vector-jbing bash dashboard/deploy.sh   # rebuild, refuse secrets / response fields, force-push gh-pages
```

`PJB_ROOT=<dir>` points the build at another checkout. For example, `PJB_ROOT=~/persona-vector-jbing` reads new q1 files
straight from the research checkout before they are committed here. Without `data/release/`, the release files are
read from `~/pjb_release_staging`.

## Tabs

1. **Detection: the q1 protocol only.**
   - **Protocol:** each detector's readout (layer × tail position × C) is tuned by 5-fold CV on the training set, refit,
     and scored on the transfer sets.
   - **Sources:** `full_trait_output/q1_detection/out/q1_{model}_{train}.json` (removal *before* training,
     `q1_sweep.py`) and `q1_posthoc_{model}_{train}.json` (removal *after* training, `q1_posthoc.py`), for model ∈
     {llama, olmo, qwen, gemma} and train ∈ {hb, hbwjb}.
   - **Pending files:** a status grid shows which of the 16 files exist. A missing file shows as "pending"; nothing is
     filled in from other runs.
   - **Facets:** model, training set (HB / HB + WJB), removal timing (before / after training), component (full /
     persona removed / only persona / random K-dim span removed), readout (same as full / re-selected; before
     training only), show (folded AUC / Δ vs full), and detector.
   - **Table layout:** with one model, detectors are rows. With several models, there is one table per detector, with
     models as rows.
   - **Columns:** PAP, GCG, PAIR, GPTFuzz, PEZ and the average over those five (`avg_sets`) with its cluster-bootstrap
     CI. WJB is an extra column when it is a test set (HB-only training) and is not averaged. Also validation AUC, and
     the selected layer / position / C. Post-hoc rows add the share of the detector's weight in the persona span.
   - **Below the tables:** an all-components table and a layer × position heatmap of validation and transfer AUC.
2. **Steering.** Llama, OLMo and Gemma steering ASR, degenerate rate and coherent ASR with saved CIs.
   - **Gemma setting:** norm 572 (default; 1.98× a typical token, as for Llama and OLMo) or the earlier norm-258 paper cells. At norm
     572 every steered row uses the same prompts (HB refused-300, JBB 100, WildJB 543 refused): PersonaJB, refusal and Δμ in full /
     persona removed / persona only, the 10 most predictive traits, the 10 most steering-effective traits of the 240-trait sweep,
     random and unsteered. Also at norm 572: the layer sweep (L30 / L35 / L40 / L45, same prompts) and the 240-trait sweep
     (predictive AUC vs clean ASR; its own pool of 400 HarmBench behaviours without templates).
   - **Facets:** benchmark, cumulative vs single dose, α, scoring, direction, component, snapshot and run.
   - **Snapshots:** Llama rows carry their snapshot (official `0e9e39f2` vs mirror `247cb7c3`, not comparable).
   - **Gemma:** only the looks_like_text-corrected cells.
   - **Also:** directional ablation.
3. **Persona erasure.** Is persona information needed to detect jailbreaks, and does removing it break unrelated tasks?
   - **Edits** of the q1 readout activations (raw-probe and PersonaJB cells, HB + WJB training), probes retrained afterwards: persona
     span projected out; projection erasure (span + cross-covariance, so no linear readout recovers the persona coordinates); LEACE.
   - **Controls:** rank-matched uniform random subspaces (3 draws each, not fitted to data).
   - **Read-outs:** persona R², jailbreak transfer AUC (Δ and paired CI vs no edit and vs the matched control), HarmBench
     semantic-category AUC, 9 general tasks (SST-2, AG News, cities, CoLA, MNLI, MMLU answer correctness, negated cities, common
     claims, arithmetic).
4. **Persona share of space.**
   - **Spans:** how much of the refusal, Δμ, w_in / w_out, normality and PC directions lies in the persona span vs
     Alpaca / Wikipedia / No Robots / Gaussian / PCA spans.
   - **Also:** the q1 per-cell persona shares and the post-hoc detector-weight shares.
   - **Caveat:** Gemma panels carry the massive-dimension caveat.
5. **Sources.** Every file `build_data.py` read.

No Robots span removal is labelled "NOT a valid control" wherever it appears. No Robots activations are persona-rich,
so removing their span removes persona content too.

## Not shown

**Superseded:**
- The earlier single-file `q1_gemma.json`, made with the old refusal recipe.

**Kept in the repo, not on the dashboard** (Viktor's decision, 2026-10-28):
- Every other detection result: hb_select, the single-set / persona_four / persona_variants runs, probe methods,
  Dominic's OLMo activations, the trait-PC and activation-PCA sweeps, the norm-share removal controls, the clean-repo
  release run, and the Qwen per-position and layer sweeps.

**Signed AUCs only:**
- `norm_share_controls/subspace_tests.json`, `direction_controls_data.json`.

**Steering:**
- Raw (uncorrected) Gemma summary ASRs, superseded by `results/gemma_steering_cell_cis.json`.
- `outputs_gemma_singledose/`: no corrected summary is saved, and making one would mean re-scoring.
- `norm_share_controls/probe/`: not jailbreak ASR.

## Source file → tab / section

| File | Tab | Section |
|---|---|---|
| `full_trait_output/q1_detection/q1_sweep.py` | Detection | q1 sweep (removal before training): writer |
| `full_trait_output/q1_detection/q1_posthoc.py` | Detection | q1 post-hoc (removal after training): writer |
| `full_trait_output/norm_share_controls/steer_bench/steer_nopersona_bench.json` | Steering | steer_bench (official snapshot, refused-300 / JBB-100 / WildJB-298) |
| `full_trait_output/norm_share_controls/steer_bench/personaonly_harmbench.json` | Steering | steer_bench persona-only arms |
| `full_trait_output/norm_share_controls/steer_v2_results` | Steering | controls v2 run (official snapshot, HB refused-300) |
| `full_trait_output/norm_share_controls/nopersona_steer_results.json` | Steering | persona-removed steering pairs (corrected scoring, normal CI) |
| `full_trait_output/outputs_controls_cumulative` | Steering | controls, cumulative (mirror snapshot) |
| `full_trait_output/steer_fixed_directions_controls` | Steering | controls, single dose α=0.5 (mirror snapshot) |
| `full_trait_output/outputs_alpha_sweep` | Steering | single-dose α sweep (mirror snapshot) |
| `results/steering_cell_cis.json` | Steering | paper Llama cells (steering_cell_cis.json) |
| `results/wjb_refused_cells.json` | Steering | paper Llama cells, WildJB refused-only (wjb_refused_cells.json) |
| `results/gemma_steering_cell_cis.json` | Steering | Gemma steering cells, gibberish-corrected (gemma_steering_cell_cis.json) |
| `results/gemma_steering_cell_cis_classic.json` | Steering | Gemma classic attacks (no steering) |
| `full_trait_output/norm_share_controls/gemma/steer/steer_split_gemma.json` | Steering | Gemma persona-component steering (HB refused-300) |
| `results/ablation_cells.json` | Steering | Directional ablation (Llama) |
| `full_trait_output/steer_port/outputs_gemma_norm572`, `outputs_gemma_norm572_traits` | Steering | Gemma norm 572: figure, persona-component Δ, cumulative curves |
| `full_trait_output/steer_port/outputs_gemma_layers` | Steering | Gemma layer sweep (L35 / L40 / L45) |
| `full_trait_output/steer_port/outputs_gemma_alltraits` + `persona_individual_auc_gemma_evalmerge/results.json` | Steering | Gemma 240-trait sweep vs predictive AUC |
| `full_trait_output/q1_detection/out/erase_{model}_{cell}.json` | Persona erasure | edits, controls and read-outs (`q1_erase.py`) |
| `data/release/derived/llama/analysis/persona_span.json` | Persona share | Clean-repo release persona span |
| `data/release/derived/gemma/analysis/persona_span.json` | Persona share | Clean-repo release persona span |
| `full_trait_output/norm_share_controls/results.json` | Persona share | results |
| `full_trait_output/norm_share_controls/w_out_in_spans.json` | Persona share | w_out_in_spans |
| `full_trait_output/norm_share_controls/gemma/w_out_in_spans.json` | Persona share | w_out_in_spans |
| `full_trait_output/norm_share_controls/pcs_in_persona.json` | Persona share | pcs_in_persona |
| `full_trait_output/norm_share_controls/persona_part_spans.json` | Persona share | persona_part_spans |
| `full_trait_output/norm_share_controls/gemma/persona_part_spans.json` | Persona share | persona_part_spans |
| `full_trait_output/norm_share_controls/persona_part_pervector.json` | Persona share | persona_part_pervector |
| `full_trait_output/norm_share_controls/gemma/persona_part_pervector.json` | Persona share | persona_part_pervector |
| `full_trait_output/norm_share_controls/what_is_dmu_out.json` | Persona share | what_is_dmu_out |
| `full_trait_output/norm_share_controls/gemma/what_is_dmu_out.json` | Persona share | what_is_dmu_out |
| `full_trait_output/norm_share_controls/why_dmu_no_drop.json` | Persona share | why_dmu_no_drop |
