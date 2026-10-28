# Persona-jailbreaks results dashboard

A static, local dashboard for the persona-jailbreaks project ("Jailbreaks Live in the Persona Subspace"). It follows
the MR-Eval dashboard pattern: `build_data.py` collects result files that are already saved on disk into one
`data.json`, and `index.html` + `style.css` render it with facet chips, tables and charts. There is no build step and no
external JS or CSS; the charts are inline SVG.

## Files

- `build_data.py`: reads the saved result files and writes `data.json`. It never recomputes a score. The only arithmetic
  is re-aggregation of saved numbers: means over saved per-seed, per-draw or per-family AUCs, rates over saved per-prompt
  judge labels, and differences of two saved point estimates.
- `data.json`: generated, about 0.9 MB.
- `index.html`, `style.css`: the page.

## Run

```bash
cd ~/persona-jb-dashboard
python3 build_data.py                      # rebuild data.json
python3 -m http.server 8811 --bind 127.0.0.1
# open http://127.0.0.1:8811/
```

When `~/persona-vector-jbing/full_trait_output/q1_detection/out/q1_gemma.json` (or any `q1_<model>.json`) lands, run
`python3 build_data.py` again. The Q1 setup, its layer × position heatmaps and the Q1 persona-share table fill in
automatically. Until then, the page marks Q1 as pending. `--smoke-q1 PATH --out OTHER.json` is for development only:
it loads a smoke file and shows a red SMOKE banner. The default build never reads smoke files.

## Conventions enforced

- **Folded AUC only.** Every AUC is max(a, 1 − a). Signed fields are dropped, for example `mean_signed`,
  `avg_transfer_signed`, pca_k_sweep `auc` / `auc_paper`, few_directions `hb` / `xf_signed`, and normality T3 `[1]`.
  Files whose AUCs are all signed are excluded (see below).
- **Models are rows; datasets and conditions are columns.** The Detection tab adds a second table whose columns are the
  component conditions.
- **No provenance fallbacks.** A cell missing for a model or setup shows "—". Δ vs full / Δ vs raw use the saved paired
  CI when one exists. Otherwise they show the difference of the two saved point estimates with no CI, and the tooltip
  says so.
- **One protocol per setup.** Every detection source sits in its own Setup with a protocol line. Results from different
  protocols never share a table. Where two files share a protocol and the build checks that their numbers agree, they are
  merged (persona_four + persona_variants; remove_persona_all_controls + remove_nonpersona_span).
- No Robots span removal is labelled **"NOT a valid control"** wherever it appears.

## Tabs

1. **Detection.** Pick a Setup, then Model (multi), Training set, Component (full / persona removed / only persona /
   random-span null / other spans), Readout (Q1: re-selected vs fixed at the full detector's layer and position; Qwen:
   tail position), C, Show (folded AUC / Δ vs full / Δ vs raw probe) and Detector (multi, grouped).
   - The main table has one row per model × detector. Columns: folded AUC per transfer set, average transfer with its
     saved CI, validation AUC, and the selected layer / position / C or other notes.
   - Under it: a dot chart of the average, an all-conditions table, a k-sweep line chart for families with at least 4
     values of k, a layer × position heatmap of validation / transfer AUC for setups that save a grid, with the
     selected cell outlined (Q1, hb_select, Qwen sweep), and, for single-set training, the complementarity table.
   - Setups:
- `q1`: Q1 sweep: component removal (HB+WJB train, 5-fold CV readout selection)
- `hbsel`: Readout selection (single 80/20 val split): PersonaJB vs raw
- `hbsel_cv`: Readout selection (5-fold template-grouped CV): PersonaJB vs raw
- `single`: Single-set training (paper readout, C on train-set val split)
- `dom`: OLMo-3-7B on Dominic's activations (single-set training)
- `olmo_layers`: OLMo-3-7B raw probe: layer sweep (single-set training)
- `paper50`: Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries)
- `removal50`: Persona-removal controls (50 HB pool splits, C=10, dictionary detectors)
- `removal10`: Persona-part tests (10 HB pool splits, C=10, Llama)
- `dir50`: Single directions with persona component removed (50 HB pool splits)
- `few`: Few directions: k best train-AUC directions per dictionary (Llama, 50 splits, C=10)
- `single_vec`: Single ordinary vectors as detectors (h·v, in-sample, Llama)
- `normality`: Benign-vs-harmful detector on plain prompts (normality test, Llama)
- `refdmu`: Refusal / Δμ direction per family, intact vs persona removed (in-sample)
- `release`: Clean-repo release run (50 splits, config chosen on train-pool val)
- `qwen_pos`: Qwen3-32B L48 per tail position (49 pool splits, C=10, HB+WJB)
- `qwen_sweep`: Qwen3-32B layer × position sweep (20 pool splits, C=10)
- `probe_methods`: Probe-method sweep: single-set and leave-one-family-out (Llama L16, Qwen L48)
- `pcak_llama`: Activation PCA(k) sweep, Llama protocol (50 seeds)
- `pcak_gemma`: Activation PCA(k) sweep, Gemma protocol (50 seeds)
2. **Steering.** Facets: model, benchmark (HarmBench refused / JBB / WildJB / WildJB refused), dose (cumulative ≤ α
   vs single dose at α), α, scoring (harness / corrected / Gemma looks_like_text corrected), metric (ASR, degenerate
   rate, coherent ASR, saved paired Δ), direction group, component variant, snapshot and run.
   - Every Llama row carries its snapshot tag: official `0e9e39f2`, mirror `247cb7c3`, or "not stated in files". A
     banner explains why the two are not comparable.
   - For Gemma, only the gibberish-corrected cells are used.
   - Below the table: a dot chart with CIs, cumulative ASR-vs-α curves (re-aggregated from per-prompt min-α), single-dose
     α curves, a collapsible directional-ablation table and a table of runs with their protocols.
3. **Persona share of space.** Squared-norm share of refusal, Δμ, w_in / w_out, normality and PCs in the persona span
   vs Alpaca / Wikipedia / No Robots / Gaussian / PCA spans.
   - Includes: null distributions, removal-span statistics, the persona part of ordinary activations (with per-vector
     scatter), per-PC shares, the first norm-share controls, what Δμ's remainder is, the clean-repo release shares, and
     the Q1 `info` persona_share fields (pending).
   - Gemma panels carry the massive-dimension caveat and the 0.045 random null.
4. **Sources.** Every file read, with its tab and section.

## Source file → tab / section

| File | Tab | Section | Note |
|---|---|---|---|
| `~/persona-vector-jbing/full_trait_output/q1_detection/q1_sweep.py` | Detection | Q1 sweep (schema) (setup `q1`) | writer of q1_{model}.json; docstring = protocol |
| `~/persona-vector-jbing/full_trait_output/q1_detection/out/q1_{model}.json` | Detection | Q1 sweep (setup `q1`) | PENDING: not on disk at build time; tab shows 'pending' |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_llama.json` | Detection | Readout selection (single 80/20 val split): PersonaJB vs raw (setup `hbsel`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_gemma.json` | Detection | Readout selection (single 80/20 val split): PersonaJB vs raw (setup `hbsel`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_olmo.json` | Detection | Readout selection (single 80/20 val split): PersonaJB vs raw (setup `hbsel`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_qwen.json` | Detection | Readout selection (single 80/20 val split): PersonaJB vs raw (setup `hbsel`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_llama_cv.json` | Detection | Readout selection (5-fold template-grouped CV): PersonaJB vs raw (setup `hbsel_cv`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_gemma_cv.json` | Detection | Readout selection (5-fold template-grouped CV): PersonaJB vs raw (setup `hbsel_cv`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_olmo_cv.json` | Detection | Readout selection (5-fold template-grouped CV): PersonaJB vs raw (setup `hbsel_cv`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/hb_select_out/hb_select_qwen_cv.json` | Detection | Readout selection (5-fold template-grouped CV): PersonaJB vs raw (setup `hbsel_cv`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/persona_four.json` | Detection | Single-set training (paper readout, C on train-set val split) (setup `single`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/persona_variants.json` | Detection | Single-set training (paper readout, C on train-set val split) (setup `single`) | PCA_k / top_k / rand_k / concat variants |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/four_models.json` | Detection | Single-set training (paper readout, C on train-set val split) (setup `single`) | raw/Gaussian: identical to persona_four (verified at build); only C values read |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/dom_compare/dom_olmo_compare.json` | Detection | OLMo-3-7B on Dominic's activations (single-set training) (setup `dom`) |  |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/olmo_probe.json` | Detection | OLMo-3-7B raw probe: layer sweep (single-set training) (setup `olmo_layers`) | layer_sweep only; 'transfer' at L16 duplicates persona_four raw |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_gemma.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=10; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_gemma_C0.01.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=0.01; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_gemma_C0.1.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=0.1; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_gemma_C1.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=1; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=10; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_C0.01.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=0.01; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_C0.1.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=0.1; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_C1.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=1; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_extra_controls.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=10; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_lda_C0.01.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=0.01; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_lda_C0.1.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=0.1; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_lda_C1.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=1; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_lda_C10.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=10; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/summary_llama_subspace_controls.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | C=10; signed fields dropped |
| `~/persona-vector-jbing/full_trait_output/trait_pc_sweep/aggregate.json` | Detection | Paper protocol: 50 HB pool splits (trait-PC sweep and subspace dictionaries) (setup `paper50`) | val-C rows (C chosen per seed on inner val) |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/remove_persona_all_controls.json` | Detection | Persona-removal controls (50 HB pool splits, C=10, dictionary detectors) (setup `removal50`) | conditions: persona / No Robots span / random span |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/remove_nonpersona_span.json` | Detection | Persona-removal controls (50 HB pool splits, C=10, dictionary detectors) (setup `removal50`) | conditions: Wikipedia, Wikipedia−P, No Robots−P, Alpaca−P spans ('persona' and 'random' duplicate the file above, checked) |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/remove_persona_all_controls.json` | Detection | Persona-removal controls (50 HB pool splits, C=10, dictionary detectors) (setup `removal50`) | conditions: persona / No Robots span / random span |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/remove_nonpersona_span.json` | Detection | Persona-removal controls (50 HB pool splits, C=10, dictionary detectors) (setup `removal50`) | conditions: Wikipedia, Wikipedia−P, No Robots−P, Alpaca−P spans ('persona' and 'random' duplicate the file above, checked) |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/pcs_persona_component.json` | Detection | Persona-part tests (10 HB pool splits, C=10, Llama) (setup `removal10`) |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/persona_part_test.json` | Detection | Persona-part tests (10 HB pool splits, C=10, Llama) (setup `removal10`) | det_out_hb / det_out_xf only; single-vector AUCs go to Persona share |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/appendix_controls_data.json` | Detection + Persona share | Single directions with persona component removed (50 HB pool splits) / direction share (setup `dir50`) | auc block → Detection; share blocks → Persona share |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/appendix_controls_data.json` | Detection + Persona share | Single directions with persona component removed (50 HB pool splits) / direction share (setup `dir50`) | auc block → Detection; share blocks → Persona share |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/few_directions.json` | Detection | Few directions: k best train-AUC directions per dictionary (Llama, 50 splits, C=10) (setup `few`) | xf_folded only; hb and xf_signed are signed AUC → dropped |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/single_vector_auc.json` | Detection | Single ordinary vectors as detectors (h·v, in-sample, Llama) (setup `single_vec`) |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/normality_in_persona.json` | Detection + Persona share | Benign-vs-harmful detector on plain prompts (normality test, Llama) / normality direction share (setup `normality`) | T3 → Detection (signed field dropped); T2.share → Persona share |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/refdmu_tables.json` | Detection | Refusal / Δμ direction per family, intact vs persona removed (in-sample) (setup `refdmu`) | A block only; B block (d′, within-group AUC) not shown |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/refdmu_tables.json` | Detection | Refusal / Δμ direction per family, intact vs persona removed (in-sample) (setup `refdmu`) | A block only; B block (d′, within-group AUC) not shown |
| `~/pjb_release_staging/derived/llama/detection.json` | Detection | Clean-repo release run (50 splits, config chosen on train-pool val) (setup `release`) |  |
| `~/pjb_release_staging/derived/gemma/detection.json` | Detection | Clean-repo release run (50 splits, config chosen on train-pool val) (setup `release`) |  |
| `~/persona-vector-jbing/full_trait_output/qwen3_32b/transfer/detect_transfer.json` | Detection | Qwen3-32B L48 per tail position (49 pool splits, C=10, HB+WJB) (setup `qwen_pos`) |  |
| `~/persona-vector-jbing/full_trait_output/qwen3_32b/sweep/out/sweep.json` | Detection | Layer × position heatmap (Qwen sweep) (setup `qwen_sweep`) |  |
| `~/persona-vector-jbing/full_trait_output/qwen3_32b/probe_methods/out/probe_methods.json` | Detection | Probe-method sweep: single-set and leave-one-family-out (Llama L16, Qwen L48) (setup `probe_methods`) |  |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/c10_llama_k229.json` | Detection | Activation PCA(k) sweep, Llama protocol (50 seeds) (setup `pcak_llama`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/c10_llama_raw.json` | Detection | Activation PCA(k) sweep, Llama protocol (50 seeds) (setup `pcak_llama`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/sweep_llama.json` | Detection | Activation PCA(k) sweep, Llama protocol (50 seeds) (setup `pcak_llama`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/sweep_llama_k229.json` | Detection | Activation PCA(k) sweep, Llama protocol (50 seeds) (setup `pcak_llama`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/c10_gemma_k240.json` | Detection | Activation PCA(k) sweep, Gemma protocol (50 seeds) (setup `pcak_gemma`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/c10_gemma_k240_unbal.json` | Detection | Activation PCA(k) sweep, Gemma protocol (50 seeds) (setup `pcak_gemma`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/c10_gemma_raw.json` | Detection | Activation PCA(k) sweep, Gemma protocol (50 seeds) (setup `pcak_gemma`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/c10_gemma_raw_unbal.json` | Detection | Activation PCA(k) sweep, Gemma protocol (50 seeds) (setup `pcak_gemma`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/sweep_gemma.json` | Detection | Activation PCA(k) sweep, Gemma protocol (50 seeds) (setup `pcak_gemma`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/pca_k_sweep/sweep_gemma_k240.json` | Detection | Activation PCA(k) sweep, Gemma protocol (50 seeds) (setup `pcak_gemma`) | auc_sym only (signed auc / auc_paper dropped) |
| `~/persona-vector-jbing/full_trait_output/olmo3_7b/detect/complement/complement.json` | Detection | Complementarity: does PersonaJB catch what the raw probe misses? |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/steer_bench/steer_nopersona_bench.json` | Steering | steer_bench (official snapshot, refused-300 / JBB-100 / WildJB-298) | Llama L16, cumulative α 0.1→0.5 (each prompt stops at its first jailbreak), greedy, gpt-4.1-mini in-loop judge, 1 sample |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/steer_bench/personaonly_harmbench.json` | Steering | steer_bench persona-only arms | asr for arms missing from steer_nopersona_bench; paired vs_full diffs |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/steer_v2_results` | Steering | controls v2 run (official snapshot, HB refused-300) | Per-generation files (steer_fixed_direction.py), Llama L16, cumulative α 0.1→0.5, HB refused-300 seed 0, harness scoring |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/nopersona_steer_results.json` | Steering | persona-removed steering pairs (corrected scoring, normal CI) | nopersona_steer_results.py: corrected ASR (char-degenerate rule) with vs without the persona component, HB refused-300,  |
| `~/persona-vector-jbing/full_trait_output/outputs_controls_cumulative` | Steering | controls, cumulative (mirror snapshot) | Per-generation results.jsonl + summary.json (config.model = mirror 247cb7c3), Llama L16, HB refused-300 seed 0, cumulati |
| `~/persona-vector-jbing/full_trait_output/steer_fixed_directions_controls` | Steering | controls, single dose α=0.5 (mirror snapshot) | Per-generation results.jsonl + summary.json (config.model = mirror 247cb7c3), Llama L16, HB refused-300 seed 0, single d |
| `~/persona-vector-jbing/full_trait_output/outputs_alpha_sweep` | Steering | single-dose α sweep (mirror snapshot) | outputs_alpha_sweep/a{α}: single dose at each α (0.1..0.5), PersonaJB / refusal / Δμ, HB refused-300 seed 0, mirror snap |
| `~/persona-vector-jbing/results/steering_cell_cis.json` | Steering | paper Llama cells (steering_cell_cis.json) | Paper steering table cells: cumulative α (PersonaJB ran to 0.7 and was recut), harness scoring, pools HB refused 1935 (1 |
| `~/persona-vector-jbing/results/wjb_refused_cells.json` | Steering | paper Llama cells, WildJB refused-only (wjb_refused_cells.json) | Same runs as the paper cells, restricted to the 975 (traits: 200) WildJB prompts the unsteered model refused; CI as save |
| `~/persona-vector-jbing/results/gemma_steering_cell_cis.json` | Steering | Gemma steering cells, gibberish-corrected (gemma_steering_cell_cis.json) | extract_gemma_steering_cells.py: Gemma-4-31B-it L30, cumulative α 0.1→0.5, gpt-4.1-mini judge; jailbreaks failing the lo |
| `~/persona-vector-jbing/results/gemma_steering_cell_cis_classic.json` | Steering | Gemma classic attacks (no steering) | Attack ASR on HarmBench test cases, CI as saved (bootstrap). |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/steer/steer_split_gemma.json` | Steering | Gemma persona-component steering (HB refused-300) | norm_share_controls/gemma/steer (plot_steer_gemma.py): HB refused-300 (pool seed 0 of the 3117), cumulative α 0.1→0.5, c |
| `~/persona-vector-jbing/results/ablation_cells.json` | Steering | Directional ablation (Llama) | refused / still-jailbroken / degenerate rates with CIs as saved |
| `~/pjb_release_staging/derived/llama/analysis/persona_span.json` | Persona share | Clean-repo release persona span |  |
| `~/pjb_release_staging/derived/gemma/analysis/persona_span.json` | Persona share | Clean-repo release persona span |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/results.json` | Persona share | results |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/w_out_in_spans.json` | Persona share | w_out_in_spans |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/w_out_in_spans.json` | Persona share | w_out_in_spans |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/pcs_in_persona.json` | Persona share | pcs_in_persona |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/persona_part_spans.json` | Persona share | persona_part_spans |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/persona_part_spans.json` | Persona share | persona_part_spans |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/persona_part_pervector.json` | Persona share | persona_part_pervector |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/persona_part_pervector.json` | Persona share | persona_part_pervector |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/what_is_dmu_out.json` | Persona share | what_is_dmu_out |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/what_is_dmu_out.json` | Persona share | what_is_dmu_out |  |
| `~/persona-vector-jbing/full_trait_output/norm_share_controls/why_dmu_no_drop.json` | Persona share | why_dmu_no_drop |  |

## Excluded or unclear (not shown), with reasons

**Signed AUC only**
- `norm_share_controls/subspace_tests.json`: every AUC is signed. The memory note also says its full-space LDA reference
  overfits.
- `norm_share_controls/direction_controls_data.json`: `ref` / `dmu` are signed projection AUCs. Its shares duplicate
  appendix_controls_data.

**Superseded**
- `olmo3_7b/detect/olmo_probe_without_pair.json`, `olmo_gaussian_without_pair.json`: earlier runs with PAIR dropped. The
  OLMo PAIR run now exists, and persona_four / four_models include it.
- `norm_share_controls/results.json` `B_directions_in_persona.shuffled_*`: the pooled permutation is inflated. The
  stratified nulls in appendix_controls_data are shown instead.
- `outputs_gemma_steer/**/summary.json`, `arms_summary_*.json`: raw Gemma ASRs, replaced by the looks_like_text-corrected
  `results/gemma_steering_cell_cis.json`.
- `norm_share_controls/gemma/steer/split_steer_results.json`: same cells as `steer_split_gemma.json`, which adds CIs.

**Duplicates of a shown source**
- `four_models.json` cells: identical to persona_four, checked at build. Only its C values are read. Its raw − Gaussian
  Bonferroni Δ is not shown.
- `norm_share_controls/persona_row_check.json` (both models): reproduces the paper50 PersonaJB row.
- `pcs_persona_component_50.json`: identical to the removal50 Alpaca PCA-229 row.
- `nopersona_steer_results.json` rows other than No Robots: duplicates of steer_bench corrected rows, which have
  bootstrap CIs.
- `steer_v2_results/*.jsonl` arms whose ASR equals a steer_bench arm: same run, dropped at build.

**No corrected summary saved**
- `outputs_gemma_singledose/` (28 cells): only raw labels are saved. The valid version needs the looks_like_text guard
  applied to the responses, which is a re-scoring, so it is not done here.

**Not jailbreak ASR**
- `norm_share_controls/probe/`: behavioural regex-marker probes.

**Inputs, not results**
- `results/steering_cells.json`, `results/gemma_steering_cells.json`: inputs of the CI scripts.
- `results/gemma_gibberish_audit.json`.

**Not a comparable transfer metric**
- `why_dmu_no_drop.json` `xf3`: mean over only 3 families. Only `top10_pc_share_in_persona` is shown.

**Unclear or not audited**
- `olmo3_7b/detect/sweep_out/olmo_sweep_pjb.json`: OLMo layer × position grid under single-set training. The schema and
  selection rule were not audited.
- `olmo_probe.json` `within`: within-set CV, in-distribution only.
- Qwen `transfer/diag/{cross_family,cross_family_gaussian,within_family_tuned,diag_L48,template_variants}.json`,
  `transfer/preview*/detect_transfer.json`, `stage2_out/detection.json`, `l48_run/detect_pos.json`,
  `llama_control/*.json`: earlier or diagnostic runs whose protocols were not audited against the shown ones.
  cross_family_tuned feeds persona_four's Llama and Qwen cells, so it is covered there.
- trait_pc_sweep `region_tests_*.json`, `diagnose_pcs.json`, `lda_axes_llama.json`, `aggregate_lda.json`,
  `within_noise_check.json`: diagnostics such as in-family nonlinear tests and persona-ID accuracy of the LDA axes.
  These are not transfer detection results. The LDA detector AUCs themselves are shown, from summary_llama_lda_C*.json.
- The `published` blocks in trait_pc_sweep summaries: paper-table numbers with mixed per-cell protocols.
- `normality_in_persona.json` T1 / T4: trait effect sizes and a "persona intensity" AUC.
- `refdmu_tables.json` B: d′ and within-group statistics.
- `pjb_release_staging/derived/*/analysis/{pca.json,family_traits.json,single_trait_auc.csv}`: analysis tables, not
  detection or share results.
- `results/steering_alpha_curves.json`: per-pair min α with a schedule up to 0.7, not aligned with the others.
- The Q1 smoke output in the session scratchpad: development only, never in the default build.

**Snapshot caveat (Llama steering)**
- The `paper_llama` PersonaJB and single-trait cells are Dominic's runs. The local files do not state their snapshot, so
  they are tagged "snapshot not stated in files". The refusal and Δμ paper cells are tagged mirror, based on the
  plot_steer_bench.py comment.
