#!/usr/bin/env bash
# Fill the GCG column of the Gemma detection table.
#
# Runs only after run_gcg_gemma_finish.sh has produced
# full_trait_output/gcg_activations_gemma_hb/{classified_responses.jsonl,activations.pt}.
#
# Rows this fills: Trait projections, Raw activations, PCA(3), JBShield, GradSafe,
# Llama Guard, Verbalized (no CoT / w CoT), Perplexity.
#
# NOT filled here: Mahalanobis contrastive and Jailbreak-Leaves-a-Trace (SVM/RF).
# Those read full_trait_output/rtv_activations_gemma/gcg_{rows.jsonl,activations.pt},
# a separate collection the finish chain does not do (Llama has it, Gemma does not).
# Collect that first if those two cells are wanted; the two scripts now skip GCG
# gracefully when the file is absent rather than silently omitting the column.
set -uo pipefail
PROJ=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B
cd "$PROJ"
set -a; source /dlabscratch1/bazina/assistant-axis-llama3.1-8B/.env; set +a
export HF_HOME=/mnt/dlabscratch1/bazina/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CHUNK_Q=1024
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

G=full_trait_output/gcg_activations_gemma_hb
if [ ! -s "$G/classified_responses.jsonl" ]; then
  echo "ABORT: $G/classified_responses.jsonl missing or empty — run the finish chain first."
  exit 1
fi
echo "GCG rows available: $(wc -l < $G/classified_responses.jsonl)"

# The file-bridge pod gets preempted, which kills this script mid-flight. Make every
# stage skippable so a relaunch resumes instead of recomputing (or worse, overwriting
# a complete result with a partial one). $2 is a file that must exist and be non-empty
# for the stage to count as done.
run () {
  local label="$1" marker="$2"; shift 2
  echo; echo "=========== $label ==========="
  if [ -s "$marker" ]; then echo "[skip] already complete: $marker"; return 0; fi
  "$@" && echo "[ok]" || echo "[FAILED: $label]"
}

TV=gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter
# Restrict to the L2 logistic grid. The full 62-config grid costs ~868 s/seed (~12 h per
# feature type) because of the saga/ElasticNet and LinearSVC specs, which never won a single
# seed in the published Gemma run — every seed there selected logreg_l2_C1.0_bal or _raw.
# Per-spec AUCs are unaffected by which other specs are in the grid, and the table reads
# logreg_l2_C1.0_bal, so this changes cost only.
L2GRID=$(printf "logreg_l2_C%s_%s," 0.003 bal 0.003 raw 0.01 bal 0.01 raw 0.03 bal 0.03 raw \
  0.1 bal 0.1 raw 0.3 bal 0.3 raw 1.0 bal 1.0 raw 3.0 bal 3.0 raw 10.0 bal 10.0 raw)
L2GRID=${L2GRID%,}
COMMON_TRAIT="--layer 30 --n_seeds 50 --trait_row_index 0 --trait_vectors_dir $TV
  --only_models $L2GRID
  --human_classified_path  full_trait_output/harmbench_activations_gemma/classified_responses.jsonl
  --human_activations_path full_trait_output/harmbench_activations_gemma/activations.pt
  --pair_classified_path   full_trait_output/pair_activations_gemma_hb/classified_responses.jsonl
  --pair_activations_path  full_trait_output/pair_activations_gemma_hb/activations.pt
  --pap_classified_path    full_trait_output/pap_activations_gemma_hb/classified_responses.jsonl
  --pap_activations_path   full_trait_output/pap_activations_gemma_hb/activations.pt
  --gptfuzz_classified_path full_trait_output/gptfuzz_activations_gemma_hb/classified_responses.jsonl
  --gptfuzz_activations_path full_trait_output/gptfuzz_activations_gemma_hb/activations.pt
  --pez_classified_path    full_trait_output/pez_activations_gemma_hb/classified_responses.jsonl
  --pez_activations_path   full_trait_output/pez_activations_gemma_hb/activations.pt
  --gcg_classified_path    $G/classified_responses.jsonl
  --gcg_activations_path   $G/activations.pt
  --augment_classified_path  full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl
  --augment_activations_path full_trait_output/wildjailbreak_activations_gemma/activations.pt"

# --- Trait projections / Raw activations / PCA(3) (CPU) ---
for FT in trait raw pca; do
  case $FT in
    trait) OUT=all_traits_sweep_v2_gemma_gcg_all_traits; EXTRA="--feature_type trait" ;;
    raw)   OUT=all_traits_sweep_v2_gemma_gcg_raw;        EXTRA="--feature_type raw" ;;
    pca)   OUT=all_traits_sweep_v2_gemma_gcg_pca3;       EXTRA="--feature_type pca --pca_n_components 3" ;;
  esac
  run "all_traits ($FT)" "full_trait_output/$OUT/results.json" .venv/bin/python full_trait_tools/run_all_traits_sweep_v2.py \
      $COMMON_TRAIT $EXTRA --output_dir full_trait_output/$OUT
done

# --augment_wjb matches the published Gemma rows, whose output dirs carry the _hbwjb
# suffix. Without it every family shifts and the new GCG cell would not be
# protocol-consistent with the other five cells in its row.
# --- Baselines that read DATASETS_GEMMA (GCG entry added, exp -> gemma4_31b) ---
run "JBShield"      "full_trait_output/baselines_jbshield_fjd_gemma_gcg/jbshield_fjd_results.json" .venv_gemma/bin/python full_trait_tools/run_baselines_jbshield_fjd.py --gemma --augment_wjb \
    --output_dir full_trait_output/baselines_jbshield_fjd_gemma_gcg
# GradSafe has no family filter, but it checkpoints per prompt. Pointing it at the
# EXISTING output dir means the 996 already-scored rows replay from cache in seconds
# and only GCG is computed fresh; it also reuses the 46GB calibration cache there,
# skipping ~3h of recalibration. It rewrites gradsafe_all_attacks.json with GCG added.
run "GradSafe"      "" .venv_gemma/bin/python full_trait_tools/run_baselines_gradsafe.py --gemma \
    --calib_responses full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl \
    --balance_positives \
    --output_dir full_trait_output/baselines_gradsafe_gemma_bal
run "Llama Guard"   "full_trait_output/baselines_all_attacks_gemma_gcg/llama_guard_all_attacks.json" .venv_gemma/bin/python full_trait_tools/run_baselines_llama_guard.py --gemma \
    --output_dir full_trait_output/baselines_all_attacks_gemma_gcg
# Only GCG is new; the other five families are already in the published table and this
# baseline costs ~23 s/prompt because it generates a meta-response for each one.
run "Verbalized"    "full_trait_output/baselines_self_exam_gemma_gcg/self_exam_all_attacks.json" .venv_gemma/bin/python full_trait_tools/run_baselines_self_exam.py --gemma \
    --only_families GCG \
    --output_dir full_trait_output/baselines_self_exam_gemma_gcg
run "Perplexity"    "full_trait_output/baselines_harmbench_ppl_smoothllm_gemma_gcg/harmbench_ppl_smoothllm_results.json" .venv_gemma/bin/python full_trait_tools/run_baselines_harmbench_ppl_smoothllm.py \
    --gemma --baselines perplexity --recompute_ppl --n_seeds 50 \
    --output_dir full_trait_output/baselines_harmbench_ppl_smoothllm_gemma_gcg

# --- RTV-based rows: run only if the GCG RTV collection exists ---
if [ -s full_trait_output/rtv_activations_gemma/gcg_activations.pt ]; then
  run "Mahalanobis" "full_trait_output/baselines_rtv_mahalanobis_gemma_gcg/rtv_mahalanobis_results.json" .venv/bin/python full_trait_tools/run_baseline_rtv_mahalanobis.py --gemma \
      --output_dir full_trait_output/baselines_rtv_mahalanobis_gemma_gcg
  run "JLT"         "full_trait_output/baselines_jlt_hidden_tensor_gemma_gcg/jlt_hidden_tensor_results.json" .venv/bin/python full_trait_tools/run_baseline_jlt_hidden_tensor.py --gemma --augment_wjb \
      --output_dir full_trait_output/baselines_jlt_hidden_tensor_gemma_gcg
else
  echo; echo "SKIP Mahalanobis + JLT: full_trait_output/rtv_activations_gemma/gcg_activations.pt absent."
  echo "     Collect GCG RTV activations first if those two cells are needed."
fi
echo; echo "$(date -Is) GCG detector sweep done"
