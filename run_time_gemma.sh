#!/usr/bin/env bash
# Time Gemma-4-31B detection baselines (GPU) + detection heads (node CPU),
# for the Gemma cost-AUC figure. GradSafe latency comes from the finished
# gradsafe_gemma checkpoint JSONLs, not from this job.
set -uo pipefail

BASE=""
for c in /dlabscratch1 /mnt/dlabscratch1 /mnt/dlab/scratch/dlabscratch1; do
  if [ -d "$c/bazina" ]; then BASE=$c; break; fi
done
PROJ=$BASE/bazina/assistant-axis-llama3.1-8B
export OUT_DIR=$BASE/moskvore/gemma_timing/out
mkdir -p "$OUT_DIR"
export HF_HOME=$BASE/bazina/.cache/huggingface
export HF_HUB_OFFLINE=0
export PYTHONPATH="$PROJ/full_trait_tools:$BASE/moskvore/ci_rescore"

cd "$PROJ"
echo "=== GPU baselines (Gemma-4-31B, Llama Guard) ===" | tee "$OUT_DIR/run.log"
.venv_gemma/bin/python "$BASE/moskvore/gemma_timing/time_baselines_gpu_gemma.py" \
    --n 48 --out "$OUT_DIR/gpu_timings.json" 2>&1 | tee -a "$OUT_DIR/run.log"

echo "=== CPU heads (same node) ===" | tee -a "$OUT_DIR/run.log"
.venv/bin/python "$BASE/moskvore/gemma_timing/time_heads_gemma_cluster.py" \
    2>&1 | tee -a "$OUT_DIR/run.log"

echo "DONE_ALL" | tee -a "$OUT_DIR/run.log"
