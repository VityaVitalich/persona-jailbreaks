#!/bin/bash
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/qwen3
MODEL=/mnt/dlabscratch1/lichen/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
OUT=$W/diag; mkdir -p $OUT; export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s $OUT/diag_acts.pt ] && { echo "already done"; exit 0; }
$PY $W/diag_run/qwen_diag.py $MODEL $W/stage2_scripts/qwen_inputs.jsonl $W/stage2/responses_hb.jsonl $OUT ${1:-} > $OUT/diag.log 2>&1
echo "exit $?" >> $OUT/diag.log
