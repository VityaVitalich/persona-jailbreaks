#!/bin/bash
set -uo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/qwen3
MODEL=/mnt/dlabscratch1/lichen/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
OUT=$W/l48; mkdir -p $OUT; export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY $W/l48_run/qwen_acts_pos.py $MODEL $W/stage2_scripts/qwen_inputs.jsonl $OUT > $OUT/l48.log 2>&1
echo "exit $?" >> $OUT/l48.log
