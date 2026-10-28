#!/bin/bash
# Qwen3-32B detection data: generation (vLLM) -> layer-32 readout (HF) -> gpt-4.1-mini judge.  bash run_qwen.sh [--test]
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/qwen3
MODEL=/mnt/dlabscratch1/lichen/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
set -a; . /mnt/dlabscratch1/moskvore/persona_steer/openai.env; set +a
TAG=${1:-full}; OUT=$W/out_${TAG#--}; mkdir -p "$OUT"; cd "$W"
"$PY" -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
[ -s "$OUT/responses.jsonl" ] || "$PY" qwen_gen.py "$MODEL" qwen_inputs.jsonl "$OUT/responses.jsonl" $@ 2>&1 | grep -v "it/s\]" | tail -n 40
[ -s "$OUT/acts_hb.pt" ] || "$PY" qwen_acts.py "$MODEL" qwen_inputs.jsonl "$OUT" $@
[ -s "$OUT/classified_hb.jsonl" ] || "$PY" qwen_judge.py qwen_inputs.jsonl "$OUT/responses.jsonl" "$OUT"
[ "$TAG" = "--test" ] || [ -s "$OUT/extra_acts.pt" ] || "$PY" qwen_extra_acts.py "$MODEL" gemma_acts_inputs.jsonl "$OUT/extra_acts.pt"
echo ALL_DONE
