#!/bin/bash
# Qwen3-32B on its own GCG/PAIR/PEZ/GPTFuzz test cases: greedy responses (vLLM, as stage 2), layer-32/48 readout at the
# last 9 template positions, gpt-4.1-mini judge. Resumable per step.   bash run_transfer.sh SET[,SET..]
set -uo pipefail
SETS=$1
W=/mnt/dlabscratch1/moskvore/persona_steer/qwen3; S=$W/transfer_scripts; OUT=$W/transfer; mkdir -p $OUT/logs
MODEL=/mnt/dlabscratch1/lichen/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
set -a; . $W/../openai.env; set +a
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/STATUS; }
INP=$OUT/qwen_inputs_own.jsonl
$PY $S/build_inputs_own.py $W/attacks $W/harmbench/data/behavior_datasets/harmbench_behaviors_text_all.csv $INP.tmp.$HOSTNAME > $OUT/logs/build_$SETS.log 2>&1 && mv $INP.tmp.$HOSTNAME $INP; log "build: $(tr '\n' ' ' < $OUT/logs/build_$SETS.log)"
for st in ${SETS//,/ }; do
  [ -s $OUT/responses_$st.jsonl ] || { $PY $S/qwen_gen.py $MODEL $INP $st $OUT/responses_$st.jsonl > $OUT/logs/gen_$st.log 2>&1; log "gen $st rc=$?"; }
  [ -s $OUT/acts_pos_$st.pt ] || { $PY $S/qwen_acts_pos.py $MODEL $INP $OUT $st > $OUT/logs/acts_$st.log 2>&1; log "acts $st rc=$?"; }
  [ -s $OUT/classified_$st.jsonl ] || { $PY $S/qwen_judge.py $INP $OUT/responses_$st.jsonl $OUT 9 > $OUT/logs/judge_$st.log 2>&1; log "judge $st rc=$? $(tail -2 $OUT/logs/judge_$st.log | tr '\n' ' ')"; }
done
log "done $SETS"
