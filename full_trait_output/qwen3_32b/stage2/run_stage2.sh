#!/bin/bash
# Qwen3-32B persona-JB stage 2, end to end on one node (job qwen-stage2). Resumable: every step skips outputs that
# exist (all outputs are written to .tmp then renamed). A lock with a heartbeat keeps a second pod (runai auto-retry)
# from writing at the same time. Each step runs in its own subshell with set -euo pipefail; a failed step is logged to
# STATUS and the next independent step still runs.     bash run_stage2.sh [--test]
set -uo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/qwen3
MODEL=/mnt/dlabscratch1/lichen/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
TEST=""; [ "${1:-}" = "--test" ] && TEST="--test"
OUT=$W/stage2${TEST:+_test}; S=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT/logs"; cd "$S"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # flashinfer JIT fails on B300 (sm_103a); attention backend set in qwen_gen.py
set -a; . /mnt/dlabscratch1/moskvore/persona_steer/openai.env; set +a
# ---- lock (heartbeat file touched every 60 s; a lock older than 10 min is stale) ----
LOCK=$OUT/.lock
# a live holder keeps the heartbeat fresh; a preempted pod's lock goes stale after 10 min and is taken over
for i in $(seq 1 30); do
  if [ -e "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 600 ] && [ "$(cat "$LOCK")" != "$HOSTNAME" ]; then
    echo "lock held by $(cat "$LOCK"), waiting ($i/30)"; sleep 60
  else break; fi
done
if [ -e "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 600 ] && [ "$(cat "$LOCK")" != "$HOSTNAME" ]; then
  echo "lock still live after 30 min; refusing to double-write"; exit 1; fi
echo "$HOSTNAME" > "$LOCK"; ( while true; do touch "$LOCK"; sleep 60; done ) & HB=$!; trap 'kill $HB 2>/dev/null; rm -f "$LOCK"' EXIT
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
step() {  # step NAME DONEFILE CMD...   (skips if DONEFILE exists)
  local name=$1 done=$2; shift 2
  if [ -n "$done" ] && [ -s "$done" ]; then log "$name: skip (done)"; return 0; fi
  log "$name: start"; local t0=$SECONDS
  ( set -euo pipefail; "$@" ) > "$OUT/logs/$name.log" 2>&1; local rc=$?
  log "$name: $([ $rc -eq 0 ] && echo ok || echo "FAIL rc=$rc") ($((SECONDS - t0)) s)"; return 0
}
log "=== run start on $HOSTNAME ${TEST:+(TEST)} $("$PY" -c 'import torch;print(torch.cuda.get_device_name(0))' 2>&1)"
for st in hb wjb pap; do
  [ "${QWEN_BACKEND:-vllm}" = "vllm" ] && step "1_${st}_generate_vllm" "$OUT/responses_$st.jsonl" "$PY" qwen_gen.py "$MODEL" qwen_inputs.jsonl "$st" "$OUT/responses_$st.jsonl" $TEST
  step "1_${st}_generate_hf" "$OUT/responses_$st.jsonl" "$PY" qwen_gen_hf.py "$MODEL" qwen_inputs.jsonl "$st" "$OUT/responses_$st.jsonl" $TEST   # fallback; skipped if vLLM succeeded
  step "2_${st}_readout"  "$OUT/acts_$st.pt"         "$PY" qwen_acts.py "$MODEL" qwen_inputs.jsonl "$st" "$OUT/acts_$st.pt" $TEST
  if [ -s "$OUT/responses_$st.jsonl" ]; then
    step "3_${st}_judge" "$OUT/classified_$st.jsonl" "$PY" qwen_judge.py qwen_inputs.jsonl "$OUT/responses_$st.jsonl" "$OUT" 9
  else log "3_${st}_judge: skip (no responses)"; fi
done
step "4_readout_check" "$OUT/readout_check.json" "$PY" qwen_readout_check.py "$MODEL" qwen_inputs.jsonl "$OUT"
if [ -s "$OUT/classified_hb.jsonl" ] && [ -s "$OUT/acts_hb.pt" ]; then
  step "5_detection" "$OUT/detection.json" "$PY" qwen_detect.py "$OUT" trait_matrix_layer32.npy
else log "5_detection: skip (HarmBench not collected)"; fi
[ -n "$TEST" ] || step "7_control_dictionary_acts" "$OUT/extra_acts.pt" "$PY" qwen_extra_acts.py "$MODEL" gemma_acts_inputs.jsonl "$OUT/extra_acts.pt"
"$PY" qwen_summary.py "$OUT" "$([ -n "$TEST" ] && echo "$OUT" || echo "$W")" > "$OUT/logs/6_summary.log" 2>&1; log "6_summary: rc=$?"
log "=== run end"
