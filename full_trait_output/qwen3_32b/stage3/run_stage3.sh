#!/bin/bash
# Qwen3-32B stage 3 (job qwen-stage3): GCG, PAIR, PEZ, GPTFuzz with the test cases optimised against Gemma-4-31B,
# transferred to Qwen (generation + layer-32 readout + judge), readout check, PersonaJB detection with all five families,
# appended to SUMMARY.md / summary.json. Waits for stage 2 (STATUS last line "=== run end", timeout 6 h; exits cleanly if
# stage 2 did not produce HarmBench data). Writes into the stage-2 output dir (shared judge ledger, cap $9 cumulative).
# Resumable (outputs written .tmp -> rename, steps skip done outputs); heartbeat lock .lock3 against retry pods.
#     bash run_stage3.sh [--test]   (--test: 4 prompts per family into stage2_test, no waiting)
set -uo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/qwen3
MODEL=/mnt/dlabscratch1/lichen/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
TEST=""; [ "${1:-}" = "--test" ] && TEST="--test"
OUT=$W/stage2${TEST:+_test}; S=$(cd "$(dirname "$0")" && pwd); mkdir -p "$OUT/logs"; cd "$S"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0   # flashinfer JIT fails on B300 (sm_103a); attention backend set in qwen_gen.py
set -a; . /mnt/dlabscratch1/moskvore/persona_steer/openai.env; set +a
LAB="test cases optimised against Gemma-4-31B, transferred to Qwen"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS3"; }
# ---- wait for stage 2 ----
if [ -z "$TEST" ]; then
  for i in $(seq 1 360); do
    tail -n 1 "$OUT/STATUS" 2>/dev/null | grep -q "=== run end" && break
    [ $((i % 10)) -eq 1 ] && echo "$(date -u +%T) waiting for stage 2 ($i/360): $(tail -n 1 "$OUT/STATUS" 2>/dev/null)"
    sleep 60
  done
  if ! tail -n 1 "$OUT/STATUS" 2>/dev/null | grep -q "=== run end"; then log "stage 2 did not finish within 6 h; exiting"; exit 0; fi
  if [ ! -s "$OUT/classified_hb.jsonl" ] || [ ! -s "$OUT/acts_hb.pt" ]; then log "stage 2 ended without HarmBench data; exiting"; exit 0; fi
fi
# ---- lock (heartbeat; a preempted pod's lock goes stale after 10 min and is taken over) ----
LOCK=$OUT/.lock3
for i in $(seq 1 30); do
  if [ -e "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 600 ] && [ "$(cat "$LOCK")" != "$HOSTNAME" ]; then echo "lock held by $(cat "$LOCK"), waiting"; sleep 60; else break; fi
done
if [ -e "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -lt 600 ] && [ "$(cat "$LOCK")" != "$HOSTNAME" ]; then echo "lock still live after 30 min; refusing to double-write"; exit 1; fi
echo "$HOSTNAME" > "$LOCK"; ( while true; do touch "$LOCK"; sleep 60; done ) & HB=$!; trap 'kill $HB 2>/dev/null; rm -f "$LOCK"' EXIT
step() {  # step NAME DONEFILE CMD...
  local name=$1 done=$2; shift 2
  if [ -n "$done" ] && [ -s "$done" ]; then log "$name: skip (done)"; return 0; fi
  log "$name: start"; local t0=$SECONDS
  ( set -euo pipefail; "$@" ) > "$OUT/logs/$name.log" 2>&1; local rc=$?
  log "$name: $([ $rc -eq 0 ] && echo ok || echo "FAIL rc=$rc") ($((SECONDS - t0)) s)"; return 0
}
log "=== stage 3 start on $HOSTNAME ${TEST:+(TEST)} [$LAB]"
for st in gcg pair pez gptfuzz; do
  [ "${QWEN_BACKEND:-vllm}" = "vllm" ] && step "s3_1_${st}_generate_vllm" "$OUT/responses_$st.jsonl" "$PY" qwen_gen.py "$MODEL" qwen_inputs_transfer.jsonl "$st" "$OUT/responses_$st.jsonl" $TEST
  step "s3_1_${st}_generate_hf" "$OUT/responses_$st.jsonl" "$PY" qwen_gen_hf.py "$MODEL" qwen_inputs_transfer.jsonl "$st" "$OUT/responses_$st.jsonl" $TEST
  step "s3_2_${st}_readout" "$OUT/acts_$st.pt" "$PY" qwen_acts.py "$MODEL" qwen_inputs_transfer.jsonl "$st" "$OUT/acts_$st.pt" $TEST
  if [ -s "$OUT/responses_$st.jsonl" ]; then step "s3_3_${st}_judge" "$OUT/classified_$st.jsonl" "$PY" qwen_judge.py qwen_inputs_transfer.jsonl "$OUT/responses_$st.jsonl" "$OUT" 9
  else log "s3_3_${st}_judge: skip (no responses)"; fi
done
step "s3_4_readout_check" "$OUT/readout_check_transfer.json" "$PY" qwen_readout_check.py "$MODEL" qwen_inputs_transfer.jsonl "$OUT" gcg,pair,pez,gptfuzz readout_check_transfer.json
if [ -s "$OUT/classified_hb.jsonl" ] && [ -s "$OUT/acts_hb.pt" ]; then
  step "s3_5_detection_all5" "$OUT/all5_detection.json" "$PY" qwen_detect.py "$OUT" trait_matrix_layer32.npy all5_
else log "s3_5_detection_all5: skip (HarmBench not collected)"; fi
"$PY" qwen_summary3.py "$OUT" "$([ -n "$TEST" ] && echo "$OUT" || echo "$W")" > "$OUT/logs/s3_6_summary.log" 2>&1; log "s3_6_summary: rc=$?"
log "=== stage 3 end"
