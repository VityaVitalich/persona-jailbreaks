#!/bin/bash
# Fan out the Gemma-4-31B steering reproduction over the B300 pool: one runai job per
# (benchmark, arm) for HarmBench and WildJB, one job for all arms on JBB (100 prompts).
#   bash submit_gemma_steer_fanout.sh [bs_hb=48] [bs_wjb=96] [judge_workers=12] [benchmarks: hb,wjb,jbb]
set -euo pipefail
BS_HB="${1:-48}"; BS_WJB="${2:-96}"; JW="${3:-12}"; WHICH="${4:-hb,wjb,jbb}"
RN=258.088   # layer-30 residual norm, calibrated identically by both B300 probe jobs (Dominic's recipe)
export SUPPRESS_DEPRECATION_MESSAGE=true
W=/mnt/dlabscratch1/moskvore/persona_steer
IMG=ghcr.io/jkminder/dlab-runai-images/pytorch:master
ARMS=(w_persona__full refusal_pos1_neg__full dmu__full
      trait__romantic trait__stoic trait__dispassionate trait__emotional trait__literal
      trait__flirty trait__patient trait__dominant trait__reserved trait__confident
      w_raw__full random_full_s0__full random_persona_s0__full)
short() { echo "$1" | sed -e 's/__full//' -e 's/trait__//' -e 's/_/-/g' | cut -c1-20; }
EXISTING=$(runai-rcp-prod list jobs -p dlab-moskvore 2>/dev/null | awk '$2 ~ /Running|Pending|Succeeded|Unknown|ContainerCreating/ {print $1}')
submit() {  # name bench arms extra...   (skips jobs already running/queued/finished)
  local name=$1 bench=$2 arms=$3; shift 3
  if echo "$EXISTING" | grep -qx "$name"; then echo "  [$name] exists (running/queued/done) - skipped"; return; fi
  runai-rcp-prod submit "$name" -p dlab-moskvore -i "$IMG" -g 1 --pvc dlab-scratch:/mnt --node-pools b300 \
    -- /bin/bash "$W/run_gemma_steer.sh" "$bench" "$arms" --residual_norm "$RN" --judge_workers "$JW" --token_budget "${TOKEN_BUDGET:-180000}" "$@" \
    2>&1 | grep -E "submitted|rror|already exists" | sed "s/^/  [$name] /"
}
if [[ "$WHICH" == *hb* ]]; then
  for a in "${ARMS[@]}"; do submit "gs-hb-$(short "$a")" harmbench "$a" --batch_size "$BS_HB"; done
fi
if [[ "$WHICH" == *wjb* ]]; then
  for a in "${ARMS[@]}"; do submit "gs-wjb-$(short "$a")" wildjb "$a" --batch_size "$BS_WJB"; done
fi
if [[ "$WHICH" == *jbb* ]]; then
  submit "gs-jbb-all" jbb "none,$(IFS=,; echo "${ARMS[*]}")" --batch_size 100 --posthoc_jbb_judge
fi
runai-rcp-prod list jobs -p dlab-moskvore 2>/dev/null | grep -E "NAME|gs-" | awk '{print $1, $2, $10, $11}' | column -t
