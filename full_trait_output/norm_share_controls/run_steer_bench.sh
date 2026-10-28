#!/bin/bash
# Persona-removal steering on JBB / WildJB (+ same-snapshot persona / random references on HarmBench).
# Bundle: norm_share_controls/build_steer_bench_bundle.py. Runs inside a runai pod.
# Usage: bash run_steer_bench.sh <shard jbb|wjb1|wjb2|wjb3|hb>. Cumulative alpha 0.1-0.5, official meta-llama snapshot.
# WildJB: first 600 of the published 2000-behaviour sample (seed 42); ASR is later restricted to the refused subset.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/steer_bench_layer16.pt
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache HF_DATASETS_CACHE=/mnt/dlabscratch1/moskvore/hf_cache/datasets TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a
ALL=alpacajb_d2,alpacajb_d2_nopersona,alpacajb_d2_nonr,sel_pca_1,sel_pca_1_nopersona,sel_acts_1,sel_acts_1_nopersona,persona,random_full_s0
EXTRA=()
case "$1" in
  jbb) BENCH=jbb; NAMES=$ALL ;;
  wjb1) BENCH=wildjb; NAMES=alpacajb_d2,alpacajb_d2_nopersona,alpacajb_d2_nonr; EXTRA=(--limit 600) ;;
  wjb2) BENCH=wildjb; NAMES=sel_pca_1,sel_pca_1_nopersona,sel_acts_1; EXTRA=(--limit 600) ;;
  wjb3) BENCH=wildjb; NAMES=sel_acts_1_nopersona,persona,random_full_s0; EXTRA=(--limit 600) ;;
  hb) BENCH=harmbench; NAMES=persona,random_full_s0; EXTRA=(--n_refused 300 --pool_seed 0) ;;
  *) echo "shard?"; exit 1 ;;
esac
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL'); assert 'Cutting Knowledge Date' in t.chat_template; print('official template ok')"
echo "=== $(date -u +%FT%TZ) | shard $1 | $BENCH | $NAMES ==="
"$PY" jailbreaking_tools/steer_fixed_direction.py --benchmark "$BENCH" --directions "$BUNDLE" --names "$NAMES" \
    --model "$MODEL" --classified_path "$CLS" --output_dir "outputs_controls_v2/bench_$1" \
    --alpha_schedule 0.1 0.2 0.3 0.4 0.5 --batch_size 32 --judge_workers 8 ${EXTRA[@]+"${EXTRA[@]}"}
echo ALL_DONE
