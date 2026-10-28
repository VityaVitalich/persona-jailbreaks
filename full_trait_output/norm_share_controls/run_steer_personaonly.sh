#!/bin/bash
# Persona-only steering arms (persona component of each direction), HarmBench 300-pool only.
# Bundle: norm_share_controls/build_steer_personaonly.py. Runs inside a runai pod.
# Usage: bash run_steer_personaonly.sh <shard refdmu|alpaca>. Cumulative alpha 0.1-0.5, official meta-llama snapshot.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/steer_personaonly_layer16.pt
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache HF_DATASETS_CACHE=/mnt/dlabscratch1/moskvore/hf_cache/datasets TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a
BENCH=harmbench; EXTRA=(--n_refused 300 --pool_seed 0)
case "$1" in
  refdmu) NAMES=refusal_personaonly,dmu_personaonly ;;
  alpaca) NAMES=alpacajb_d2_personaonly,sel_pca_1_personaonly ;;
  *) echo "shard?"; exit 1 ;;
esac
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL'); assert 'Cutting Knowledge Date' in t.chat_template; print('official template ok')"
echo "=== $(date -u +%FT%TZ) | shard $1 | $BENCH | $NAMES ==="
"$PY" jailbreaking_tools/steer_fixed_direction.py --benchmark "$BENCH" --directions "$BUNDLE" --names "$NAMES" \
    --model "$MODEL" --classified_path "$CLS" --output_dir "outputs_controls_v2/personaonly_$1" \
    --alpha_schedule 0.1 0.2 0.3 0.4 0.5 --batch_size 32 --judge_workers 8 ${EXTRA[@]+"${EXTRA[@]}"}
echo ALL_DONE
