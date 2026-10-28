#!/bin/bash
# Matched-control steering, persona-removed Alpaca arms (norm_share_controls/build_steer_nopersona.py). Runs inside a runai pod.
# Usage: bash run_steer_nopersona.sh <shard det|sel>. Cumulative alpha 0.1-0.5, HarmBench refused n=300 seed 0,
# official meta-llama snapshot (system-header chat template), same harness as run_controls_cumulative.sh.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/steer_nopersona_layer16.pt
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache HF_DATASETS_CACHE=/mnt/dlabscratch1/moskvore/hf_cache/datasets TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a
case "$1" in
  det) NAMES=alpacajb_d2_nopersona,alpacajb_d2_nonr ;;
  sel) NAMES=sel_pca_1_nopersona,sel_acts_1_nopersona ;;
  *) echo "shard?"; exit 1 ;;
esac
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL'); assert 'Cutting Knowledge Date' in t.chat_template; print('official template ok')"
echo "=== $(date -u +%FT%TZ) | shard $1 | $NAMES ==="
"$PY" jailbreaking_tools/steer_fixed_direction.py --benchmark harmbench --directions "$BUNDLE" --names "$NAMES" \
    --model "$MODEL" --classified_path "$CLS" --output_dir "outputs_controls_v2/nopersona_$1" \
    --alpha_schedule 0.1 0.2 0.3 0.4 0.5 --n_refused 300 --pool_seed 0 --batch_size 32 --judge_workers 8
echo ALL_DONE
