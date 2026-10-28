#!/bin/bash
# Matched-control steering, Wikipedia / No Robots arms (norm_share_controls/build_steer_wiki.py). Runs inside a runai pod.
# Usage: bash run_steer_controls_v2.sh <shard 1|2|3>. Cumulative alpha 0.1-0.5, HarmBench refused n=300 seed 0,
# official meta-llama snapshot (system-header chat template), same harness as run_controls_cumulative.sh.
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/steer_extra_layer16.pt
export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache HF_DATASETS_CACHE=/mnt/dlabscratch1/moskvore/hf_cache/datasets TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a
case "$1" in
  wiki) NAMES=wikijb_d0,wikijb_d1,wikijb_d2,sel_wiki_1,sel_wiki_2,sel_wiki_3 ;;
  nr) NAMES=nrjb_d0,nrjb_d1,nrjb_d2,sel_nr_1,sel_nr_2,sel_nr_3 ;;
  *) echo "shard?"; exit 1 ;;
esac
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL'); assert 'Cutting Knowledge Date' in t.chat_template; print('official template ok')"
echo "=== $(date -u +%FT%TZ) | shard $1 | $NAMES ==="
"$PY" jailbreaking_tools/steer_fixed_direction.py --benchmark harmbench --directions "$BUNDLE" --names "$NAMES" \
    --model "$MODEL" --classified_path "$CLS" --output_dir "outputs_controls_v2/extra_$1" \
    --alpha_schedule 0.1 0.2 0.3 0.4 0.5 --n_refused 300 --pool_seed 0 --batch_size 32 --judge_workers 8
echo ALL_DONE
