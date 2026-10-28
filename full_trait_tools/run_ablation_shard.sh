#!/bin/bash
# One SHARD of the ablation run: a subset of arms on one benchmark, so the 25 arms
# can be spread over several B300 GPUs instead of walking them serially.
#
# Arms are fully independent -- each writes its own outputs_ablation/<bench>/<arm>/
# directory and nothing is shared except the per-benchmark _run_config.json guard,
# which every shard writes identically. Resume is per-arm and append-only, so a
# shard picking up a half-finished arm continues it rather than redoing it.
#
#   bash run_ablation_shard.sh <benchmark> <comma-separated-arms>
set -euo pipefail

W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"

BENCH="${1:?usage: run_ablation_shard.sh <benchmark> <arms>}"
NAMES="${2:?usage: run_ablation_shard.sh <benchmark> <arms>}"

MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
WJB_CLS=inputs/wjb_classified_responses.jsonl
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/ablation_directions_layer16.pt

export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a          # judge key, never echoed

echo "=== $(date -u +%FT%TZ) | shard | $BENCH | arms: $NAMES ==="
exec "$PY" jailbreaking_tools/ablate_fixed_direction.py \
    --benchmark "$BENCH" \
    --directions "$BUNDLE" \
    --names "$NAMES" \
    --model "$MODEL" \
    --classified_path "$CLS" \
    --wjb_classified_path "$WJB_CLS" \
    --output_dir outputs_ablation \
    --batch_size 48 --judge_workers 16
