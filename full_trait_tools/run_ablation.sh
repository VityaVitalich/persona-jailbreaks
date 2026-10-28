#!/bin/bash
# Directional ablation (Arditi-style: every layer, every position) of the six
# fig:attack directions on the INITIALLY-COMPLIED pools -- HarmBench 1245
# human_jailbreak pairs, WildJB 1025 adversarial prompts -- plus the no-hook
# baseline arm and six random-direction controls, at two placements (all layers, and
# layer 16 only). One greedy generation per prompt per arm, scored by the judge that
# produced the pool labels.
# Runs inside the runai pod; mirrors run_controls_cumulative.sh. Writes a new
# outputs_ablation/ tree, the canonical outputs/ tree is untouched.
#
#   bash run_ablation.sh          full run (harmbench, then wildjb)
#   bash run_ablation.sh smoke    3 prompts per benchmark -> outputs_ablation_smoke/
set -euo pipefail

W=/mnt/dlabscratch1/moskvore/persona_steer
cd "$W"

# The weights the pool labels were collected with, from Dominic's own HF cache.
# Byte-identical to the VityaVitalich mirror (same safetensors blob hashes) but its
# tokenizer ships the official 4614-char chat template with the "Cutting Knowledge
# Date / Today Date" system header; the mirror's 348-char template omits it and the
# model then refuses the prompts it complied with.
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
CLS=/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output/harmbench_activations/classified_responses.jsonl
WJB_CLS=inputs/wjb_classified_responses.jsonl      # local copy of the detection run's labels (975 refused / 1025 complied)
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
BUNDLE=inputs/ablation_directions_layer16.pt

export HF_HOME=/mnt/dlabscratch1/moskvore/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
set -a; . "$W/openai.env"; set +a          # judge key, never echoed

# Three placements. `none` has no hook, so it is placement-independent and runs once.
#   <dir>      ablate at EVERY layer (Arditi recipe)
#   <dir>@16   ablate at layer 16 block output only = where fig:attack steering ADDED
# Layer 16 is the Llama layer throughout this paper; layer 30 belongs to the Gemma
# track and is deliberately not run here.
DIRS=persona,refusal,dmu,utilitarian,closure_seeking,edgy
CTRL=random_full_s0,random_full_s1,random_full_s2,random_persona_s0,random_persona_s1,random_persona_s2
suffix() { echo "$1" | tr ',' '\n' | sed "s/\$/$2/" | paste -sd, -; }
NAMES="none,$DIRS,$CTRL,$(suffix "$DIRS" "@16"),$(suffix "$CTRL" "@16")"
OUT=outputs_ablation
EXTRA=()
if [ "${1:-}" = "smoke" ]; then
    OUT=outputs_ablation_smoke
    EXTRA=(--test)
fi

"$PY" -c "import torch; c=torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', c, torch.cuda.get_device_name(0) if c else '', '| capability', torch.cuda.get_device_capability(0) if c else '')"

for BENCH in harmbench wildjb; do
    echo "=== $(date -u +%FT%TZ) | ablation | $BENCH | out=$OUT | arms: $NAMES ==="
    "$PY" jailbreaking_tools/ablate_fixed_direction.py \
        --benchmark "$BENCH" \
        --directions "$BUNDLE" \
        --names "$NAMES" \
        --model "$MODEL" \
        --classified_path "$CLS" \
        --wjb_classified_path "$WJB_CLS" \
        --output_dir "$OUT" \
        --batch_size 48 --judge_workers 12 \
        ${EXTRA[@]+"${EXTRA[@]}"}
done
echo "=== $(date -u +%FT%TZ) | ablation done ==="
