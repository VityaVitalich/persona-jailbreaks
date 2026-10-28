#!/bin/bash
set -euo pipefail
W=/mnt/dlabscratch1/moskvore/persona_steer/wiki
MODEL=/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
PY=/mnt/dlabscratch1/moskvore/conda-envs/mr-eval/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$W"; "$PY" wiki_acts.py "$MODEL" norobots_prompts.jsonl norobots_acts.pt; echo ALL_DONE
