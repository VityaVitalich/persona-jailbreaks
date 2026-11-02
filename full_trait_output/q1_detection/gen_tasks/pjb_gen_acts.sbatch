#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# General-task activations (SST-2, AG News, cities true/false) for the erasure test's general probes, with the same
# per-model recipes as pjb_q1_ctrl.sbatch (Llama/Gemma acts_multi.py, OLMo olmo3_7b/scripts/acts.py). Inputs gen_inputs.jsonl.
#   sbatch -A <acct> -p debug --environment=/users/vvmoskvoretskii/MR-Eval/container/serving.toml pjb_gen_acts.sbatch
set -uo pipefail
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb; Q=$P/q1_detection; H=$P/hb_select; I=$Q/gen_inputs.jsonl; S=sst2,agnews,cities
export HF_HOME=$U/hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
snap() { python3 -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"; }
mkdir -p $Q/gen_acts_llama $Q/gen_acts_olmo $Q/gen_acts
( CUDA_VISIBLE_DEVICES=0 python3 -u $H/acts_multi.py /capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-8B-Instruct $I $Q/gen_acts_llama $S 4,8,12,16,20,24,28,31 0 1 \
  && python3 -u $H/acts_multi.py --merge $Q/gen_acts_llama $S 4,8,12,16,20,24,28,31 1 ) > $Q/logs/gen_llama.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$U/pyoverlay/gemma4_tf58 python3 -u $H/acts_multi.py google/gemma-4-31B-it $I $Q/gen_acts $S 15,20,25,30,35,40,45 0 1 --gemma \
  && PYTHONPATH=$U/pyoverlay/gemma4_tf58 python3 -u $H/acts_multi.py --merge $Q/gen_acts $S 15,20,25,30,35,40,45 1 ) > $Q/logs/gen_gemma.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2 python3 -u $P/olmo3_7b/scripts/acts.py $(snap allenai/Olmo-3-7B-Instruct) $I $Q/gen_acts_olmo $S && python3 -u - <<'PY'
import torch, numpy as np, json
D = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/gen_acts_olmo"
for st in ("sst2", "agnews", "cities"):
    a = torch.load(f"{D}/acts_{st}.pt")
    for j, L in enumerate(a["layers"]): np.save(f"{D}/{st}_L{L}.npy", a["X"][:, j].numpy())
    json.dump({"pair_ids": a["pair_ids"], "tail_tokens": a["tail_tokens"], "layers": a["layers"]}, open(f"{D}/{st}_meta.json", "w")); print("converted", st, tuple(a["X"].shape))
PY
) > $Q/logs/gen_olmo.log 2>&1 &
wait
for m in llama olmo gemma; do echo "== $m"; grep -h "rows in\|merged\|converted\|: [0-9]* in\|Error\|Traceback" $Q/logs/gen_$m.log | cut -c1-200; done
ls $Q/gen_acts_llama/*.npy $Q/gen_acts_olmo/*.npy $Q/gen_acts/*.npy | wc -l   # expect 24 + 24 + 21 = 69
