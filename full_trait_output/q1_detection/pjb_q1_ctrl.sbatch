#!/bin/bash
#SBATCH --account=a0265
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/logs/%x-%j.out
# Control-prompt activations (alpaca, wiki, harmful) for the q1 sweep, one GPU per model, each with the recipe that made
# that model's detection activations: Llama/Gemma acts_multi.py (hb_select), OLMo olmo3_7b/scripts/acts.py, Qwen
# qwen3_32b/sweep/qwen_acts_sweep.py (harmbench image, as its sweep). Gemma alpaca/wiki exist already -> harmful only.
#   sbatch -p debug --environment=/users/vvmoskvoretskii/MR-Eval/container/serving.toml pjb_q1_ctrl.sbatch
set -uo pipefail
U=/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii; P=$U/persona_jb; Q=$P/q1_detection; H=$P/hb_select; I=$Q/ctrl_inputs.jsonl
export HF_HOME=$U/hf_cache TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
snap() { python3 -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"; }
mkdir -p $Q/ctrl_acts_llama $Q/ctrl_acts_olmo $Q/ctrl_acts_qwen
( CUDA_VISIBLE_DEVICES=0 python3 -u $H/acts_multi.py /capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-8B-Instruct $I $Q/ctrl_acts_llama alpaca,wiki,harmful 4,8,12,16,20,24,28,31 0 1 \
  && python3 -u $H/acts_multi.py --merge $Q/ctrl_acts_llama alpaca,wiki,harmful 4,8,12,16,20,24,28,31 1 ) > $Q/logs/ctrl_llama.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$U/pyoverlay/gemma4_tf58 python3 -u $H/acts_multi.py google/gemma-4-31B-it $I $Q/ctrl_acts harmful 15,20,25,30,35,40,45 0 1 --gemma \
  && PYTHONPATH=$U/pyoverlay/gemma4_tf58 python3 -u $H/acts_multi.py --merge $Q/ctrl_acts harmful 15,20,25,30,35,40,45 1 ) > $Q/logs/ctrl_gemma.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2 python3 -u $P/olmo3_7b/scripts/acts.py $(snap allenai/Olmo-3-7B-Instruct) $I $Q/ctrl_acts_olmo alpaca,wiki,harmful && python3 -u - <<'PY'
import torch, numpy as np, json
D = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb/q1_detection/ctrl_acts_olmo"
for st in ("alpaca", "wiki", "harmful"):
    a = torch.load(f"{D}/acts_{st}.pt")
    for j, L in enumerate(a["layers"]): np.save(f"{D}/{st}_L{L}.npy", a["X"][:, j].numpy())
    json.dump({"pair_ids": a["pair_ids"], "tail_tokens": a["tail_tokens"], "layers": a["layers"]}, open(f"{D}/{st}_meta.json", "w")); print("converted", st, tuple(a["X"].shape))
PY
) > $Q/logs/ctrl_olmo.log 2>&1 &
srun --overlap -n1 --gres=gpu:4 --environment=/users/vvmoskvoretskii/MR-Eval/container/harmbench.toml bash -c \
  "export HF_HOME=$HF_HOME HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
   M=\$(python3 -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-32B', local_files_only=True))\"); \
   CUDA_VISIBLE_DEVICES=3 python3 -u $P/qwen3_32b/sweep/qwen_acts_sweep.py \$M $Q/ctrl_acts_qwen alpaca,wiki,harmful $I" > $Q/logs/ctrl_qwen.log 2>&1 &
wait
for m in llama olmo qwen gemma; do echo "== $m"; grep -h "rows in\|merged\|converted\|: [0-9]* in\|Error\|Traceback" $Q/logs/ctrl_$m.log | cut -c1-200; done
ls $Q/ctrl_acts_llama/*.npy $Q/ctrl_acts_olmo/*.npy $Q/ctrl_acts_qwen/*.npy $Q/ctrl_acts/harmful_*.npy | wc -l   # expect 24 + 24 + 48 + 7 = 103
