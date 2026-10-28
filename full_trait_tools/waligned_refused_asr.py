"""ASR of w_aligned steering on the baseline-refused HarmBench pairs (tab:attack cell).

The templated w_aligned run (harmbench_waligned_tpl_a7, 400 behaviours x 112 templates)
covers all 3180 human_jailbreak pairs, so the started-refused / started-jailbroken split
used by steer_persona_w_hb_balanced.py can be recovered by joining on
(behavior_id, jailbreak_idx) against the HarmBench baseline labels.
"""
import glob
import json

HB = "full_trait_output/harmbench_activations/classified_responses.jsonl"
STEER = "full_trait_output/harmbench_waligned_tpl_a7/w_aligned/shard*/results.jsonl"

baseline = {}
for line in open(HB):
    r = json.loads(line)
    if r.get("attack_type") == "human_jailbreak":
        baseline[(r["behavior_id"], r["jailbreak_idx"])] = bool(r["jailbroken"])

steered = {}
for path in glob.glob(STEER):
    for line in open(path):
        r = json.loads(line)
        steered[(r["behavior_id"], r["jailbreak_idx"])] = r

refused = [k for k, jb in baseline.items() if not jb]
started_jb = [k for k, jb in baseline.items() if jb]
missing = [k for k in baseline if k not in steered]
assert not missing, f"{len(missing)} baseline pairs absent from the steering run"


def asr(keys):
    return sum(bool(steered[k]["final_jailbroken"]) for k in keys) / len(keys)


print(f"n_started_not_jailbroken = {len(refused)}   ASR = {asr(refused):.4f}")
print(f"n_started_jailbroken     = {len(started_jb)}   ASR = {asr(started_jb):.4f}")
print(f"n_total                  = {len(baseline)}   ASR = {asr(list(baseline)):.4f}")
