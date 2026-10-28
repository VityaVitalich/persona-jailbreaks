#!/usr/bin/env python3
"""Qwen3-32B detection inputs, the same prompts and pair ids as the Gemma sets:
  hb   HarmBench human_jailbreak 3180 (20 templates x 159, template.format(behavior)) + 159 direct requests
       (pairs = Gemma harmbench_activations_gemma rows; templates = gemma_steering_inputs/hb_templates.json)
  wjb  WildJailbreak eval adversarial_harmful 2000 (pairs = Gemma wildjailbreak rows; prompt = behavior_text)
  pap  HarmBench PAP olmo3_7b_top5 test cases (target-agnostic), built as collect_harmbench_attack_activations_gemma
       build_pairs; must reproduce the Gemma pap rows' (pair_id, behavior_id) order
Row: set, pair_id, behavior_id, behavior_text (judge input), semantic_category, attack_type, jailbreak_idx, prompt.
Writes qwen_inputs.jsonl."""
import csv, json
from pathlib import Path
H = Path(__file__).resolve().parent; O = H.parent.parent
L = lambda p: [json.loads(l) for l in open(p) if l.strip()]
out = []
tp = json.load(open(O / "gemma_steering_inputs/hb_templates.json"))
for r in L(O / "harmbench_activations_gemma/classified_responses.jsonl"):
    p = r["behavior_text"] if r["attack_type"] == "direct_request" else tp[str(r["jailbreak_idx"])].format(r["behavior_text"])
    out.append(dict(set="hb", **{k: r[k] for k in ("pair_id", "behavior_id", "behavior_text", "semantic_category", "attack_type", "jailbreak_idx")}, prompt=p))
src = {int(x["i"]): x["adversarial"].strip() for x in L(H / "wjb_eval_adversarial_harmful.jsonl")}
for r in L(O / "wildjailbreak_activations_gemma/classified_responses.jsonl"):
    assert src[int(r["jailbreak_idx"])] == r["behavior_text"].strip(), r["pair_id"]
    out.append(dict(set="wjb", **{k: r[k] for k in ("pair_id", "behavior_id", "behavior_text", "semantic_category", "attack_type", "jailbreak_idx")}, prompt=r["behavior_text"].strip()))
beh = {b["BehaviorID"]: b for b in csv.DictReader(open(H / "harmbench_behaviors_text_all.csv", encoding="utf-8"))}
merged = {}
for bdir in sorted((H / "pap_test_cases/test_cases_individual_behaviors").iterdir()):
    if not (bdir / "test_cases.json").exists(): continue
    for bid, ps in json.load(open(bdir / "test_cases.json")).items():
        flat = [q for p in ps for q in (p if isinstance(p, list) else [p]) if isinstance(q, str)]
        if flat: merged[bid] = merged.get(bid, []) + flat
pid, pap = 0, []
for bid, ps in merged.items():
    if bid not in beh: continue
    for p in ps:
        if p and p.strip():
            pap.append(dict(set="pap", pair_id=pid, behavior_id=bid, behavior_text=beh[bid]["Behavior"], semantic_category=beh[bid].get("SemanticCategory", ""),
                            attack_type="pap", jailbreak_idx=-1, prompt=p.strip())); pid += 1
g = L(O / "pap_activations_gemma_hb/classified_responses.jsonl")
assert [(int(r["pair_id"]), r["behavior_id"]) for r in g] == [(r["pair_id"], r["behavior_id"]) for r in pap], "PAP order differs from Gemma"
out += pap
with open(H / "qwen_inputs.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
import collections; print(collections.Counter(r["set"] for r in out), "| chars max", max(len(r["prompt"]) for r in out))
