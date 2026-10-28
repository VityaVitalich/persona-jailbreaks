#!/usr/bin/env python3
"""
Rebuild the exact prompt text of every Llama detection row (persona-jailbreaks paper, tab:detection).

Rows and pair_ids come from the saved response dumps; prompt text is reconstructed the way the
collectors built it:
  HB       human_jailbreak rows of harmbench_activations: HarmBench JAILBREAKS[jailbreak_idx].format(behavior)
           (collect_harmbench_activations.py; templates 16/44 skipped there, none of them sampled)
  GCG/PAP/GPTFuzz/PEZ  HarmBench test_cases.json, flattened per behaviour in file order
           (collect_attack_activations.py / collect_all_layers_activations.build_attack_rows)
  PAIR     per behaviour two rows: test_cases[b][0] (attack_type PAIR) then the plain behaviour
           (attack_type plain), exactly as build_attack_rows rebuilds them
  WJB      WildJailbreak adversarial prompt = behavior_text of the saved row (not in the Llama
           detection table; classified for completeness)
Alignment (behaviour id + attack_type per row) is asserted against the saved rows.

Output: prompts.jsonl  {set, pair_id, behavior_id, attack_type, jailbroken, prompt}
"""
import json
from pathlib import Path

OUT = Path.home() / "persona-vector-jbing/full_trait_output"
HERE = OUT / "roleplay_check"
SRC = HERE / "src"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def flatten(prompts):
    flat = []
    for p in prompts:
        flat.extend(p if isinstance(p, list) else [p])
    return [p for p in flat if isinstance(p, str) and p.strip()]


def sanitize(t):  # = collectors' sanitize_text
    return t.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def jb(v):
    return {"True": 1, "False": 0, True: 1, False: 0}.get(v)


def main():
    ns = {}
    exec(open(SRC / "jailbreaks.py", encoding="utf-8").read(), ns)
    JB = ns["JAILBREAKS"]
    out = []

    for r in load_jsonl(OUT / "harmbench_activations/classified_responses.jsonl"):
        if r["attack_type"] != "human_jailbreak":
            continue
        idx = int(r["jailbreak_idx"])
        out.append(dict(set="HB", pair_id=int(r["pair_id"]), behavior_id=r["behavior_id"],
                        attack_type="human_jailbreak", template=idx, jailbroken=jb(r["jailbroken"]),
                        prompt=JB[idx].format(r["behavior_text"])))

    for fam in ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]:
        rows = load_jsonl(OUT / f"{fam.lower()}_activations/responses.jsonl")
        tc = json.load(open(SRC / f"{fam}_test_cases.json", encoding="utf-8"))
        rebuilt = []
        for bid, prompts in tc.items():
            if fam == "PAIR":
                rebuilt += [(bid, "PAIR", prompts[0]), (bid, "plain", None)]
            else:
                rebuilt += [(bid, fam, p) for p in flatten(prompts)]
        rebuilt = [x for x in rebuilt if x[0] in {r["behavior_id"] for r in rows}]
        assert len(rebuilt) == len(rows), (fam, len(rebuilt), len(rows))
        for r, (bid, at, p) in zip(rows, rebuilt):
            assert r["behavior_id"] == bid and r["attack_type"] == at, (fam, r["pair_id"], bid, at)
            out.append(dict(set=fam, pair_id=int(r["pair_id"]), behavior_id=bid, attack_type=at,
                            jailbroken=jb(r["jailbroken"]), prompt=p if p is not None else r["behavior_text"]))

    for r in load_jsonl(OUT / "wildjailbreak_activations/classified_responses.jsonl"):
        out.append(dict(set="WJB", pair_id=int(r["pair_id"]), behavior_id=r["behavior_id"],
                        attack_type=r["attack_type"], jailbroken=jb(r["jailbroken"]), prompt=r["behavior_text"]))

    with open(HERE / "prompts.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            o["prompt"] = sanitize(o["prompt"])
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter((o["set"], o["attack_type"]) for o in out))
    print("unique prompts:", len({o["prompt"] for o in out}), "of", len(out))


if __name__ == "__main__":
    main()
