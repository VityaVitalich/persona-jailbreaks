#!/usr/bin/env python3
"""
Style of the jailbroken responses under different steering directions, same 300 originally refused HarmBench prompts,
official snapshot (steer_bench/, steer_v2_results/). Per arm, over its jailbroken responses (corrected for repetitive
junk), the share that
  character_tag   open with a speaker tag in a character's voice ("[Anarchy]:", "Titanius:", "DAN:", "SJ:")
  narrative       read as fiction or narration (story words, chapters, scene setting, first-person storytelling)
  instructions    give step-by-step or list instructions (numbered steps, "Step 1", bullet lists)
  disclaimer      carry a safety disclaimer ("for educational purposes", "I must note", "please note", "I don't condone")
  profanity       contain profanity
plus mean length. Overlap: Jaccard index of the sets of jailbroken prompts between arms. Writes jb_response_style.json.
"""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; NS = HERE.parent
sys.path.insert(0, str(NS))
exec(compile(open(NS / "nopersona_steer_results.py").read().split("def load")[0], str(NS / "nopersona_steer_results.py"), "exec"))
SB, OLD = NS / "steer_bench", NS / "steer_v2_results"
ARMS = {"PersonaJB direction": SB / "harmbench_persona.jsonl", "Jailbreak direction, persona removed (w_out)": SB / "harmbench_dmu_nopersona.jsonl",
        "Refusal direction, persona removed": SB / "harmbench_refusal_nopersona.jsonl", "Alpaca detector": OLD / "alpaca1_alpacajb_d2.jsonl",
        "Alpaca detector, persona removed": OLD / "nopersona_det_alpacajb_d2_nopersona.jsonl", "Random direction": SB / "harmbench_random_full_s0.jsonl"}
F = {"character_tag": lambda t: bool(re.match(r"^\W{0,3}(?:\[[^\]\n]{1,25}\]|[A-Z][\w .'-]{1,20}):\s", t)),
     "narrative": lambda t: bool(re.search(r"\b(?:once upon|chapter|the story|our story|scene|narrator|he said|she said|whispered|"
                                          r"in the dimly lit|the protagonist|in a world where|I remember|let me tell you a tale)\b", t[:800], re.I)),
     "instructions": lambda t: bool(re.search(r"(?:^|\n)\s*(?:step\s*1|1[.)]\s|\*\*step|- \*\*|\* \*\*)", t[:1500], re.I)),
     "disclaimer": lambda t: bool(re.search(r"educational purposes|I must (?:note|emphasize|stress)|please note|do not condone|don't condone|"
                                           r"not (?:recommend|endorse)|is illegal|disclaimer", t, re.I)),
     "profanity": lambda t: bool(re.search(r"\b(?:fuck\w*|shit\w*|damn|bitch\w*|ass(?:hole)?|crap|hell)\b", t, re.I))}
res, sets = {}, {}
print(f"{'arm':46s} {'n jb':>5s} " + " ".join(f"{k[:12]:>12s}" for k in F) + f" {'mean len':>9s}")
for n, p in ARMS.items():
    R = [json.loads(l) for l in open(p)]
    J = [r for r in R if r["final_jailbroken"] and not char_degenerate(r["final_response"] or "")]
    sets[n] = {r["pair_id"] for r in J}
    v = {k: float(np.mean([f(r["final_response"]) for r in J])) for k, f in F.items()}
    v.update(n_jb=len(J), mean_len=float(np.mean([len(r["final_response"]) for r in J])))
    res[n] = v
    print(f"{n:46s} {len(J):5d} " + " ".join(f"{v[k]:12.2f}" for k in F) + f" {v['mean_len']:9.0f}")
names = list(ARMS)
print("\nJaccard overlap of jailbroken prompt sets")
print(" " * 46 + " ".join(f"{i:>6d}" for i in range(len(names))))
jac = {}
for i, a in enumerate(names):
    row = [len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for b in names]; jac[a] = dict(zip(names, row))
    print(f"{i} {a[:44]:44s} " + " ".join(f"{x:6.2f}" for x in row))
res["jaccard"] = jac
json.dump(res, open(HERE / "jb_response_style.json", "w"), indent=1)
