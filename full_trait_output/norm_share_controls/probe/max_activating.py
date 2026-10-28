#!/usr/bin/env python3
"""
Max-activating texts for w_out (the jailbreak direction with its persona component removed) and, for comparison,
w_in (its persona component), on saved layer-16 last-token activations of natural text:
  Wikipedia   2000 WikiText-103 paragraphs read as plain text (wiki/wiki_acts.pt "raw")
  No Robots   2000 benign user requests in the chat template (wiki/norobots_acts.pt "chat")
  Alpaca      797 plain Alpaca requests + 897 plain harmful requests (roleplay_check synthetic_v2)
For each set: the 8 highest and 8 lowest scoring texts on each direction (z-scored within the set), and the
correlation of the score with text length. Sign convention: + = toward jailbroken. Writes max_activating.json.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; NS = HERE.parent; OUT = NS.parent
b = torch.load(NS / "steer_probe_layer16.pt", weights_only=False)["vectors"]
W = {"w_out": b["w_out"].double().numpy(), "w_in": b["w_in"].double().numpy()}
sets = {}
for name, actf, rd, txtf in (("Wikipedia", "wiki_acts.pt", "raw", "wiki_passages.jsonl"), ("No Robots", "norobots_acts.pt", "chat", "norobots_prompts.jsonl")):
    A = torch.load(NS / "wiki" / actf, map_location="cpu")[rd]; T = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(NS / "wiki" / txtf)}
    ids = [i for i in A if i in T]; sets[name] = (np.stack([A[i].float().numpy() for i in ids]).astype(np.float64), [T[i] for i in ids])
R = [json.loads(l) for l in open(OUT / "roleplay_check/synthetic_v2_prompts.jsonl")]
V = torch.load(OUT / "roleplay_check/synthetic_v2_acts.pt", map_location="cpu")
pl = [r for r in R if r["frame"] == "plain"]
sets["Plain requests (Alpaca + harmful)"] = (np.stack([V[r["id"]].float().numpy() for r in pl]).astype(np.float64), [f"[{r['harm']}] {r['prompt']}" for r in pl])
res = {}
for sn, (X, T) in sets.items():
    L = np.array([len(t) for t in T])
    for dn, w in W.items():
        s = X @ w; z = (s - s.mean()) / s.std(); o = np.argsort(z)
        res[f"{sn}|{dn}"] = dict(corr_length=float(np.corrcoef(s, np.log(L))[0, 1]),
                                 top=[(float(z[i]), T[i][:300]) for i in o[::-1][:8]], bottom=[(float(z[i]), T[i][:300]) for i in o[:8]])
        print(f"\n######## {sn} | {dn}   corr(score, log length) {res[f'{sn}|{dn}']['corr_length']:+.2f}")
        print("  HIGH (toward jailbroken):"); [print(f"   {zz:+.1f}  {t[:170]!r}") for zz, t in res[f"{sn}|{dn}"]["top"]]
        print("  LOW (toward refused):"); [print(f"   {zz:+.1f}  {t[:170]!r}") for zz, t in res[f"{sn}|{dn}"]["bottom"]]
json.dump(res, open(HERE / "max_activating.json", "w"), indent=1)
