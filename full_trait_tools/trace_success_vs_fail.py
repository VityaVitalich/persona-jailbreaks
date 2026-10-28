#!/usr/bin/env python3
"""Per-token 3-trait logit trajectories for template-79 jailbreak attempts that SUCCEEDED
vs FAILED. Same template (aligned scaffolding); compare the two mean curves."""
import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VECS = "full_trait_output/paper_data/building_blocks/persona_trait_vectors_llama"
HB = "full_trait_output/paper_data/activations/harmbench"
HB_LOCAL = "/dlabscratch1/bazina/HarmBench/baselines/human_jailbreaks/jailbreaks.py"
PERSONAS = ["egalitarian", "utilitarian", "principled"]
L, TMPL, N = 16, 79, 35
OUT = Path("full_trait_output/persona_token_aggregate"); OUT.mkdir(parents=True, exist_ok=True)

pv = []
for p in PERSONAS:
    v = torch.load(f"{VECS}/{p}.pt", map_location="cpu", weights_only=False)["vector"].float().numpy()[L]
    pv.append(v / np.linalg.norm(v))
V3 = np.stack(pv)

acts = torch.load(f"{HB}/activations.pt", map_location="cpu", weights_only=False)
rows = [json.loads(l) for l in open(f"{HB}/classified_responses.jsonl") if l.strip()]
Xtr, ytr = [], []
for r in rows:
    if r.get("attack_type") != "human_jailbreak" or r.get("jailbroken") is None:
        continue
    k = r["pair_id"] if r["pair_id"] in acts else str(r["pair_id"])
    if k not in acts:
        continue
    Xtr.append(acts[k]["16"].float().numpy() @ V3.T); ytr.append(int(bool(r["jailbroken"])))
clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=4000))]).fit(np.array(Xtr), np.array(ytr))
del acts
print("clf3 trained", flush=True)


def logit3(A):
    return clf.decision_function(np.atleast_2d(A) @ V3.T)


t79 = [(r["behavior_id"], r["behavior_text"], int(bool(r["jailbroken"]))) for r in rows
       if r.get("attack_type") == "human_jailbreak" and r.get("jailbreak_idx") == TMPL and r.get("jailbroken") is not None]
succ = [(b, t) for b, t, y in t79 if y == 1][:N]
fail = [(b, t) for b, t, y in t79 if y == 0][:N]
print(f"template-79: {len(succ)} successful, {len(fail)} failed", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def traj(behavior_text):
    chat = tok.apply_chat_template([{"role": "user", "content": JB[TMPL].format(behavior_text)}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    return logit3(hs).astype(np.float32)


succ_traj = [traj(t) for _, t in succ]
fail_traj = [traj(t) for _, t in fail]
np.savez(OUT / "success_vs_fail.npz",
         succ_traj=np.array(succ_traj, dtype=object),
         fail_traj=np.array(fail_traj, dtype=object))
print("saved", OUT / "success_vs_fail.npz", flush=True)
