#!/usr/bin/env python3
"""Aggregate per-token 3-trait classifier logit over many direct-vs-jailbreak (template-79)
prompts, to test 'suppress-then-spike' as a measured effect rather than an anecdote."""
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
L, TMPL, N = 16, 79, 40
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
print(f"clf3 trained on {len(ytr)} pairs", flush=True)


def logit3(A):
    return clf.decision_function(np.atleast_2d(A) @ V3.T)


dr_ref = {r["behavior_id"] for r in rows if r.get("attack_type") == "direct_request" and not r["jailbroken"]}
t79 = {r["behavior_id"]: r["behavior_text"] for r in rows
       if r.get("attack_type") == "human_jailbreak" and r.get("jailbreak_idx") == TMPL and r["jailbroken"]}
flips = sorted(set(dr_ref) & set(t79))[:N]
del acts
print(f"{len(flips)} flip behaviors", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def run(text):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    return ids[0].tolist(), hs


def onset(ids, bt_text):
    bt = tok(bt_text, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(bt) + 1):
        if ids[i:i + len(bt)] == bt:
            return i
    for i in range(len(ids) - 4 + 1):
        if ids[i:i + 4] == bt[:4]:
            return i
    return -1


jb_traj, jb_onset, dr_read, jb_read = [], [], [], []
for b in flips:
    bt = t79[b]
    ids, hs = run(JB[TMPL].format(bt)); lg = logit3(hs)
    jb_traj.append(lg.astype(np.float32)); jb_onset.append(onset(ids, bt)); jb_read.append(float(lg[-1]))
    _, hs2 = run(bt); dr_read.append(float(logit3(hs2)[-1]))
    print(f"{b:40s} dr={dr_read[-1]:+.1f} jb={jb_read[-1]:+.1f} onset={jb_onset[-1]}", flush=True)

np.savez(OUT / "aggregate.npz",
         behaviors=np.array(flips, dtype=object),
         jb_traj=np.array(jb_traj, dtype=object),
         jb_onset=np.array(jb_onset),
         dr_read=np.array(dr_read), jb_read=np.array(jb_read))
print("saved", OUT / "aggregate.npz", flush=True)
