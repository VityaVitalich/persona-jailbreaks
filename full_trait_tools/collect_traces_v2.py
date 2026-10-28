#!/usr/bin/env python3
"""Collect per-token 3-trait logit trajectories + payload onset for:
direct requests, template-79 successful/failed jailbreaks, and templates 83/88 successes.
Onset found via token offset mapping (robust). Enables: controlled direct-vs-jailbreak,
success-vs-fail+direct, and multi-template comparisons."""
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
L = 16
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


dr_ref = {r["behavior_id"] for r in rows if r.get("attack_type") == "direct_request" and not r["jailbroken"]}
dr_txt = {r["behavior_id"]: r["behavior_text"] for r in rows if r.get("attack_type") == "direct_request"}


def by_tmpl(idx, jb):
    return {r["behavior_id"]: r["behavior_text"] for r in rows
            if r.get("attack_type") == "human_jailbreak" and r.get("jailbreak_idx") == idx
            and r.get("jailbroken") is not None and bool(r["jailbroken"]) == jb}


TMPLS = [13, 36, 56]  # best classifier separation with a success/fail mix

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def run(user_text, payload_text):
    chat = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
    enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]]).to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    cs = chat.rfind(payload_text[:40])
    onset = 0
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= cs < b or a >= cs:
            onset = i; break
    Hn = hs / (np.linalg.norm(hs, axis=1, keepdims=True) + 1e-12)
    pp = (Hn @ V3.T).astype(np.float32)   # [n_tok, 3] persona cosines
    return logit3(hs).astype(np.float32), onset, pp


records = []  # (group, behavior_id, onset, classifier_traj, persona_proj[n,3])
def collect(group, mapping, build, n):
    for b in sorted(mapping)[:n]:
        tr, on, pp = run(build(mapping[b]), mapping[b])
        records.append((group, b, on, tr, pp))
        print(f"{group:10s} {b[:36]:36s} onset={on} read={tr[-1]:+.1f}", flush=True)

t79s = by_tmpl(79, True)
flip = sorted(set(dr_ref) & set(t79s))
collect("direct",    {b: dr_txt[b] for b in flip if b in dr_txt}, lambda t: t, 45)
collect("jailbreak", {b: t79s[b] for b in flip},                   lambda t: JB[79].format(t), 45)

np.savez(OUT / "traces_v2.npz",
         group=np.array([r[0] for r in records], dtype=object),
         behavior=np.array([r[1] for r in records], dtype=object),
         onset=np.array([r[2] for r in records]),
         traj=np.array([r[3] for r in records], dtype=object),
         persona=np.array([r[4] for r in records], dtype=object))
print("saved", OUT / "traces_v2.npz", flush=True)
