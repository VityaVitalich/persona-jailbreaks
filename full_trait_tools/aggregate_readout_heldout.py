#!/usr/bin/env python3
"""Held-out control for the aggregate readout figure. To rule out train-set memorization,
we refit the trait detector (same 229-trait feature space, logreg L2 C=10) on a
BEHAVIOR-DISJOINT training split, then build the readout aggregate on the held-out
behaviors the detector never saw. Benign (JBB) is fully held-out. Groups: benign, direct
(refused), jailbreak WORKED (complied), jailbreak FAILED (refused), pooled over strong
templates, aligned by the pre-generation readout token. GPU. Writes *_heldout.*"""
import os, json, pickle
os.environ["HF_DATASETS_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PKL = "full_trait_output/paper_data/detection_llama/trained_models/trait_detector.pkl"
HB = "full_trait_output/paper_data/activations/harmbench"
HB_LOCAL = "/dlabscratch1/bazina/HarmBench/baselines/human_jailbreaks/jailbreaks.py"
TEMPLATES = [4, 13, 14, 36, 56, 66, 79, 88]
L = 16
W = 26
OUT = Path("full_trait_output"); OUT.mkdir(parents=True, exist_ok=True)

T = pickle.load(open(PKL, "rb"))["trait_matrix"]          # (229, 4096) fixed trait projection

rows = [json.loads(l) for l in open(f"{HB}/classified_responses.jsonl") if l.strip()]
beh = {r["behavior_id"]: r["behavior_text"] for r in rows}
behaviors = sorted({r["behavior_id"] for r in rows
                    if r.get("attack_type") == "human_jailbreak" and r.get("jailbroken") is not None})
train_beh = set(behaviors[::2]); test_beh = set(behaviors[1::2])
print(f"behaviors: train={len(train_beh)} test={len(test_beh)}", flush=True)

# --- refit detector on TRAIN behaviors' human_jailbreak pairs (all templates) ---
acts = torch.load(f"{HB}/activations.pt", map_location="cpu", weights_only=False)
Xtr, ytr = [], []
for r in rows:
    if r.get("attack_type") != "human_jailbreak" or r.get("jailbroken") is None:
        continue
    if r["behavior_id"] not in train_beh:
        continue
    k = r["pair_id"] if r["pair_id"] in acts else str(r["pair_id"])
    if k not in acts:
        continue
    Xtr.append(acts[k]["16"].float().numpy() @ T.T); ytr.append(int(bool(r["jailbroken"])))
del acts
clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=10.0, max_iter=5000))]).fit(np.array(Xtr), np.array(ytr))
print(f"refit on {len(ytr)} train pairs, jb_rate={np.mean(ytr):.2f}", flush=True)


def logit(A):
    return clf.decision_function(np.atleast_2d(A) @ T.T)


# --- test prompts (held-out behaviors) ---
worked, failed, direct = [], [], []
for r in rows:
    if r["behavior_id"] not in test_beh:
        continue
    at, jb = r.get("attack_type"), r.get("jailbroken")
    if at == "direct_request" and jb is False:
        direct.append(r["behavior_text"])
    elif at == "human_jailbreak" and r.get("jailbreak_idx") in TEMPLATES and jb is not None:
        (worked if bool(jb) else failed).append((r["jailbreak_idx"], r["behavior_text"]))
benign_prompts = [x["Goal"] for x in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["benign"]]
print(f"held-out: worked={len(worked)} failed={len(failed)} direct={len(direct)} benign={len(benign_prompts)}", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def tail_logit(text):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    lg = logit(hs)
    seg = np.full(W + 1, np.nan); t = lg[-(W + 1):]; seg[W + 1 - len(t):] = t
    return seg


def collect(name, texts):
    rr = []
    for i, t in enumerate(texts):
        rr.append(tail_logit(t))
        if i % 50 == 0:
            print(f"  {name} {i}/{len(texts)}", flush=True)
    return np.array(rr)


groups = {
    "benign request (complied)":        ("#8e44ad", collect("benign", benign_prompts)),
    "direct request (refused)":         ("#3b6fb0", collect("direct", direct)),
    "jailbreak that WORKED (complied)": ("#1a9641", collect("worked", [JB[t].format(x) for t, x in worked])),
    "jailbreak that FAILED (refused)":  ("#c0392b", collect("failed", [JB[t].format(x) for t, x in failed])),
}
np.savez(OUT / "aggregate_readout_heldout.npz", **{k.split(" (")[0]: v[1] for k, v in groups.items()}, W=W)

# held-out separation AUC at the readout token (worked vs failed)
wv = groups["jailbreak that WORKED (complied)"][1][:, -1]
fv = groups["jailbreak that FAILED (refused)"][1][:, -1]
auc = roc_auc_score([1] * len(wv) + [0] * len(fv), np.concatenate([wv, fv]))
print(f"HELD-OUT readout AUC (worked vs failed) = {auc:.3f}", flush=True)

x = np.arange(-W, 1)
fig, ax = plt.subplots(figsize=(12, 5.4))
for lab, (col, M) in groups.items():
    n = np.sum(~np.isnan(M), axis=0)
    m = np.nanmean(M, axis=0); se = np.nanstd(M, axis=0) / np.sqrt(np.maximum(n, 1))
    keep = n >= 5
    ax.plot(x[keep], m[keep], color=col, lw=2.6, label=f"{lab}  (n={M.shape[0]})")
    ax.fill_between(x[keep], (m - 1.96 * se)[keep], (m + 1.96 * se)[keep], color=col, alpha=0.18)
    ax.plot([0], [m[-1]], marker="*", ms=20, color=col, markeredgecolor="k", zorder=6)
ax.axhline(0, color="red", ls="--", lw=1.5, label="classifier boundary")
ax.axvspan(-4.5, 0.5, color="0.85", alpha=0.55, zorder=0)
ax.axvline(-4.5, color="0.4", lw=1.0)
ax.text(-2.0, ax.get_ylim()[1], "assistant header\n+ readout", ha="center", va="top", fontsize=8, color="0.3")
ax.set_xlabel("token position relative to the pre-generation readout token (0 = readout)")
ax.set_ylabel("all-traits classifier logit\n(+ comply / - refuse)")
ax.set_title(f"Held-out aggregate readout (detector refit on disjoint behaviors; "
             f"worked-vs-failed AUC={auc:.2f}, layer {L}, 95% CI)", fontsize=11)
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "aggregate_readout_heldout.png", dpi=180)
print("saved", OUT / "aggregate_readout_heldout.png", flush=True)
for lab, (col, M) in groups.items():
    print(f"{lab:40s} n={M.shape[0]:4d}  readout mean = {np.nanmean(M[:, -1]):+.2f}  content mean(pos<=-6) = {np.nanmean(M[:, :W-5]):+.2f}")
