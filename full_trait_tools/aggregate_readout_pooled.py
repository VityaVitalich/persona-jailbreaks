#!/usr/bin/env python3
"""Aggregate readout figure, pooled across several strong (success/fail-balanced) jailbreak
templates for tighter CIs. Groups: benign request, direct request (refused), jailbreak that
WORKED (complied), jailbreak that FAILED (refused). Aligned by the pre-generation readout
token (position 0). No generation needed (dataset jailbroken labels). GPU.
Writes *_pooled.* -- leaves aggregate_readout_benign.* untouched."""
import os, json, pickle
os.environ["HF_DATASETS_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CLF = "full_trait_output/paper_data/detection_llama/trained_models/trait_detector.pkl"
CLASSIFIED = "full_trait_output/paper_data/activations/harmbench/classified_responses.jsonl"
HB_LOCAL = "/dlabscratch1/bazina/HarmBench/baselines/human_jailbreaks/jailbreaks.py"
TEMPLATES = [4, 13, 14, 36, 56, 66, 79, 88]   # success/fail-balanced templates
L = 16
W = 26
OUT = Path("full_trait_output"); OUT.mkdir(parents=True, exist_ok=True)

clf = pickle.load(open(CLF, "rb"))
Tmat, pipe = clf["trait_matrix"], clf["pipeline"]
sign = clf.get("sign", 1)

beh = {}
worked, failed, direct = [], [], []   # lists of prompt strings
for l in open(CLASSIFIED):
    if not l.strip():
        continue
    r = json.loads(l)
    beh.setdefault(r["behavior_id"], r["behavior_text"])
for l in open(CLASSIFIED):
    if not l.strip():
        continue
    r = json.loads(l)
    at = r.get("attack_type"); jb = r.get("jailbroken")
    if at == "direct_request" and jb is False:
        direct.append(r["behavior_text"])
    elif at == "human_jailbreak" and r.get("jailbreak_idx") in TEMPLATES and jb is not None:
        (worked if bool(jb) else failed).append((r["jailbreak_idx"], r["behavior_text"]))

benign_prompts = [r["Goal"] for r in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["benign"]]
print(f"templates={TEMPLATES}\nworked={len(worked)} failed={len(failed)} direct={len(direct)} benign={len(benign_prompts)}", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def tail_logit(text):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    logit = pipe.decision_function(hs @ Tmat.T)
    if sign < 0:
        logit = -logit
    seg = np.full(W + 1, np.nan)
    t = logit[-(W + 1):]
    seg[W + 1 - len(t):] = t
    return seg


def collect(name, texts):
    rows = []
    for i, t in enumerate(texts):
        rows.append(tail_logit(t))
        if i % 50 == 0:
            print(f"  {name} {i}/{len(texts)}", flush=True)
    return np.array(rows)


groups = {
    "benign request (complied)":        ("#8e44ad", collect("benign", benign_prompts)),
    "direct request (refused)":         ("#3b6fb0", collect("direct", direct)),
    "jailbreak that WORKED (complied)": ("#1a9641", collect("worked", [JB[t].format(x) for t, x in worked])),
    "jailbreak that FAILED (refused)":  ("#c0392b", collect("failed", [JB[t].format(x) for t, x in failed])),
}
np.savez(OUT / "aggregate_readout_pooled.npz", **{k.split(" (")[0]: v[1] for k, v in groups.items()}, W=W)

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
ax.set_title(f"Aggregate readout, pooled over {len(TEMPLATES)} jailbreak templates "
             f"(layer {L}, 95% CI)", fontsize=12)
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "aggregate_readout_pooled.png", dpi=180)
print("saved", OUT / "aggregate_readout_pooled.png", flush=True)
for lab, (col, M) in groups.items():
    print(f"{lab:40s} n={M.shape[0]:4d}  readout mean = {np.nanmean(M[:, -1]):+.2f}  content mean(pos<=-6) = {np.nanmean(M[:, :W-5]):+.2f}")
