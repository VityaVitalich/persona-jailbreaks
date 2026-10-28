#!/usr/bin/env python3
"""Aggregate of the readout-token figure: mean all-traits classifier logit across many
behaviors, aligned by the pre-generation READOUT token (position 0), split into three
groups -- direct request, successful template-79 jailbreak, failed template-79 jailbreak.
The shared assistant-header tokens (positions -4..0) are where the groups fan out to their
verdicts. No generation needed (uses the dataset's jailbroken labels). GPU."""
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
TEMPLATE = 79
L = 16
W = 26          # tokens back from the readout token to keep
N_PER = 45      # cap per jailbreak group (balanced)
OUT = Path("full_trait_output"); OUT.mkdir(parents=True, exist_ok=True)

clf = pickle.load(open(CLF, "rb"))
Tmat, pipe = clf["trait_matrix"], clf["pipeline"]
sign = clf.get("sign", 1)

beh, succ, fail = {}, [], []
for l in open(CLASSIFIED):
    if not l.strip():
        continue
    r = json.loads(l)
    beh.setdefault(r["behavior_id"], r["behavior_text"])
    if r.get("attack_type") == "human_jailbreak" and r.get("jailbreak_idx") == TEMPLATE and r.get("jailbroken") is not None:
        (succ if bool(r["jailbroken"]) else fail).append(r["behavior_id"])
succ = sorted(set(succ)); fail = sorted(set(fail))[:N_PER]
direct_beh = sorted(set(succ) | set(fail))
benign_prompts = [r["Goal"] for r in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["benign"]]
print(f"succ={len(succ)} fail={len(fail)} direct={len(direct_beh)} benign={len(benign_prompts)}", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def tail_logit(text):
    """Per-token all-traits logit, last W+1 tokens aligned so index W = readout token."""
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    logit = pipe.decision_function(hs @ Tmat.T)
    if sign < 0:
        logit = -logit
    seg = np.full(W + 1, np.nan)
    t = logit[-(W + 1):]
    seg[W + 1 - len(t):] = t          # right-align: last element = readout token
    return seg


def collect(name, texts):
    rows = []
    for i, t in enumerate(texts):
        rows.append(tail_logit(t))
        if i % 10 == 0:
            print(f"  {name} {i}/{len(texts)}", flush=True)
    return np.array(rows)


groups = {
    "benign request (complied)":            ("#8e44ad", collect("benign", benign_prompts)),
    "direct request (refused)":            ("#3b6fb0", collect("direct", [beh[b] for b in direct_beh])),
    "jailbreak that WORKED (complied)":     ("#1a9641", collect("succ", [JB[TEMPLATE].format(beh[b]) for b in succ])),
    "jailbreak that FAILED (refused)":      ("#c0392b", collect("fail", [JB[TEMPLATE].format(beh[b]) for b in fail])),
}
np.savez(OUT / "aggregate_readout_benign.npz", **{k.split(" (")[0]: v[1] for k, v in groups.items()}, W=W)

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
ax.set_title(f"Aggregate readout: the verdict crystallizes at the readout token "
             f"(template-{TEMPLATE}, layer {L}, 95% CI)", fontsize=12)
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "aggregate_readout_benign.png", dpi=180)
print("saved", OUT / "aggregate_readout_benign.png", flush=True)
for lab, (col, M) in groups.items():
    print(f"{lab:40s} readout mean = {np.nanmean(M[:, -1]):+.2f}   content mean(pos<=-6) = {np.nanmean(M[:, :W-5]):+.2f}")
