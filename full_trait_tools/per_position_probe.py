#!/usr/bin/env python3
"""Per-token-position probe control for the localization claim. At a fixed layer (16),
for each token position relative to the pre-generation readout token, fit a FRESH logistic
regression on the 229-trait projections of TRAIN behaviors and evaluate worked-vs-failed
AUC on held-out TEST behaviors. Every position gets its own probe, so the content-vs-readout
gap cannot be an artifact of the detector being readout-calibrated. GPU (collection) +
offline probe fitting. Pooled over the 8 balanced templates."""
import json, pickle
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

T = pickle.load(open(PKL, "rb"))["trait_matrix"]          # (229, 4096)
rows = [json.loads(l) for l in open(f"{HB}/classified_responses.jsonl") if l.strip()]
beh = {r["behavior_id"]: r["behavior_text"] for r in rows}
behaviors = sorted({r["behavior_id"] for r in rows
                    if r.get("attack_type") == "human_jailbreak" and r.get("jailbroken") is not None})
train_beh = set(behaviors[::2]); test_beh = set(behaviors[1::2])

# jailbreak prompts pooled over templates, tagged (split, label)
prompts = []   # (template, behavior_text, split, y)
for r in rows:
    if r.get("attack_type") != "human_jailbreak" or r.get("jailbreak_idx") not in TEMPLATES or r.get("jailbroken") is None:
        continue
    split = "train" if r["behavior_id"] in train_beh else "test"
    prompts.append((r["jailbreak_idx"], r["behavior_text"], split, int(bool(r["jailbroken"]))))
print(f"prompts={len(prompts)} train={sum(p[2]=='train' for p in prompts)} test={sum(p[2]=='test' for p in prompts)}", flush=True)

ns = {}; exec(open(HB_LOCAL).read(), ns); JB = ns["JAILBREAKS"]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()


def tail_proj(text):
    """229-trait projection at each of the last W+1 token positions, right-aligned
    (index W = readout token); NaN where the prompt is shorter than the window."""
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    proj = hs @ T.T                                  # [n_tok, 229]
    seg = np.full((W + 1, 229), np.nan, dtype=np.float32)
    t = proj[-(W + 1):]
    seg[W + 1 - len(t):] = t
    return seg


X, Y, S = [], [], []
for i, (tmpl, text, split, y) in enumerate(prompts):
    X.append(tail_proj(JB[tmpl].format(text))); Y.append(y); S.append(split)
    if i % 100 == 0:
        print(f"  {i}/{len(prompts)}", flush=True)
X = np.stack(X); Y = np.array(Y); S = np.array(S)
np.savez(OUT / "per_position_probe_data.npz", X=X, Y=Y, S=S, W=W)
print("collected", X.shape, flush=True)

# per-position probe: fresh logreg fit on train, AUC on test
tr = S == "train"; te = S == "test"
aucs, ns_te = [], []
for p in range(W + 1):
    Xp = X[:, p, :]
    ok = ~np.isnan(Xp).any(axis=1)
    trp = tr & ok; tep = te & ok
    if trp.sum() < 20 or tep.sum() < 20 or len(set(Y[trp])) < 2 or len(set(Y[tep])) < 2:
        aucs.append(np.nan); ns_te.append(int(tep.sum())); continue
    clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=10.0, max_iter=5000))]).fit(Xp[trp], Y[trp])
    aucs.append(roc_auc_score(Y[tep], clf.decision_function(Xp[tep]))); ns_te.append(int(tep.sum()))
aucs = np.array(aucs)
np.savez(OUT / "per_position_probe_auc.npz", aucs=aucs, ns_te=np.array(ns_te), W=W)
x = np.arange(-W, 1)
for p, a, n in zip(x, aucs, ns_te):
    print(f"pos {p:+3d}  AUC={a:.3f}  n_test={n}", flush=True)

fig, ax = plt.subplots(figsize=(11, 5))
keep = ~np.isnan(aucs)
ax.plot(x[keep], aucs[keep], color="#1a3e6e", lw=2.6, marker="o", ms=4)
ax.axhline(0.5, color="0.5", ls=":", lw=1.0, label="chance")
ax.axvspan(-4.5, 0.5, color="0.85", alpha=0.55, zorder=0)
ax.axvline(-4.5, color="0.4", lw=1.0)
ax.text(-2.0, 0.52, "final tokens\n(readout)", ha="center", va="bottom", fontsize=8, color="0.3")
ax.plot([0], [aucs[-1]], marker="*", ms=20, color="#1a9641", markeredgecolor="k", zorder=6)
ax.set_ylim(0.45, 1.0)
ax.set_xlabel("token position relative to the pre-generation readout token (0 = readout)")
ax.set_ylabel("held-out worked-vs-failed AUC\n(fresh probe fit at each position)")
ax.set_title("Position-specific probes: outcome is only decodable at the readout token\n"
             f"(layer {L}, 229-trait features, behavior-disjoint split)", fontsize=12)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "per_position_probe.png", dpi=180)
print("saved", OUT / "per_position_probe.png", flush=True)
