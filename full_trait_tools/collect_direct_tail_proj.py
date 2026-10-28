#!/usr/bin/env python3
"""
collect_direct_tail_proj.py

Collect per-token 229-trait projections for the REFUSED DIRECT REQUESTS,
mirroring the per_position_probe_recast protocol (layer 16, last W+1 prompt
tokens right-aligned so index W = pre-generation readout token, NaN-padded).
The jailbreak groups already have this dump (per_position_probe_recast_data.npz);
direct requests were never collected, so the per-trait readout figure could not
show the third group without this.

Differences from the cluster recast collection, both deliberate:
  * runs locally on MPS with the NousResearch weight mirror (identical weights
    and chat template to meta-llama/Llama-3.1-8B-Instruct);
  * direct prompts are shorter than the window, so the <|begin_of_text|>
    attention-sink token can fall inside it — its projection is masked to NaN
    (jailbreak prompts are all longer than the window, so this case never
    arose there).

Validates the forward pass by correlating readout-token projections against
the cluster-collected activations in harmbench_activations/activations.pt.

Usage:
  python3 full_trait_tools/collect_direct_tail_proj.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"   # weight mirror of meta-llama
# the NousResearch mirror strips the official chat template's default system
# block ("Cutting Knowledge Date ... Today Date: 26 Jul 2024"), which the
# cluster collections included; take the tokenizer from the unsloth mirror,
# which keeps the official template verbatim
TOKENIZER = "unsloth/Meta-Llama-3.1-8B-Instruct"
PKL = "full_trait_output/trait_detector.pkl"
HB = "full_trait_output/harmbench_activations"
L = 16
W = 26
OUT = Path("full_trait_output/direct_request_tail_proj.npz")

T = pickle.load(open(PKL, "rb"))["trait_matrix"]     # (229, 4096)

rows = [json.loads(l) for l in open(f"{HB}/classified_responses.jsonl") if l.strip()]
direct = [r for r in rows if r.get("attack_type") == "direct_request"
          and r.get("jailbroken") is False]
print(f"refused direct requests: {len(direct)}", flush=True)

tok = AutoTokenizer.from_pretrained(TOKENIZER)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="mps").eval()


def tail_proj(text):
    """229-trait projection at each of the last W+1 token positions,
    right-aligned (index W = readout token); NaN-padded, BOS masked."""
    chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True,
                   return_dict=True).hidden_states[L + 1][0].float().cpu().numpy()
    proj = hs @ T.T                                  # [n_tok, 229]
    if ids.shape[1] <= W + 1:                        # BOS inside the window
        proj[0] = np.nan
    seg = np.full((W + 1, 229), np.nan, dtype=np.float32)
    t = proj[-(W + 1):]
    seg[W + 1 - len(t):] = t
    return seg


X, pair_ids = [], []
for i, r in enumerate(direct):
    X.append(tail_proj(r["behavior_text"]))
    pair_ids.append(str(r["pair_id"]))
    if i % 20 == 0:
        print(f"  {i}/{len(direct)}", flush=True)
X = np.stack(X)
np.savez(OUT, X=X, pair_ids=np.array(pair_ids), W=W)
print("saved", OUT, X.shape, flush=True)

# --- validate readout-token projections against the cluster collection ---
acts = torch.load(f"{HB}/activations.pt", map_location="cpu", weights_only=False)
mine, theirs = [], []
for seg, pid in zip(X, pair_ids):
    k = pid if pid in acts else (int(pid) if int(pid) in acts else None)
    if k is None:
        continue
    mine.append(seg[-1])
    theirs.append(acts[k]["16"].float().numpy() @ T.T)
mine, theirs = np.array(mine), np.array(theirs)
r = np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]
rel = np.abs(mine - theirs).mean() / np.abs(theirs).mean()
print(f"readout validation vs cluster acts: n={len(mine)}  corr={r:.6f}  "
      f"mean rel err={rel:.4f}", flush=True)
