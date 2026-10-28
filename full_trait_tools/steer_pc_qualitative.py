#!/usr/bin/env python3
"""Qualitative results for the PCA axes (Dominic, pca_llama table): steer a benign prompt
along +/- each principal component of the layer-16 activations and show what the model
generates, Assistant-Axis style (a descending coefficient ladder, multiple prompts).
Recomputes PC1/PC2 from the paper's activation cloud, orients them by the trait poles, and
adds coeff*||h||*PC to the layer-16 residual stream at every position during generation."""
import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VECS = "full_trait_output/paper_data/building_blocks/persona_trait_vectors_llama"
HB = "full_trait_output/paper_data/activations/harmbench"
L = 16
OUT = Path("full_trait_output/pc_qualitative"); OUT.mkdir(parents=True, exist_ok=True)

# --- PCA on the layer-16 activation cloud (matches the pca_llama table) ---
acts = torch.load(f"{HB}/activations.pt", map_location="cpu", weights_only=False)
A = np.stack([v["16"].float().numpy() for v in acts.values()]).astype(np.float32)
mu = A.mean(0); Ac = A - mu
U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
var = (S ** 2) / (S ** 2).sum()
PC = Vt.copy()                                   # rows = components (unit)
print("var%%: PC1 %.1f  PC2 %.1f  PC3 %.1f  PC4 %.1f" % tuple(100 * var[:4]), flush=True)


def tvec(name):
    v = torch.load(f"{VECS}/{name}.pt", map_location="cpu", weights_only=False)["vector"].float().numpy()[L]
    return v / (np.linalg.norm(v) + 1e-12)


# each component oriented so + points to the pole named in the pca table
AXDEF = [
    ("PC2", "+ principled",        "- evil/hostile",        1, "principled"),
    ("PC3", "+ introspective",     "- reserved",            2, "introspective"),
    ("PC4", "+ ritualistic",       "- casual",              3, "ritualistic"),
]
AXES = []
for name, pos_lab, neg_lab, idx, orient in AXDEF:
    v = PC[idx].copy()
    if v @ tvec(orient) < 0:
        v = -v
    AXES.append((name, pos_lab, neg_lab, v))

PROMPTS = [
    ("phone", "What is your opinion on people who talk loudly on the phone in public?"),
    ("neighbor", "What should I do about a neighbor who is constantly rude to me?"),
    ("meaning", "What do you think is the meaning of a good life?"),
    ("morning", "How should I structure my ideal morning routine?"),
]
MULTS = [1.3, 0.9, 0.0, -0.9, -1.3]
OUTNAME = "steer_pc_qualitative_v3.json"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
layer = model.model.layers[L]

state = {"vec": None, "mult": 0.0}


def hook(mod, inp, out):
    if state["mult"] == 0.0 or state["vec"] is None:
        return out
    h = out[0] if isinstance(out, tuple) else out
    v = torch.tensor(state["vec"], dtype=h.dtype, device=h.device)
    # steer only the last position (the token being generated) to preserve prompt reading
    norm = h[:, -1:, :].norm(dim=-1, keepdim=True)
    h[:, -1:, :] = h[:, -1:, :] + state["mult"] * norm * v
    return (h,) + out[1:] if isinstance(out, tuple) else h


layer.register_forward_hook(hook)


def gen(text):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=70, do_sample=False)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")


results = {}
for axname, pos_lab, neg_lab, vec in AXES:
    for pk, ptext in PROMPTS:
        rows = []
        for m in MULTS:
            state["vec"] = vec; state["mult"] = float(m)
            g = gen(ptext)
            rows.append({"mult": m, "text": g})
            print(f"[{axname} {pk} mult={m:+.1f}] {g[:150]}", flush=True)
        state["mult"] = 0.0
        results[f"{axname}::{pk}"] = {"axis": axname, "pos": pos_lab, "neg": neg_lab,
                                      "prompt": ptext, "rows": rows}
json.dump(results, open(OUT / OUTNAME, "w"), indent=1)
print("saved", OUT / OUTNAME, flush=True)
