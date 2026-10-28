#!/usr/bin/env python3
"""Qwen3-32B readout for the layer x position sweep: decoder-layer outputs at LAYERS, for the last 9 prompt positions
(chat-template tail <|im_end|> \\n <|im_start|> assistant \\n <think> \\n\\n </think> \\n\\n) plus the mean over the
user-message tokens (index 9). Same chat text as qwen_acts_pos.py (enable_thinking=False), HF bf16, sdpa, left padding.
Writes OUTDIR/{set}_L{layer}.npy fp16 [n, 10, 5120] and OUTDIR/{set}_meta.json {"pair_ids", "tail_tokens"}.
usage: qwen_acts_sweep.py MODEL OUTDIR SET[,SET..] inputs.jsonl [inputs2.jsonl ...]"""
import json, os, sys, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, outdir, SETS, inps = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:]
LAYERS, NPOS, BUDGET = list(range(4, 64, 4)) + [63], 9, 32768
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
PRE = tok("<|im_start|>user\n", add_special_tokens=False).input_ids
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
cap, M = {}, {}
def mk(l):
    def hook(m, i, o):
        h = (o[0] if isinstance(o, tuple) else o).float()
        cap[l] = torch.cat([h[:, -NPOS:], (M["w"] @ h)], 1).cpu()   # [B, 10, d]; M["w"]: [B, 1, S] user-token mean weights
    return hook
for l in LAYERS: model.model.layers[l].register_forward_hook(mk(l))
allrows = [json.loads(l) for f in inps for l in open(f)]
for st in SETS:
    if all(os.path.exists(f"{outdir}/{st}_L{l}.npy") for l in LAYERS): print(st, "exists, skip"); continue
    rows = [r for r in allrows if r["set"] == st]
    T = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    ids = [tok(t, add_special_tokens=False).input_ids for t in T]; assert all(x[:2] == PRE[:2] for x in ids)
    npre = [next((k for k in range(len(PRE)) if x[k] != PRE[k]), len(PRE)) for x in ids]   # a leading newline/space in the prompt can merge with the template's \n
    print(st, 'rows whose prompt merges into the template prefix:', sum(n < len(PRE) for n in npre), flush=True)
    tail = tok.convert_ids_to_tokens(ids[0][-NPOS:]); print(st, len(rows), "tail", tail, flush=True)
    L = [len(x) for x in ids]; order = sorted(range(len(T)), key=lambda i: L[i]); out = {l: np.zeros((len(rows), NPOS + 1, 5120), np.float16) for l in LAYERS}; b, t0 = 0, time.time()
    while b < len(order):
        e = b + 1
        while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 64: e += 1
        idx = order[b:e]; S = max(L[i] for i in idx)
        enc = tok([T[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0"); assert enc.input_ids.shape[1] == S
        w = torch.zeros(len(idx), 1, S, device="cuda:0")
        for j, i in enumerate(idx): s0 = S - L[i] + npre[i]; s1 = S - NPOS; w[j, 0, s0:s1] = 1.0 / max(s1 - s0, 1)
        M["w"] = w
        with torch.no_grad(): model.model(**enc, use_cache=False)
        for l in LAYERS: out[l][idx] = cap[l].numpy().astype(np.float16)
        b = e
    for l in LAYERS: np.save(f"{outdir}/{st}_L{l}.npy.tmp.npy", out[l]); os.replace(f"{outdir}/{st}_L{l}.npy.tmp.npy", f"{outdir}/{st}_L{l}.npy")
    json.dump({"pair_ids": [r["pair_id"] for r in rows], "tail_tokens": tail + ["<user-mean>"], "layers": LAYERS}, open(f"{outdir}/{st}_meta.json", "w"))
    print(f"{st}: {len(rows)} in {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
