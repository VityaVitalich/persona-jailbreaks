#!/usr/bin/env python3
"""Diagnostic for the Qwen3-32B readout (HarmBench human_jailbreak + direct_request).
1. HF correctness: greedy next token from the full HF forward vs the first token vLLM generated (responses_hb.jsonl).
2. Activations at the last prompt token for layers L (decoder-layer outputs), and at layer 32 / 48 for the last 6 prompt
   positions; same chat text as qwen_acts.py (enable_thinking=False).
3. Same, with the generation prompt ending at "<|im_start|>assistant\n" (no empty think block).
Writes diag_acts.pt {"slots": [...], "pair_ids": [...], "X": fp16 [n, n_slots, 5120]} and diag_first_token.json."""
import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, resp, outdir = sys.argv[1:5]; test = "--test" in sys.argv
LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]; POS_LAYERS = [32, 48]; NPOS = 6; BUDGET = 32768
rows = [r for r in map(json.loads, open(inp)) if r["set"] == "hb"][: (16 if test else None)]
R = {r["pair_id"]: r for r in map(json.loads, open(resp))}
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
msgs = lambda r: [{"role": "user", "content": r["prompt"]}]
T_think = [tok.apply_chat_template(msgs(r), tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
T_plain = [tok.apply_chat_template(msgs(r), tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
print("tail think :", repr(T_think[0][-80:])); print("tail plain :", repr(T_plain[0][-80:]))
print("tail tokens think:", tok.convert_ids_to_tokens(tok(T_think[0], add_special_tokens=False).input_ids[-NPOS:]))
print("tail tokens plain:", tok.convert_ids_to_tokens(tok(T_plain[0], add_special_tokens=False).input_ids[-NPOS:]), flush=True)
cap = {}
def mk(l):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap[l] = h[:, -NPOS:].float().cpu()
    return hook
for l in sorted(set(LAYERS) | set(POS_LAYERS)): model.model.layers[l].register_forward_hook(mk(l))
def embed(texts, want_logits):
    L = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]; order = sorted(range(len(texts)), key=lambda i: L[i])
    out, top1, b = {}, {}, 0
    while b < len(order):
        e = b + 1
        while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 32: e += 1
        enc = tok([texts[i] for i in order[b:e]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad(): o = model(**enc, use_cache=False, logits_to_keep=1)
        for j, i in enumerate(order[b:e]):
            out[i] = {l: cap[l][j] for l in cap}
            if want_logits: top1[i] = int(o.logits[j, -1].argmax())
        b = e
    return out, top1
t0 = time.time(); A, top1 = embed(T_think, True); print(f"think: {time.time()-t0:.0f}s", flush=True)
t0 = time.time(); B, _ = embed(T_plain, False); print(f"plain: {time.time()-t0:.0f}s", flush=True)
slots, X = [], []
for l in LAYERS: slots.append(f"think_L{l}_pos-1")
for l in POS_LAYERS:
    for p in range(2, NPOS + 1): slots.append(f"think_L{l}_pos-{p}")
for l in LAYERS: slots.append(f"plain_L{l}_pos-1")
def vec(D, s):
    _, L, p = s.split("_"); return D[int(L[1:])][-int(p[4:])]
for i in range(len(rows)):
    X.append(torch.stack([vec(A[i] if s.startswith("think") else B[i], s) for s in slots]).half())
torch.save({"slots": slots, "pair_ids": [r["pair_id"] for r in rows], "X": torch.stack(X)}, f"{outdir}/diag_acts.pt.tmp")
os.replace(f"{outdir}/diag_acts.pt.tmp", f"{outdir}/diag_acts.pt")
agree, ex = 0, []
for i, r in enumerate(rows):
    v = tok(R[r["pair_id"]]["response_text"], add_special_tokens=False).input_ids
    ok = bool(v) and v[0] == top1[i]; agree += ok
    if len(ex) < 12 and not ok: ex.append({"pair_id": r["pair_id"], "hf": tok.decode([top1[i]]), "vllm": tok.decode(v[:1]) if v else ""})
res = {"n": len(rows), "first_token_agree": agree / len(rows), "disagree_examples": ex}
json.dump(res, open(f"{outdir}/diag_first_token.json", "w"), indent=1); print(res, flush=True); print("done", flush=True)
