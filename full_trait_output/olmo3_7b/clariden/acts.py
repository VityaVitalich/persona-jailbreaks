#!/usr/bin/env python3
"""OLMo-3-7B-Instruct readout (serving image, transformers 4.57) at decoder-layer outputs LAYERS for the last NPOS prompt
positions (the chat-template tail <|im_end|> \\n <|im_start|> assistant \\n), same chat text as gen.py. HF bf16, sdpa,
left padding, token-budget batches; forward stops after the last hooked layer.
Writes acts_{set}.pt {"pair_ids", "layers", "tail_tokens", "X": fp16 [n, n_layers, NPOS, 4096]}.
usage: acts.py MODEL inputs.jsonl OUTDIR SET[,SET..] [--test]"""
import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, outdir = sys.argv[1:4]; test = "--test" in sys.argv; SETS = sys.argv[4].split(",")
LAYERS, NPOS, BUDGET = [4, 8, 12, 16, 20, 24, 28, 31], 5, 65536
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()  # serving image: no accelerate, so no device_map
class Stop(Exception): pass
cap = {}
def mk(l):
    def hook(m, i, o):
        cap[l] = (o[0] if isinstance(o, tuple) else o)[:, -NPOS:].float().cpu()
        if l == max(LAYERS): raise Stop
    return hook
for l in LAYERS: model.model.layers[l].register_forward_hook(mk(l))
allrows = [json.loads(l) for l in open(inp)]
for st in SETS:
    out = f"{outdir}/acts_{st}.pt"
    if os.path.exists(out): print(st, "exists, skip"); continue
    rows = [r for r in allrows if r["set"] == st][: (8 if test else None)]
    T = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in rows]
    ids0 = tok(T[0], add_special_tokens=False).input_ids; tail = tok.convert_ids_to_tokens(ids0[-NPOS:]); print(st, "tail tokens", tail, flush=True)
    L = [len(tok(t, add_special_tokens=False).input_ids) for t in T]; order = sorted(range(len(T)), key=lambda i: L[i]); vals, b, t0 = {}, 0, time.time()
    while b < len(order):
        e = b + 1
        while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 128: e += 1
        enc = tok([T[i] for i in order[b:e]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        try:
            with torch.no_grad(): model(**enc, use_cache=False)
        except Stop: pass
        for j, i in enumerate(order[b:e]): vals[i] = torch.stack([cap[l][j] for l in LAYERS]).half()
        b = e
    torch.save({"pair_ids": [r["pair_id"] for r in rows], "layers": LAYERS, "tail_tokens": tail, "X": torch.stack([vals[i] for i in range(len(rows))])}, out + ".tmp")
    os.replace(out + ".tmp", out); print(f"{st}: {len(rows)} in {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
