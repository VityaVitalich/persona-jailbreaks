#!/usr/bin/env python3
"""Detection readout of Qwen3-32B for one set: output of decoder layer 32 (= hidden_states[33]; the Assistant Axis
vectors' row-32 convention, which hooks layer outputs) at the last prompt token, same chat text as qwen_gen.py. HF bf16,
sdpa, left padding, length-sorted batches under a token budget; forward stops after layer 32.
Writes {pair_id: {"32": fp16[5120]}} atomically. usage: qwen_acts.py MODEL inputs.jsonl SET OUT.pt [--test]"""
import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, st, out = sys.argv[1:5]; test = "--test" in sys.argv; LAYER, BUDGET = 32, 32768
rows = [r for r in map(json.loads, open(inp)) if r["set"] == st][: (4 if test else None)]
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
layers = model.model.layers; assert len(layers) == 64, len(layers)
T = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
L = [len(tok(t, add_special_tokens=False).input_ids) for t in T]
class Stop(Exception): pass
cap = {}
def hook(m, i, o): cap["h"] = (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu(); raise Stop
layers[LAYER].register_forward_hook(hook)
order = sorted(range(len(rows)), key=lambda i: L[i]); vals, t0, b = {}, time.time(), 0
while b < len(order):
    e = b + 1
    while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 64: e += 1
    enc = tok([T[i] for i in order[b:e]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    try:
        with torch.no_grad(): model(**enc, use_cache=False)
    except Stop: pass
    for i, v in zip(order[b:e], cap["h"]): vals[i] = v
    b = e
print(f"{st}: embedded {len(vals)} in {time.time()-t0:.0f}s; median norm {torch.stack(list(vals.values())).norm(dim=1).median():.1f}", flush=True)
torch.save({rows[i]["pair_id"]: {str(LAYER): vals[i].half()} for i in range(len(rows))}, out + ".tmp"); os.replace(out + ".tmp", out); print("done", flush=True)
