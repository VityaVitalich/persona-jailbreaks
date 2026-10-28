#!/usr/bin/env python3
"""Stage 4: layer-32 last-token activations of Qwen3-32B for the control-dictionary inputs (the Gemma set
gemma_acts_inputs.jsonl: 2000 Wikipedia paragraphs read as raw text, 2000 No Robots requests and 12,558 role-play
prompts as a user turn, 24 HarmBench verify rows). chat = chat template, enable_thinking=False, add_generation_prompt;
raw = the first 256 tokens of the text, no template (Qwen has no BOS token). Writes extra_acts.pt {id: fp16[5120]}.
usage: python qwen_extra_acts.py MODEL gemma_acts_inputs.jsonl OUT.pt"""
import json, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, out = sys.argv[1:4]; BUDGET = 32768
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
rows = [json.loads(l) for l in open(inp)]
def text(r):
    if r["mode"] == "raw": return (tok.bos_token or "") + tok.decode(tok(r["text"], add_special_tokens=False).input_ids[:256])
    return tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
T = [text(r) for r in rows]; L = [len(tok(t, add_special_tokens=False).input_ids) for t in T]
class Stop(Exception): pass
cap = {}
def hook(m, i, o): cap["h"] = (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu(); raise Stop
model.model.layers[32].register_forward_hook(hook)
order = sorted(range(len(rows)), key=lambda i: L[i]); res, t0, b = {}, time.time(), 0
while b < len(order):
    e = b + 1
    while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 128: e += 1
    enc = tok([T[i] for i in order[b:e]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    try:
        with torch.no_grad(): model(**enc, use_cache=False)
    except Stop: pass
    for i, v in zip(order[b:e], cap["h"]): res[rows[i]["id"]] = v.half()
    b = e
torch.save(res, out); print("extra acts done", len(res), f"{time.time()-t0:.0f}s", flush=True)
