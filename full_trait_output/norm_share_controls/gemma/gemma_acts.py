#!/usr/bin/env python3
"""Layer-30 last-token activations of Gemma-4-31B-it (the PersonaJB readout: hidden_states[31], position -1) for the
ordinary-text dictionaries and the role-play prompts, forward only (no generation).
Input rows {id, text, mode}: mode "chat" = single user turn with the chat template and add_generation_prompt (the
recipe of collect_harmbench_activations_gemma.py), mode "raw" = <bos> + the first 256 tokens, no template.
Model recipe = steer_fixed_direction_gemma.load_model (gqa_fold attention, left padding, chunked_sdpa_scope). The
forward stops after decoder layer 30 (its output is hidden_states[31]).
Writes {id: fp16[5376]}.   usage: python gemma_acts.py MODEL inputs.jsonl out.pt [token_budget]"""
import json, sys, time, torch
sys.path.insert(0, "/mnt/dlabscratch1/moskvore/persona_steer/full_trait_tools")
sys.path.insert(0, "/mnt/dlabscratch1/moskvore/persona_steer/jailbreaking_tools")
from chunked_sdpa import chunked_sdpa_scope
from steer_fixed_direction_gemma import load_model, find_decoder_layers, sanitize
model_path, inp, out = sys.argv[1:4]; BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 32768
model, tok = load_model(model_path, "cuda:0")
layers = find_decoder_layers(model)
rows = [json.loads(l) for l in open(inp)]
def text(r):
    if r["mode"] == "raw":
        return tok.bos_token + tok.decode(tok(r["text"], add_special_tokens=False).input_ids[:256])
    return tok.apply_chat_template([{"role": "user", "content": sanitize(r["text"])}], tokenize=False, add_generation_prompt=True)
T = [text(r) for r in rows]; L = [len(tok(t, add_special_tokens=False).input_ids) for t in T]
print("example chat:", repr(T[next(i for i, r in enumerate(rows) if r["mode"] == "chat")][:300]), flush=True)
print("example raw:", repr(T[next(i for i, r in enumerate(rows) if r["mode"] == "raw")][:120]), flush=True)
class Stop(Exception): pass
cap = {}
def hook(m, i, o):
    cap["h"] = (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu(); raise Stop
layers[30].register_forward_hook(hook)
order = sorted(range(len(rows)), key=lambda i: L[i]); res, t0, b = {}, time.time(), 0
while b < len(order):
    e = b + 1
    while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 128: e += 1
    idx = order[b:e]
    enc = tok([T[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    try:
        with torch.no_grad(), chunked_sdpa_scope(): model(**enc, use_cache=False)
    except Stop: pass
    for i, v in zip(idx, cap["h"]): res[rows[i]["id"]] = v.half()
    b = e
    if len(res) % 2000 < len(idx): print(f"{len(res)}/{len(rows)} {time.time()-t0:.0f}s maxlen {L[idx[-1]]}", flush=True)
torch.save(res, out); print("done", len(res), f"{time.time()-t0:.0f}s", flush=True)
