#!/usr/bin/env python3
"""Stage 2: the detection readout of Qwen3-32B: output of decoder layer 32 (= hidden_states[33], the row-32
convention of the Assistant Axis vectors, which hook layer outputs) at the last prompt token, same chat text as
qwen_gen.py (enable_thinking=False, add_generation_prompt). HF transformers, bf16, sdpa, left padding, length-sorted
batches under a token budget; the forward stops after layer 32. Readout check: 24 prompts re-embedded one at a time
(no padding), cosine with the batched value printed and saved. Writes acts_{set}.pt {pair_id: {"32": fp16[5120]}}.
usage: python qwen_acts.py MODEL inputs.jsonl OUTDIR [--test]"""
import json, sys, time, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, outdir = sys.argv[1:4]; test = "--test" in sys.argv; LAYER = 32; BUDGET = 32768
rows = [json.loads(l) for l in open(inp)]
if test:
    import collections; c = collections.Counter(); rows = [r for r in rows if c.update([r["set"]]) or c[r["set"]] <= 4]
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
layers = model.model.layers; assert len(layers) == 64, len(layers)
T = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
L = [len(tok(t, add_special_tokens=False).input_ids) for t in T]
class Stop(Exception): pass
cap = {}
def hook(m, i, o): cap["h"] = (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu(); raise Stop
layers[LAYER].register_forward_hook(hook)
def run(texts):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    try:
        with torch.no_grad(): model(**enc, use_cache=False)
    except Stop: pass
    return cap["h"]
order = sorted(range(len(rows)), key=lambda i: L[i]); vals, t0, b = {}, time.time(), 0
while b < len(order):
    e = b + 1
    while e < len(order) and (e - b + 1) * L[order[e]] <= BUDGET and e - b < 64: e += 1
    for i, v in zip(order[b:e], run([T[i] for i in order[b:e]])): vals[i] = v
    b = e
    if len(vals) % 1000 < (e - b + 64): pass
print(f"embedded {len(vals)} in {time.time()-t0:.0f}s", flush=True)
chk = random.Random(0).sample(range(len(rows)), min(24, len(rows)))
cos = [float(torch.nn.functional.cosine_similarity(run([T[i]])[0], vals[i], dim=0)) for i in chk]
print(f"readout check (batched vs single, {len(cos)} prompts): min cos {min(cos):.5f}; median norm {torch.stack(list(vals.values())).norm(dim=1).median():.1f}", flush=True)
json.dump({"cos": cos, "idx": chk}, open(f"{outdir}/readout_check.json", "w"))
for s in sorted({r["set"] for r in rows}):
    torch.save({rows[i]["pair_id"]: {str(LAYER): vals[i].half()} for i in range(len(rows)) if rows[i]["set"] == s}, f"{outdir}/acts_{s}.pt")
print("done", flush=True)
