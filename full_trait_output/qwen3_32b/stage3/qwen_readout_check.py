#!/usr/bin/env python3
"""Readout check: re-embed 8 prompts per collected set one at a time (no padding, fresh model load) and compare with the
stored fp16 activations (cosine, norm ratio). Writes readout_check.json. usage: qwen_readout_check.py MODEL inputs.jsonl OUTDIR [SETS comma-separated] [OUTFILE]"""
import json, os, random, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, od = sys.argv[1:4]; SETS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["hb", "wjb", "pap"]; OUTF = sys.argv[5] if len(sys.argv) > 5 else "readout_check.json"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
cap = {}
model.model.layers[32].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
rows = list(map(json.loads, open(inp))); res = {}
for st in SETS:
    f = f"{od}/acts_{st}.pt"
    if not os.path.exists(f): continue
    A = torch.load(f, map_location="cpu"); pool = [r for r in rows if r["set"] == st and r["pair_id"] in A]
    out = []
    for r in random.Random(0).sample(pool, min(8, len(pool))):
        t = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        with torch.no_grad(): model(**tok(t, return_tensors="pt", add_special_tokens=False).to("cuda:0"), use_cache=False)
        s = A[r["pair_id"]]["32"].float(); out.append(dict(pair_id=r["pair_id"], cos=float(torch.nn.functional.cosine_similarity(cap["h"], s, dim=0)), norm_ratio=float(cap["h"].norm() / s.norm())))
    res[st] = dict(n=len(out), min_cos=min(o["cos"] for o in out), median_cos=sorted(o["cos"] for o in out)[len(out)//2], rows=out)
    print(st, {k: v for k, v in res[st].items() if k != "rows"}, flush=True)
json.dump(res, open(f"{od}/{OUTF}", "w"), indent=1)
