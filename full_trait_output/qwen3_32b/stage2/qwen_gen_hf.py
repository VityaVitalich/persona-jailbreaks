#!/usr/bin/env python3
"""Fallback generator (HF transformers, as the Gemma collectors): greedy, max_new_tokens 300, bf16, sdpa, left-padded
length-sorted batches under a token budget; same chat text as qwen_gen.py (enable_thinking=False). Same output schema;
atomic write; partial progress kept in OUT.partial.jsonl so a retried pod resumes. usage: qwen_gen_hf.py MODEL inputs.jsonl SET OUT [--test]"""
import json, os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, st, out = sys.argv[1:5]; test = "--test" in sys.argv; BUDGET, MAXB, NEW = 98304, 96, 300
rows = [r for r in map(json.loads, open(inp)) if r["set"] == st][: (4 if test else None)]
tok = AutoTokenizer.from_pretrained(model_path); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda:0"}).eval()
T = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
L = [len(tok(t, add_special_tokens=False).input_ids) for t in T]
part = out + ".partial.jsonl"; done = {json.loads(l)["pair_id"]: json.loads(l) for l in open(part)} if os.path.exists(part) else {}
todo = sorted((i for i in range(len(rows)) if rows[i]["pair_id"] not in done), key=lambda i: L[i]); t0 = time.time(); b = 0
print(f"{st}: {len(rows)} prompts, {len(done)} already done", flush=True)
with open(part, "a") as f:
    while b < len(todo):
        e = b + 1
        while e < len(todo) and (e - b + 1) * (L[todo[e]] + NEW) <= BUDGET and e - b < MAXB: e += 1
        idx = todo[b:e]; enc = tok([T[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, g[:, enc.input_ids.shape[1]:]):
            n = int((seq != tok.pad_token_id).sum()); x = {"set": st, "pair_id": rows[i]["pair_id"], "response_text": tok.decode(seq, skip_special_tokens=True),
                 "n_prompt_tokens": L[i], "finish_reason": "length" if n >= NEW else "stop", "backend": "hf"}
            f.write(json.dumps(x, ensure_ascii=False) + "\n"); done[x["pair_id"]] = x
        f.flush(); b = e
        print(f"  {len(done)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
with open(out + ".tmp", "w") as f:
    for r in rows: f.write(json.dumps(done[r["pair_id"]], ensure_ascii=False) + "\n")
os.replace(out + ".tmp", out); print("done", flush=True)
