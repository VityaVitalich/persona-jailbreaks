#!/usr/bin/env python3
"""L16 activations of Wikipedia passages (WikiText-103, 2000 random paragraphs >= 600 chars, detokenized),
Llama-3.1-8B-Instruct official snapshot, last token, hidden_states[17] (= the readout of every other set).
Two readings: raw   - plain text, BOS + first 256 tokens, no chat template (pretraining-like, no request)
              chat  - the same 256-token text as a single user turn with the official chat template
Writes wiki_acts.pt {"raw": {id: fp16[4096]}, "chat": {...}}."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path, inp, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_path); assert "Cutting Knowledge Date" in tok.chat_template
tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
rows = [json.loads(l) for l in open(inp)]
cut = [tok.decode(tok(r["text"], add_special_tokens=False).input_ids[:256]) for r in rows]
texts = {"raw": [tok.bos_token + c for c in cut],
         "chat": [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True) for c in cut]}
res = {}
for mode, T in texts.items():
    res[mode] = {}
    for i in range(0, len(T), 32):
        enc = tok(T[i:i+32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[17]
        last = enc.attention_mask.sum(1) - 1
        for r, v in zip(rows[i:i+32], hs[torch.arange(len(last)), last].float().cpu()): res[mode][r["id"]] = v.half()
    print(mode, len(res[mode]), flush=True)
torch.save(res, out); print("done", flush=True)
