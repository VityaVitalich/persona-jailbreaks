#!/usr/bin/env python3
"""Detection-prompt activations at many layers x the last NPOS prompt positions, for Llama-3.1-8B-Instruct and
Gemma-4-31B-it, reproducing the paper collectors (full_trait_tools/collect_*_activations*.py): prompt -> sanitize_text ->
one user turn, no system prompt, add_generation_prompt=True, encoded with add_special_tokens=False; one prompt per forward
(no padding); layer L = hidden_states[L + 1]. Gemma: AutoModelForCausalLM, trust_remote_code, sdpa, under the collectors'
chunked_sdpa_scope (head_dim 512). Shard SHARD of NSHARD takes rows i % NSHARD == SHARD of every set.
Writes OUTDIR/shards/{set}_L{L}.s{SHARD}.npy (fp16 [n_shard, NPOS, d]) + {set}.s{SHARD}.json (row indices, pair_ids, tail tokens).
usage: acts_multi.py MODEL INPUTS_JSONL OUTDIR SETS LAYERS SHARD NSHARD [--gemma] [--test]
merge: acts_multi.py --merge OUTDIR SETS LAYERS NSHARD   ->  OUTDIR/{set}_L{L}.npy + {set}_meta.json (hb_select.py format)"""
import json, os, sys, time, numpy as np
NPOS = 5
def sanitize_text(text):
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
if sys.argv[1] == "--merge":
    out, sets, layers, n = sys.argv[2], sys.argv[3].split(","), [int(x) for x in sys.argv[4].split(",")], int(sys.argv[5])
    for st in sets:
        M = [json.load(open(f"{out}/shards/{st}.s{k}.json")) for k in range(n)]; idx = np.concatenate([m["rows"] for m in M]); order = np.argsort(idx)
        assert (idx[order] == np.arange(len(idx))).all(), st
        for L in layers: np.save(f"{out}/{st}_L{L}.npy", np.concatenate([np.load(f"{out}/shards/{st}_L{L}.s{k}.npy") for k in range(n)])[order])
        pid = [p for m in M for p in m["pair_ids"]]
        json.dump({"pair_ids": [pid[k] for k in order], "tail_tokens": M[0]["tail_tokens"], "layers": layers}, open(f"{out}/{st}_meta.json", "w"))
        print("merged", st, len(idx), flush=True)
    sys.exit()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id, inp, out, sets, layers, shard, nshard = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), [int(x) for x in sys.argv[5].split(",")], int(sys.argv[6]), int(sys.argv[7])
gemma, test = "--gemma" in sys.argv, "--test" in sys.argv
os.makedirs(f"{out}/shards", exist_ok=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=gemma)
kw = dict(trust_remote_code=True, attn_implementation="sdpa") if gemma else {}
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw).to("cuda:0").eval()
if gemma:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from chunked_sdpa import chunked_sdpa_scope
else:
    from contextlib import nullcontext as chunked_sdpa_scope
rows_all = [json.loads(l) for l in open(inp)]
for st in sets:
    rows = [r for r in rows_all if r["set"] == st]; sel = [i for i in range(len(rows)) if i % nshard == shard][: (4 if test else None)]
    acc = {L: np.zeros((len(sel), NPOS, model.config.get_text_config().hidden_size), np.float16) for L in layers}; t0 = time.time(); tail = None
    for j, i in enumerate(sel):
        text = tok.apply_chat_template([{"role": "user", "content": sanitize_text(rows[i]["prompt"])}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
        if tail is None: tail = tok.convert_ids_to_tokens(ids[0, -NPOS:].tolist())
        with torch.no_grad(), chunked_sdpa_scope():
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers: acc[L][j] = hs[L + 1][0, -NPOS:].float().cpu().numpy()
        del hs
    for L in layers: np.save(f"{out}/shards/{st}_L{L}.s{shard}.npy", acc[L])
    json.dump({"rows": sel, "pair_ids": [rows[i]["pair_id"] for i in sel], "tail_tokens": tail}, open(f"{out}/shards/{st}.s{shard}.json", "w"))
    print(f"{st} shard {shard}: {len(sel)} rows in {time.time() - t0:.0f}s, tail {tail}", flush=True)
print("done", flush=True)
