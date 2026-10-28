#!/usr/bin/env python3
"""
Cluster job (RCP runai, 1 GPU): Llama-3.1-8B-Instruct L16 pre-generation activation (last prompt
token) for every prompt in --inp. Recipe = collect_harmbench_activations.process_pair: official
meta-llama weights + chat template (system 'Cutting Knowledge Date' header), single user turn,
add_generation_prompt, hidden_states[LAYER+1][0, -1].

--check rows carry 'stored' = (activation dir, pair_id): their recomputed activation is compared
with Dominic's stored one (cosine), so the pipeline is verified on the same run.

Writes --out = {id: float16 [4096]} plus a per-row .jsonl log of the check cosines.
"""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--stored_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    assert "Cutting Knowledge Date" in tok.chat_template, "not the official Llama-3.1 template"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def act(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        return model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1][0, -1].float().cpu()

    stored_cache, cos_log = {}, []
    for r in map(json.loads, open(a.check)):
        d, pid = r["stored"]
        if d not in stored_cache:
            stored_cache[d] = torch.load(Path(a.stored_root) / d / "activations.pt", map_location="cpu", weights_only=False)
        b = stored_cache[d][pid][str(LAYER)].float()
        c = torch.nn.functional.cosine_similarity(act(r["prompt"]), b, dim=0).item()
        cos_log.append({"set": r["set"], "pair_id": pid, "cos": c})
        print(f"check {r['set']:8s} pid={pid:5d} cos={c:.4f}", flush=True)
    with open(Path(a.out).with_suffix(".check.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in cos_log)
    print(f"CHECK min cos = {min(x['cos'] for x in cos_log):.4f}", flush=True)

    rows = [json.loads(l) for l in open(a.inp)]
    res = {}
    for i, r in enumerate(rows):
        res[r["id"]] = act(r["prompt"]).half()
        if i % 500 == 0: print(f"{i}/{len(rows)}", flush=True)
    torch.save(res, a.out)
    print("done", len(res), flush=True)


if __name__ == "__main__":
    main()
