#!/usr/bin/env python3
"""
Cluster job (RCP runai, 1 GPU): Llama-3.1-8B-Instruct responses to the synthetic role-play set,
generated as in the paper (Appendix app:generation): official chat template, single user turn,
greedy, max 300 new tokens. Batched with left padding, prompts sorted by length.

--check rows (with the stored response text) are generated first and compared with Dominic's
stored greedy responses (exact match of the first 100 characters), to verify the setup.
Writes --out jsonl: {id, response_text}; resumable (skips ids already written).
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=48)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    assert "Cutting Knowledge Date" in tok.chat_template
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def gen(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=300, do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]

    checks = [json.loads(l) for l in open(a.check)]
    got = gen([c["prompt"] for c in checks])
    n_ok = sum(g[:100] == c["stored_response"][:100] for g, c in zip(got, checks))
    for g, c in zip(got, checks):
        print(f"check {c['set']} pid={c['pair_id']} match100={g[:100] == c['stored_response'][:100]} | {g[:70]!r} | {c['stored_response'][:70]!r}", flush=True)
    print(f"CHECK {n_ok}/{len(checks)} first-100-char matches", flush=True)

    rows = [json.loads(l) for l in open(a.inp)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = sorted([r for r in rows if r["id"] not in done], key=lambda r: len(r["prompt"]))
    print(f"{len(done)} done, {len(todo)} to generate", flush=True)
    with open(a.out, "a") as f:
        for i in range(0, len(todo), a.batch_size):
            b = todo[i:i + a.batch_size]
            for r, resp in zip(b, gen([r["prompt"] for r in b])):
                f.write(json.dumps({"id": r["id"], "response_text": resp}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i + len(b)}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
