#!/usr/bin/env python3
"""
Cluster job for the v2 synthetic set (RCP runai, 1 GPU), two stages run as separate processes:

  gen   vLLM greedy generation, max 300 new tokens (paper setting). Prompts are rendered with the
        official Llama-3.1 chat template (system 'Cutting Knowledge Date' header) and passed as
        token ids, so vLLM cannot substitute another template.  -> --out jsonl {id, response_text}
  acts  L16 pre-generation activation at the last prompt token (the detector readout), HF
        transformers, batched with RIGHT padding (causal mask -> real positions are unaffected).
        First recomputes --check rows and prints cosine vs Dominic's stored activations.
        -> --out .pt {id: float16 [4096]}
"""
import argparse, json
from pathlib import Path

LAYER = 16


def render(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def gen(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model); assert "Cutting Knowledge Date" in tok.chat_template
    rows = [json.loads(l) for l in open(a.inp)]
    ids = [tok(render(tok, r["prompt"]), add_special_tokens=False).input_ids for r in rows]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=4096, seed=0, gpu_memory_utilization=0.9)
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(temperature=0.0, max_tokens=300, seed=0))
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "response_text": o.outputs[0].text}, ensure_ascii=False) + "\n")
    print("gen done", len(rows), flush=True)


def acts(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); assert "Cutting Knowledge Date" in tok.chat_template
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def batch_acts(prompts):
        enc = tok([render(tok, p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1]
        last = enc.attention_mask.sum(1) - 1
        return hs[torch.arange(len(prompts)), last].float().cpu()

    stored = {}
    cos = []
    checks = [json.loads(l) for l in open(a.check)]
    got = batch_acts([c["prompt"] for c in checks])
    for c, v in zip(checks, got):
        d, pid = c["stored"]
        if d not in stored: stored[d] = torch.load(Path(a.stored_root) / d / "activations.pt", map_location="cpu", weights_only=False)
        cos.append(torch.nn.functional.cosine_similarity(v, stored[d][pid][str(LAYER)].float(), dim=0).item())
    print(f"CHECK cosine vs stored: min {min(cos):.4f} mean {sum(cos)/len(cos):.4f} (n={len(cos)})", flush=True)

    rows = sorted([json.loads(l) for l in open(a.inp)], key=lambda r: len(r["prompt"]))
    res = {}
    for i in range(0, len(rows), a.batch_size):
        b = rows[i:i + a.batch_size]
        for r, v in zip(b, batch_acts([r["prompt"] for r in b])): res[r["id"]] = v.half()
        if (i // a.batch_size) % 20 == 0: print(f"acts {i + len(b)}/{len(rows)}", flush=True)
    torch.save(res, a.out); print("acts done", len(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "acts"])
    ap.add_argument("--model", required=True); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check"); ap.add_argument("--stored_root"); ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()
    gen(a) if a.stage == "gen" else acts(a)
