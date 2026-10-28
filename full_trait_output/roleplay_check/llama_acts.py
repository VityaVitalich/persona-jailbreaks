#!/usr/bin/env python3
"""
Llama-3.1-8B-Instruct L16 pre-generation activation (last prompt token), the detector's readout,
computed on the Mac (MPS, bf16). Recipe = collect_harmbench_activations.process_pair:
official chat template (system 'Cutting Knowledge Date' header; unsloth tokenizer, NousResearch
weights = byte-identical to meta-llama), single user turn, add_generation_prompt, hidden_states[17][0,-1].

  python llama_acts.py verify            # recompute a sample of saved rows, cosine vs stored
  python llama_acts.py embed IN.jsonl OUT.pt   # rows with 'id' and 'prompt' -> {id: [4096] float32}
"""
import json, os, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = Path.home() / ".cache/huggingface/hub"
snap = lambda r: next((HUB / f"models--{r}/snapshots").iterdir())
OUT = Path.home() / "persona-vector-jbing/full_trait_output"
HERE = OUT / "roleplay_check"
LAYER = 16


def load():
    tok = AutoTokenizer.from_pretrained(snap("unsloth--Meta-Llama-3.1-8B-Instruct"))
    assert "Cutting Knowledge Date" in tok.chat_template
    model = AutoModelForCausalLM.from_pretrained(snap("NousResearch--Meta-Llama-3.1-8B-Instruct"),
                                                 torch_dtype=torch.bfloat16).to("mps").eval()
    return tok, model


@torch.no_grad()
def act(tok, model, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
    hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[LAYER + 1]
    return hs[0, -1].float().cpu()


def verify():
    rows = [json.loads(l) for l in open(HERE / "prompts.jsonl")]
    src = {"HB": "harmbench_activations", "GCG": "gcg_activations", "PAIR": "pair_activations", "PAP": "pap_activations",
           "GPTFuzz": "gptfuzz_activations", "PEZ": "pez_activations", "WJB": "wildjailbreak_activations"}
    rng = random.Random(0)
    tok, model = load()
    for s, d in src.items():
        stored = torch.load(OUT / d / "activations.pt", map_location="cpu", weights_only=False)
        pool = [r for r in rows if r["set"] == s]
        picks = rng.sample(pool, 6)
        if s == "PAIR":
            picks = [r for r in pool if r["attack_type"] == "PAIR"][:3] + [r for r in pool if r["attack_type"] == "plain"][:3]
        for r in picks:
            a, b = act(tok, model, r["prompt"]), stored[r["pair_id"]][str(LAYER)].float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            print(f"{s:8s} pid={r['pair_id']:5d} {r['attack_type']:16s} cos={cos:.4f} |a|={a.norm():.2f} |b|={b.norm():.2f}", flush=True)


def embed(inp, outp):
    rows = [json.loads(l) for l in open(inp)]
    res = torch.load(outp) if os.path.exists(outp) else {}
    tok, model = load()
    for i, r in enumerate(rows):
        if r["id"] in res: continue
        res[r["id"]] = act(tok, model, r["prompt"])
        if i % 50 == 0:
            torch.save(res, outp); print(f"{i}/{len(rows)}", flush=True)
    torch.save(res, outp); print("done", len(res), flush=True)


if __name__ == "__main__":
    verify() if sys.argv[1] == "verify" else embed(sys.argv[2], sys.argv[3])
