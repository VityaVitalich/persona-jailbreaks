#!/usr/bin/env python3
"""Time Gemma-4-31B detection baselines per-request (batch=1, CUDA-synced),
mirroring full_trait_tools/time_baselines_gpu.py. GradSafe is NOT timed here
(needs multi-GPU sharding); its per-prompt latency is extracted from the
finished gradsafe_gemma checkpoint JSONLs instead."""
import argparse, json, statistics, sys, time
from pathlib import Path
import os
import torch

BASE = next(c for c in ("/dlabscratch1", "/mnt/dlabscratch1",
                        "/mnt/dlab/scratch/dlabscratch1")
            if Path(c, "bazina").is_dir())
PROJ = f"{BASE}/bazina/assistant-axis-llama3.1-8B"
sys.path.insert(0, f"{PROJ}/full_trait_tools")
sys.path.insert(0, f"{BASE}/moskvore/ci_rescore")
os.chdir(PROJ)

import run_baselines_harmbench_ppl_smoothllm as PPL
import run_baselines_self_exam as SE
import run_baselines_llama_guard as LG

GEMMA = (f"{BASE}/bazina/.cache/huggingface/hub/"
         "models--google--gemma-4-31B-it/snapshots/"
         "fb9ae262347c3945692f09a612f8bb189def854f")
CLS = f"{PROJ}/full_trait_output/harmbench_activations_gemma/classified_responses.jsonl"


def load_pairs(n):
    pairs = []
    with open(CLS) as f:
        for line in f:
            r = json.loads(line)
            p = (r.get("behavior_text") or "").strip()
            resp = (r.get("response_text") or r.get("response") or "").strip()
            if p and resp:
                pairs.append((p, resp))
            if len(pairs) >= n:
                break
    return pairs


def time_batch1(fn, items, warmup=3):
    for it in items[:warmup]:
        fn(it)
    torch.cuda.synchronize()
    ts = []
    for it in items:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(it)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts), (statistics.pstdev(ts) if len(ts) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = load_pairs(args.n)
    dev = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)} N={len(pairs)}", flush=True)
    res = {}

    # Gemma-4 head_dim=512: keep every forward inside the chunked-SDPA scope,
    # as the scoring runs did.
    try:
        from chunked_sdpa import chunked_sdpa_scope
        chunked_sdpa_scope().__enter__()
        print("chunked_sdpa scope active", flush=True)
    except Exception as e:
        print(f"chunked_sdpa not loaded ({e}); relying on plain sdpa", flush=True)

    print(f"Loading Gemma-4-31B from {GEMMA} ...", flush=True)
    model, tok, _ = PPL.load_causal_lm(GEMMA, "cuda", gemma=True)

    m, s = time_batch1(lambda pr: PPL.prompt_perplexity(model, tok, pr[0], dev, 2048), pairs)
    res["ppl"] = {"mean_ms": m, "std_ms": s}
    print(f"ppl        {m:9.2f} ± {s:7.2f} ms", flush=True)

    m, s = time_batch1(lambda pr: SE.generate_meta_response(model, tok, pr[0], SE.DIRECT_PROMPT, dev), pairs)
    res["verb_nocot"] = {"mean_ms": m, "std_ms": s}
    print(f"verb_nocot {m:9.2f} ± {s:7.2f} ms", flush=True)

    m, s = time_batch1(lambda pr: SE.generate_meta_response(model, tok, pr[0], SE.COT_PROMPT, dev), pairs)
    res["verb_cot"] = {"mean_ms": m, "std_ms": s}
    print(f"verb_cot   {m:9.2f} ± {s:7.2f} ms", flush=True)

    del model
    torch.cuda.empty_cache()

    print(f"Loading {LG.GUARD_MODEL} ...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    gtok = AutoTokenizer.from_pretrained(LG.GUARD_MODEL)
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    gmodel = AutoModelForCausalLM.from_pretrained(LG.GUARD_MODEL, dtype=torch.bfloat16,
                                                  device_map={"": dev}).eval()
    safe_id, unsafe_id = LG.get_token_ids(gtok)
    m, s = time_batch1(lambda pr: LG.guard_score(gmodel, gtok, pr[0], dev,
                                                 safe_id, unsafe_id, response=None), pairs)
    res["llama_guard_input"] = {"mean_ms": m, "std_ms": s}
    print(f"guard_in   {m:9.2f} ± {s:7.2f} ms", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
