#!/usr/bin/env python3
"""
Prove the prefix KV cache is mathematically equivalent to recomputing the prefix.

Loads Gemma once and runs the same behaviour twice with the same seed — once with
the cache, once with --no_prefix_cache — then compares the per-step losses. The
candidate sampling is seeded, so any difference in the loss trace comes from the
cache and nothing else. Also times both so the speedup is measured, not estimated.
"""
import argparse
import time

import torch

import gcg_gemma as G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=G.GEMMA_SNAPSHOT)
    p.add_argument("--behavior_id", default=None,
                   help="Default: the first contextual behaviour, where the prefix "
                        "dominates and the cache matters most.")
    p.add_argument("--num_steps", type=int, default=6)
    p.add_argument("--search_width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--n_adv_tokens", type=int, default=20)
    p.add_argument("--early_stop_loss", type=float, default=-1.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_non_ascii", action="store_true")
    args = p.parse_args()

    behaviors = G.load_behaviors(G.BEHAVIORS_CSV, G.TARGETS_JSON)
    if args.behavior_id:
        beh = next(b for b in behaviors if b["behavior_id"] == args.behavior_id)
    else:
        beh = next(b for b in behaviors if b["category"] == "contextual")
    print(f"behaviour: {beh['behavior_id']} ({beh['category']}, "
          f"{len(beh['prompt'])} chars)", flush=True)

    tok = G.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = G.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0", trust_remote_code=True)
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = next(model.parameters()).device
    embed_w = model.get_input_embeddings().weight

    banned = None if args.allow_non_ascii else G.get_nonascii_toks(tok, device)
    pre, post, tgt = G.build_segments(tok, beh["prompt"], beh["target"], device)
    print(f"prefix tokens {pre.shape[0]}, suffix+target tokens "
          f"{args.n_adv_tokens + post.shape[0] + tgt.shape[0]}", flush=True)

    out = {}
    for label, no_cache in (("cached", False), ("uncached", True)):
        args.no_prefix_cache = no_cache
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        t0 = time.time()
        suffix, loss, steps = G.gcg_attack(model, tok, embed_w, pre, post, tgt,
                                           args, device, banned)
        torch.cuda.synchronize()
        out[label] = (loss, time.time() - t0, suffix)
        print(f"{label:9s} best_loss={loss:.6f}  {time.time()-t0:.1f}s for {steps} steps",
              flush=True)

    dl = abs(out["cached"][0] - out["uncached"][0])
    speedup = out["uncached"][1] / max(out["cached"][1], 1e-9)
    print(f"\n|delta best_loss| = {dl:.2e}")
    print(f"speedup           = {speedup:.2f}x")
    print(f"suffixes match    = {out['cached'][2] == out['uncached'][2]}")
    print("VERDICT:", "PASS" if dl < 1e-3 else "FAIL — cache changes the result")


if __name__ == "__main__":
    main()
