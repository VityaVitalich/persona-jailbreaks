#!/usr/bin/env python3
"""
gcg_gemma.py

GCG adversarial suffix optimization against Gemma-4-31B-it.

HarmBench's own GCG cannot be used here: its venv pins transformers 4.45.2, which does
not recognize the `gemma4` architecture. This is the same algorithm (Zou et al., 2023,
as implemented in HarmBench/baselines/gcg) run through the project's .venv_gemma, using
HarmBench's behaviours and optimizer targets so the resulting test cases line up with
the other Gemma attack families.

Usage:
  .venv_gemma/bin/python full_trait_tools/gcg_gemma.py \\
      --output_dir full_trait_output/gcg_gemma_test_cases \\
      --num_shards 20 --shard_idx 0
"""
import argparse
import csv
import gc
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GEMMA_SNAPSHOT = ("/mnt/dlabscratch1/bazina/.cache/huggingface/hub/"
                  "models--google--gemma-4-31B-it/snapshots/"
                  "fb9ae262347c3945692f09a612f8bb189def854f")
BEHAVIORS_CSV = "/mnt/dlabscratch1/bazina/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
TARGETS_JSON = "/mnt/dlabscratch1/bazina/HarmBench/data/optimizer_targets/harmbench_targets_text.json"
MARKER = "<|__adv__|>"


def load_behaviors(csv_path, targets_path):
    targets = json.load(open(targets_path))
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            bid = row["BehaviorID"]
            if bid not in targets:
                continue
            prompt = row["Behavior"]
            if row.get("ContextString"):
                prompt = f"{row['ContextString']}\n\n---\n\n{prompt}"
            rows.append({"behavior_id": bid, "behavior": row["Behavior"],
                         "prompt": prompt, "target": targets[bid],
                         "category": row["FunctionalCategory"]})
    return rows


def build_segments(tokenizer, prompt, target, device):
    """Token ids before / after the adversarial suffix, plus the target ids."""
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{prompt} {MARKER}"}],
        tokenize=False, add_generation_prompt=True)
    pre_text, post_text = templated.split(MARKER)
    enc = lambda t: tokenizer(t, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    return enc(pre_text)[0], enc(post_text)[0], enc(target)[0]


def get_nonascii_toks(tokenizer, device):
    """Token ids HarmBench's GCG forbids in the suffix (allow_non_ascii: False).

    Mirrors HarmBench/baselines/gcg/gcg_utils.get_nonascii_toks so the Gemma
    suffixes are searched over the same restricted token set as the Llama ones;
    without it the search space here is strictly larger than the published
    Llama run's and the two are not comparable.
    """
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    bad = [i for i in range(3, tokenizer.vocab_size)
           if not is_ascii(tokenizer.decode([i]))]
    for tid in (tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id, tokenizer.unk_token_id):
        if tid is not None:
            bad.append(tid)
    return torch.unique(torch.tensor(bad, device=device))


def target_loss(model, embeds, target_ids, norm, cache=None, n_prefix=0):
    """Cross-entropy of the target continuation, batched over dim 0.

    `embeds` holds raw rows of the embedding matrix. Gemma applies its
    sqrt(hidden_size) normalizer only when it embeds input_ids itself, so a
    hand-built inputs_embeds has to be scaled here — unscaled, the model sees
    noise and the target loss sits at ~20 instead of ~1.4.

    With `cache`, `embeds` holds only the tokens after the prompt prefix and the
    prefix's keys/values are read from the cache. The prefix never changes within
    a behaviour, so recomputing it for all 512 candidates at every step is pure
    waste — on the contextual behaviours it is ~86% of the sequence. The cache is
    cropped back to the prefix afterwards so the caller can reuse it.
    """
    n_tgt = target_ids.shape[-1]
    # Only the last n_tgt+1 positions are read below. Without this the model
    # materializes batch x seq x 256k logits, which OOMs on the contextual
    # behaviours (long ContextString) at large --batch_size.
    kw = {}
    if cache is not None:
        n_new = embeds.shape[1]
        kw["past_key_values"] = cache
        kw["cache_position"] = torch.arange(n_prefix, n_prefix + n_new,
                                            device=embeds.device)
    logits = model(inputs_embeds=embeds * norm, logits_to_keep=n_tgt + 1, **kw).logits
    if cache is not None:
        cache.crop(n_prefix)
    pred = logits[:, -n_tgt - 1:-1, :]
    tgt = target_ids.expand(embeds.shape[0], -1)
    return F.cross_entropy(pred.transpose(1, 2).float(), tgt, reduction="none").mean(dim=1)


def build_prefix_cache(model, pre_e, norm, batch, device):
    """Keys/values for the fixed prompt prefix, expanded to `batch` rows."""
    from transformers import DynamicCache
    cache = DynamicCache()
    with torch.no_grad():
        model(inputs_embeds=pre_e * norm, past_key_values=cache, use_cache=True,
              cache_position=torch.arange(pre_e.shape[1], device=device))
    if batch > 1:
        cache.batch_repeat_interleave(batch)
    return cache


def gcg_attack(model, tokenizer, embed_w, pre_ids, post_ids, target_ids, args, device,
               banned_toks=None):
    vocab = embed_w.shape[0]
    norm = model.config.get_text_config().hidden_size ** 0.5
    init_id = tokenizer.encode("!", add_special_tokens=False)[-1]
    adv_ids = torch.full((args.n_adv_tokens,), init_id, device=device)
    pre_e, post_e = embed_w[pre_ids].unsqueeze(0), embed_w[post_ids].unsqueeze(0)
    tgt_e = embed_w[target_ids].unsqueeze(0)
    best_loss, best_ids = float("inf"), adv_ids.clone()
    batch_size = args.batch_size
    n_prefix = pre_ids.shape[0]
    prefix_cache, cache_batch = None, None

    for step in range(args.num_steps):
        one_hot = F.one_hot(adv_ids, vocab).to(embed_w.dtype)
        one_hot.requires_grad_(True)
        adv_e = (one_hot @ embed_w).unsqueeze(0)
        # This pass keeps activations for backward, which OOMs on the longest
        # contextual prompts even at batch 1. Checkpointing is gated on .training
        # in transformers, and attention_dropout is 0.0, so toggling train mode
        # here is numerically inert.
        model.train()
        loss = target_loss(model, torch.cat([pre_e, adv_e, post_e, tgt_e], dim=1), target_ids, norm)
        loss.backward()
        model.eval()
        grad = one_hot.grad.clone()
        model.zero_grad(set_to_none=True)
        del one_hot, adv_e

        # Sample candidates: random position, random token from that position's top-k.
        # Forbidden tokens are pushed above the max so top-k never selects them,
        # matching HarmBench's sample_control.
        if banned_toks is not None:
            grad[:, banned_toks] = grad.max() + 1
        topk = (-grad).topk(args.topk, dim=1).indices
        pos = torch.randint(0, args.n_adv_tokens, (args.search_width,), device=device)
        pick = topk[pos, torch.randint(0, args.topk, (args.search_width,), device=device)]
        cands = adv_ids.repeat(args.search_width, 1)
        cands[torch.arange(args.search_width, device=device), pos] = pick

        losses = []
        with torch.no_grad():
            i = 0
            while i < args.search_width:
                chunk = cands[i:i + batch_size]
                b = chunk.shape[0]
                suffix = torch.cat([embed_w[chunk], post_e.expand(b, -1, -1),
                                    tgt_e.expand(b, -1, -1)], dim=1)
                try:
                    if args.no_prefix_cache:
                        losses.append(target_loss(
                            model, torch.cat([pre_e.expand(b, -1, -1), suffix], dim=1),
                            target_ids, norm))
                    else:
                        if cache_batch != b:
                            prefix_cache = build_prefix_cache(model, pre_e, norm, b, device)
                            cache_batch = b
                        losses.append(target_loss(model, suffix, target_ids, norm,
                                                  cache=prefix_cache, n_prefix=n_prefix))
                except torch.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    del suffix
                    prefix_cache, cache_batch = None, None
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning(f"    OOM — dropping batch size to {batch_size}")
                    continue
                i += batch_size
        losses = torch.cat(losses)
        idx = losses.argmin()
        adv_ids = cands[idx]
        if losses[idx].item() < best_loss:
            best_loss, best_ids = losses[idx].item(), adv_ids.clone()

        if step % args.log_every == 0:
            logger.info(f"    step {step:4d}  loss {losses[idx].item():.4f}  best {best_loss:.4f}")
        if best_loss <= args.early_stop_loss:
            logger.info(f"    early stop at step {step} (loss {best_loss:.4f})")
            break

    return tokenizer.decode(best_ids), best_loss, step + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=GEMMA_SNAPSHOT)
    p.add_argument("--behaviors_csv", default=BEHAVIORS_CSV)
    p.add_argument("--targets_json", default=TARGETS_JSON)
    p.add_argument("--output_dir", default="full_trait_output/gcg_gemma_test_cases")
    p.add_argument("--num_steps", type=int, default=500)
    p.add_argument("--search_width", type=int, default=512)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--n_adv_tokens", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16, help="candidate eval chunk size")
    p.add_argument("--early_stop_loss", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only the first N behaviours of the shard")
    p.add_argument("--allow_non_ascii", action="store_true",
                   help="Allow non-ASCII / special tokens in the suffix. HarmBench runs "
                        "GCG with these forbidden, so leaving this off keeps the search "
                        "space identical to the published Llama run.")
    p.add_argument("--no_prefix_cache", action="store_true",
                   help="Recompute the prompt prefix for every candidate instead of "
                        "caching its keys/values. Slower; kept to verify the cache.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for candidate sampling, so runs are reproducible.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard{args.shard_idx}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["behavior_id"] for l in out_path.read_text().splitlines() if l.strip()}
        logger.info(f"Resuming — {len(done)} behaviours already done")

    behaviors = load_behaviors(args.behaviors_csv, args.targets_json)
    behaviors = behaviors[args.shard_idx::args.num_shards]
    if args.limit:
        behaviors = behaviors[:args.limit]
    todo = [b for b in behaviors if b["behavior_id"] not in done]
    logger.info(f"Shard {args.shard_idx}/{args.num_shards}: {len(todo)} behaviours to run")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda:0", trust_remote_code=True)
    model.eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = next(model.parameters()).device
    embed_w = model.get_input_embeddings().weight

    banned_toks = None
    if not args.allow_non_ascii:
        banned_toks = get_nonascii_toks(tokenizer, device)
        logger.info(f"  Forbidding {len(banned_toks)} non-ASCII/special tokens in the suffix")

    for i, b in enumerate(todo, 1):
        t0 = time.time()
        logger.info(f"[{i}/{len(todo)}] {b['behavior_id']}")
        pre_ids, post_ids, target_ids = build_segments(tokenizer, b["prompt"], b["target"], device)
        torch.manual_seed(args.seed)
        suffix, loss, steps = gcg_attack(model, tokenizer, embed_w, pre_ids, post_ids,
                                         target_ids, args, device, banned_toks)
        rec = {"behavior_id": b["behavior_id"], "behavior": b["behavior"],
               "category": b["category"], "adv_suffix": suffix,
               "test_case": f"{b['prompt']} {suffix}", "loss": loss,
               "steps": steps, "seconds": round(time.time() - t0, 1)}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        logger.info(f"    done in {rec['seconds']}s  loss {loss:.4f}  steps {steps}")
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Shard {args.shard_idx} complete → {out_path}")


if __name__ == "__main__":
    main()
