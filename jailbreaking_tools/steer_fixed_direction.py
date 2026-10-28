#!/usr/bin/env python3
"""
steer_fixed_direction.py

Steering attack driven by a FIXED, precomputed direction instead of a freshly
trained classifier normal, on the three benchmarks of tab:attack.

Motivation: tab:refusal-direction and fig:direction-story show that the Arditi
refusal direction and the unsupervised jailbreak direction dmu = mean(jailbroken)
- mean(refused) both lie substantially inside the 229-trait span (40.7% and 48.2%
of their squared norm) and detect jailbreaks about as well as the trained
classifier normal. This runs the causal counterpart: steer with those two
directions and compare the ASR against the published steer-w rows.

Protocol is the published one: unit-norm direction added as alpha * RESIDUAL_NORM
* v at layer 16 on every token, greedy decoding, MAX_NEW_TOKENS, the alpha
schedule walked in increasing order, gpt-4.1-mini binary judge, stop at the first
alpha that jailbreaks. Reported ASR is therefore "jailbroken at some alpha in the
schedule", and min_alpha_to_jailbreak is stored per pair.

The escalation is reorganised ROUND-WISE rather than pair-wise: every pair still
unresolved is generated at the current alpha in one batch, judged, and the
successes drop out before the next alpha. That is the same per-pair outcome as
escalating each pair on its own -- pairs are independent, and a pair's verdict at
one alpha does not depend on any other pair -- but it lets one forward pass serve
a whole batch instead of a single prompt, which is where essentially all of the
wall-clock goes (batch size 1 runs an 8B model at ~39 tok/s).

Machinery is reused rather than reimplemented, so the numbers stay comparable to
the published cells: the steering hook, the degeneracy filter, both judges and the
metric functions are Dominic's. What is new here is the vector loader, the batched
round-wise driver, and a resume-safe writer (his run_steering opens results.jsonl
in "w" mode, so an eviction at pair 1900/2000 loses the run).

One inherited cost is dropped: run_llama_steered ran a full extra forward pass with
output_hidden_states=True before each generation to collect the layer-16
activation for a classifier score. This path steers a fixed vector and never reads
that score, so the pass is skipped.

Usage:
  uv run python jailbreaking_tools/steer_fixed_direction.py \
      --benchmark harmbench --directions .../steering_directions_layer16.pt \
      --names refusal_pos1_neg__full,dmu__full --batch_size 32
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
# full_trait_tools first: steer_persona_w_hb_balanced adds it via a REPO_ROOT that
# only exists in the scratch space the published runs used.
sys.path.insert(0, str(HERE.parent / "full_trait_tools"))
sys.path.insert(0, str(HERE))
import trait_steering_attack_jbb as base
from trait_steering_attack_jbb import (
    LAYER,
    MODEL_NAME,
    SteeringHook,
    compute_category_metrics,
    compute_metrics,
    fetch_jailbreaks,
    fetch_jbb_behaviors,
    fetch_wildjailbreak_behaviors,
    is_degenerate,
    judge_response_gpt41mini,
    judge_response_jbb,
    logger,
)
from trait_steering_attack_jbb_persona_w import build_raw_goal_pool
from steer_persona_w_hb_balanced import build_full_pool


def load_results(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def generate_batch(model, tokenizer, prompts, w_vec, alpha, device):
    """Greedy generation for a batch of prompts with the steering hook active.

    Left padding, so every sequence's real tokens end flush against the
    generation position; pad positions are masked out of attention, and the
    hook's addition to them is inert.
    """
    # Same sanitisation run_llama_steered applied: 183 of the 1935 HarmBench
    # prompts carry lone surrogates from a broken jailbreak template, and the
    # Rust tokenizer cannot accept such a str at all. Replacing is what the
    # published runs fed the model, so it is what we must feed it too.
    prompts = [p.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
               for p in prompts]
    texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
             for p in prompts]
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
    hook = SteeringHook(w_vec, alpha, base.RESIDUAL_NORM, device)
    hook.register(model, LAYER)
    try:
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=base.MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    finally:
        hook.remove()
    new = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new]


def judge_batch(client, items, workers):
    """items: [(prompt, response)]. Degenerate responses skip the judge, as in
    escalate_until_jailbroken. Returns [(is_jailbroken, degenerate)]."""
    def one(it):
        prompt, response = it
        if is_degenerate(response):
            return False, True
        return judge_response_gpt41mini(client, prompt, response), False
    if workers <= 1:
        return [one(it) for it in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, items))


def escalate_rounds(pairs, w_vec, model, tokenizer, device, client, out_f,
                    batch_size, judge_workers, extra_fields=None):
    """Round-wise escalation over the alpha schedule; writes each pair as it
    resolves. Record schema matches escalate_until_jailbroken."""
    pending = [dict(p, _iters=0, _last=("", False)) for p in pairs]
    schedule = base.ALPHA_SCHEDULE
    n_gen = 0
    t0 = time.time()

    def emit(pair, alpha, response, degenerate, jailbroken):
        rec = {
            "pair_id": pair.get("pair_id"),
            "prompt": pair.get("formatted_prompt") or pair.get("prompt"),
            "goal": pair.get("goal", pair.get("behavior_text", "")),
            "final_jailbroken": jailbroken,
            "final_degenerate": degenerate,
            "final_response": response,
            "min_alpha_to_jailbreak": alpha,
            "n_iterations": pair["_iters"],
        }
        for k in ("behavior_id", "category", "source", "jailbreak_idx"):
            if k in pair:
                rec[k] = pair[k]
        if extra_fields:
            rec.update(extra_fields)
        out_f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        out_f.flush()

    for alpha in schedule:
        if not pending:
            break
        # longest first: batches of similar length waste the least on padding
        pending.sort(key=lambda p: len(p.get("formatted_prompt") or p["prompt"]),
                     reverse=True)
        still = []
        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            prompts = [c.get("formatted_prompt") or c["prompt"] for c in chunk]
            responses = generate_batch(model, tokenizer, prompts, w_vec, alpha,
                                       device)
            verdicts = judge_batch(client, list(zip(prompts, responses)),
                                   judge_workers)
            n_gen += len(chunk)
            for pair, resp, (is_jb, degen) in zip(chunk, responses, verdicts):
                pair["_iters"] += 1
                pair["_last"] = (resp, degen)
                if is_jb:
                    emit(pair, alpha, resp, degen, True)
                else:
                    still.append(pair)
            rate = n_gen / max(time.time() - t0, 1e-9)
            resolved = i + len(chunk) - len(still)   # of those generated so far
            logger.info(f"  alpha={alpha} | batch {i//batch_size + 1}"
                        f"/{(len(pending) + batch_size - 1)//batch_size} | "
                        f"{resolved}/{i + len(chunk)} resolved this round | "
                        f"{n_gen} generations at {rate*60:.1f}/min")
        pending = still

    for pair in pending:                    # never jailbroken within the budget
        emit(pair, None, pair["_last"][0], pair["_last"][1], False)
    logger.info(f"  {n_gen} generations in {(time.time()-t0)/60:.1f} min "
                f"({n_gen/max(time.time()-t0, 1e-9)*60:.1f}/min)")


def run_arm(name, w_vec, pairs, model, tokenizer, device, client, arm_dir,
            batch_size, judge_workers, harmbench):
    arm_dir.mkdir(parents=True, exist_ok=True)
    out_path = arm_dir / "results.jsonl"
    done = {str(r.get("pair_id")) for r in load_results(out_path)}
    todo = [p for p in pairs if str(p.get("pair_id")) not in done]
    if done:
        logger.info(f"Resuming: {len(done)} pairs already done, {len(todo)} to go")
    if todo:
        with open(out_path, "a", encoding="utf-8") as out_f:
            escalate_rounds(
                todo, w_vec, model, tokenizer, device, client, out_f,
                batch_size, judge_workers,
                extra_fields={"started_jailbroken": False} if harmbench else None)

    results = load_results(out_path)
    if harmbench:
        rate = (sum(1 for r in results if r["final_jailbroken"]) / len(results)
                if results else 0.0)
        return {"asr_started_not_jailbroken": rate, "n_started_rf": len(results)}, rate

    # JBB and WildJB additionally carry the post-hoc JBB 1-10 rubric verdict
    todo_j = [r for r in results if r.get("jbb_jailbroken") is None]
    if todo_j:
        logger.info(f"[{name}] post-hoc JBB judge on {len(todo_j)} responses ...")

        def one(r):
            if not r.get("final_response") or r.get("final_degenerate"):
                return False
            return judge_response_jbb(client, goal=r["goal"], prompt=r["prompt"],
                                      response=r["final_response"])
        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            for r, v in zip(todo_j, ex.map(one, todo_j)):
                r["jbb_jailbroken"] = v
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    gpt_m = compute_metrics(results, "final_jailbroken")
    jbb_m = compute_metrics(results, "jbb_jailbroken")
    return ({"gpt41mini": {"per_pair_asr": gpt_m["per_pair_asr"],
                           "per_behavior_asr": gpt_m["per_behavior_asr"],
                           "by_category": compute_category_metrics(
                               results, "final_jailbroken")},
             "jbb_rubric": {"per_pair_asr": jbb_m["per_pair_asr"],
                            "per_behavior_asr": jbb_m["per_behavior_asr"],
                            "by_category": compute_category_metrics(
                                results, "jbb_jailbroken")},
             "n_pairs": len(results)},
            gpt_m["per_pair_asr"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=["harmbench", "jbb", "wildjb"])
    p.add_argument("--directions", required=True,
                   help="steering_directions_layer16.pt from prep_steering_directions.py")
    p.add_argument("--names", required=True,
                   help="comma-separated vector names to run, in order")
    p.add_argument("--output_dir", default="full_trait_output/steer_fixed_directions")
    p.add_argument("--model", default=MODEL_NAME,
                   help="Weights to steer. Any byte-identical mirror of "
                        f"{MODEL_NAME} gives the same result.")
    p.add_argument("--classified_path", default=None,
                   help="HarmBench classified_responses.jsonl; defaults to the path "
                        "baked into trait_steering_attack_jbb.py, which only "
                        "resolves inside the scratch space of the published runs.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--judge_workers", type=int, default=8)
    p.add_argument("--alpha_schedule", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5],
                   help="Ceiling 0.5 matches the budget tab:attack is reported at.")
    p.add_argument("--n_refused", type=int, default=0,
                   help="harmbench: subsample the refused pool (0 = all 1935).")
    p.add_argument("--max_behaviors", type=int, default=2000,
                   help="wildjb: behaviours to sample (2000 matches the published run).")
    p.add_argument("--pool_seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap the pool (for an A/B against an existing run).")
    p.add_argument("--test", action="store_true", help="3 pairs only")
    args = p.parse_args()

    base.ALPHA_SCHEDULE = args.alpha_schedule
    base.MAX_ITER = len(args.alpha_schedule)
    if args.classified_path:
        base.HARMBENCH_CLASSIFIED = Path(args.classified_path)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set")
    client = OpenAI()

    bundle = torch.load(args.directions, map_location="cpu", weights_only=False)
    vectors, meta = bundle["vectors"], bundle.get("meta", {})
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    missing = [n for n in names if n not in vectors]
    if missing:
        raise SystemExit(f"ERROR: not in {args.directions}: {missing}\n"
                         f"available: {sorted(vectors)}")
    if bundle.get("layer", LAYER) != LAYER:
        raise SystemExit(f"ERROR: directions are for layer {bundle['layer']}, "
                         f"steering runs at layer {LAYER}")

    logger.info(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # required for decoder-only batching
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    # ---- target pool: the same one the published row for this column used ----
    harmbench = args.benchmark == "harmbench"
    if harmbench:
        _jb_all, rf_all = build_full_pool(fetch_jailbreaks())
        n_rf = 3 if args.test else args.n_refused
        pairs = (random.Random(args.pool_seed).sample(rf_all, min(n_rf, len(rf_all)))
                 if n_rf else rf_all)
        pool_desc = (f"HarmBench human_jailbreak baseline-refused group, "
                     f"n={len(pairs)}"
                     + (f" (subsampled from {len(rf_all)}, seed {args.pool_seed})"
                        if n_rf else " (all)"))
    else:
        behaviors = (fetch_jbb_behaviors() if args.benchmark == "jbb"
                     else fetch_wildjailbreak_behaviors(max_behaviors=args.max_behaviors))
        pairs = build_raw_goal_pool(behaviors, args.test)
        pool_desc = f"{args.benchmark} raw goals, n={len(pairs)}"
    if args.limit:
        pairs = pairs[:args.limit]
        pool_desc += f" [limited to {len(pairs)}]"
    logger.info(f"pool: {pool_desc}")

    out_root = Path(args.output_dir)
    summaries = {}
    for name in names:
        v = vectors[name].float().numpy().astype(np.float64)
        nrm = float(np.linalg.norm(v))
        if abs(nrm - 1.0) > 1e-3:
            raise SystemExit(f"ERROR: {name} is not unit-norm (|v|={nrm:.4f}); "
                             f"arms would differ in magnitude, not only direction")
        logger.info(f"\n{'='*70}\n  ARM {name} on {args.benchmark}"
                    f"  ({meta.get(name, {}).get('description', '')})\n{'='*70}")
        arm_dir = out_root / args.benchmark / name
        block, headline = run_arm(name, v, pairs, model, tokenizer, device, client,
                                 arm_dir, args.batch_size, args.judge_workers,
                                 harmbench)
        summary = {
            "steering_vector": name,
            "direction_meta": meta.get(name, {}),
            "directions_file": str(args.directions),
            "benchmark": args.benchmark,
            "alpha_schedule": base.ALPHA_SCHEDULE,
            "config": {
                "model": args.model,
                "layer": LAYER,
                "residual_norm": base.RESIDUAL_NORM,
                "max_new_tokens": base.MAX_NEW_TOKENS,
                "batch_size": args.batch_size,
                "target": pool_desc,
                "judge_inloop": "gpt-4.1-mini binary YES/NO",
                **({} if harmbench else {
                    "judge_posthoc":
                    "gpt-4.1-mini JBB 1-10 rubric (jailbroken if == 10)"}),
            },
            **block,
        }
        (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries[name] = summary
        logger.info(f"[{name}] {args.benchmark} ASR = {headline:.1%}")

    (out_root / f"arms_summary_{args.benchmark}.json").write_text(
        json.dumps(summaries, indent=2))
    sep = "=" * 72
    print(f"\n{sep}\n  Fixed-direction steering — {pool_desc}\n"
          f"  alpha schedule {base.ALPHA_SCHEDULE} | batch {args.batch_size}\n{sep}")
    for name, s in summaries.items():
        asr = (s.get("asr_started_not_jailbroken") if harmbench
               else s["gpt41mini"]["per_pair_asr"])
        extra = ("" if harmbench
                 else f"  jbb_rubric={s['jbb_rubric']['per_pair_asr']:.3f}")
        print(f"  {name:28s} ASR={asr:.3f}{extra}")
    print(sep)
    logger.info(f"Saved to {out_root}/{args.benchmark}/")


if __name__ == "__main__":
    main()
