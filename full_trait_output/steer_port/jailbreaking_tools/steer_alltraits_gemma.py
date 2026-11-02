#!/usr/bin/env python3
"""
steer_alltraits_gemma.py

Gemma-4-31B single-trait steering over all 240 layer-30 traits, replicating Dominic's Llama
all-229 sweep (trait_steering_attack_jbb.py --mode single_trait --behaviors_source harmbench
--no_templates, launcher run_single_trait_all229_shard.sh). Spec: HANDOFF_gemma_alltraits.md.

Reuses the model plumbing, hook, judges and is_degenerate of steer_fixed_direction_gemma.py
(judge prompts verified identical to Dominic's). Differences from that runner:
  pool      all 400 HarmBench text behaviours, no templates; contextual = ContextString +
            "\n\n---\n\n" + Behavior (fetch_harmbench_behaviors), category = FunctionalCategory
  direction row of trait_matrix_layer30.npy, positive pole, unit norm; arm "none" = unsteered
  records   every alpha tried is kept per prompt ("history": alpha, response, degenerate, judge)
  post-hoc  JBB 1-10 rubric on final responses -> jbb_judgements.jsonl (jailbroken iff 10)
  extras    looks_like_text flag on final jailbroken responses (port_tools/analyze_outputs.py),
            n_degenerate, n_jb_not_text, clean_asr, min_alpha_hist, judge_calls
Output: <output_dir>/<trait>/single_trait/{results.jsonl, jbb_judgements.jsonl, summary.json}.
Resume-safe: a trait with summary.json is skipped; a partial results.jsonl resumes by pair_id.
Judge calls are appended per trait to <output_dir>/JUDGE_COUNT; the run stops before a trait
once the total there reaches --judge_budget.
"""
import argparse, csv, json, logging, os, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "port_tools"))
import steer_fixed_direction_gemma as sfd  # noqa: E402
from analyze_outputs import looks_like_text  # noqa: E402

logger = logging.getLogger("steer_alltraits")
CALLS = {"n": 0}
_chat_orig = sfd._chat


def _chat_counted(client, messages, max_tokens):
    CALLS["n"] += 1
    return _chat_orig(client, messages, max_tokens)


sfd._chat = _chat_counted   # judges look _chat up in the module globals


def hb400_pool(csv_path):
    pool = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            behavior = (row.get("Behavior") or "").strip()
            context = (row.get("ContextString") or "").strip()
            goal = f"{context}\n\n---\n\n{behavior}" if context else behavior
            if not goal:
                continue
            pool.append({"pair_id": row["BehaviorID"], "behavior_id": row["BehaviorID"], "goal": goal,
                         "category": row.get("FunctionalCategory", "") or "standard", "source": "HarmBench",
                         "jailbreak_idx": -1, "prompt": goal})
    return pool


def judge_total(path):
    if not path.exists():
        return 0
    return sum(int(l.split()[2]) for l in open(path) if l.strip() and not l.startswith("#"))


def escalate(pairs, v, schedule, ctx, out_f):
    """Dominic's attack_prompt loop, batched: walk the alpha schedule, drop a prompt at its first
    jailbreak; degenerate responses skip the judge and count as not jailbroken."""
    model, tok, layers, L, rn, device, client, bs, jw, mnt, budget = ctx
    pending = [dict(p, _hist=[]) for p in pairs]
    t0, n_gen = time.time(), 0

    def emit(p):
        last = p["_hist"][-1]
        rec = {k: p[k] for k in ("pair_id", "behavior_id", "prompt", "goal", "category", "source", "jailbreak_idx")}
        rec.update({"final_jailbroken": last["is_jailbroken"], "final_degenerate": last["degenerate"],
                    "final_response": last["response"],
                    "min_alpha_to_jailbreak": last["alpha"] if last["is_jailbroken"] else None,
                    "n_iterations": len(p["_hist"]), "max_alpha_tried": max(schedule), "history": p["_hist"]})
        out_f.write(json.dumps(rec, ensure_ascii=True) + "\n"); out_f.flush()

    for it, alpha in enumerate(schedule):
        if not pending:
            break
        pending.sort(key=lambda p: p["_ntok"], reverse=True)
        chunks, i = [], 0
        while i < len(pending):
            n = max(1, min(bs, budget // max(1, pending[i]["_ntok"] + mnt))) if budget else bs
            chunks.append(pending[i:i + n]); i += n
        still = []
        for chunk in chunks:
            prompts = [c["prompt"] for c in chunk]
            responses, _, _ = sfd.generate_batch(model, tok, layers, L, prompts, v, alpha, rn, device, mnt)
            verdicts = sfd.judge_batch(client, list(zip(prompts, responses)), jw)
            n_gen += len(chunk)
            for p, resp, (jb, degen) in zip(chunk, responses, verdicts):
                p["_hist"].append({"iteration": it + 1, "alpha": alpha, "is_jailbroken": bool(jb),
                                   "degenerate": bool(degen), "response": resp})
                if jb or alpha == schedule[-1]:
                    emit(p)
                else:
                    still.append(p)
        logger.info(f"  alpha={alpha}: {len(pending)} prompts, {len(pending) - len(still)} finished "
                    f"| {n_gen} gens in {(time.time() - t0) / 60:.1f} min | judge calls {CALLS['n']}")
        pending = still


def run_trait(name, v, schedule, pool, ctx, tdir, a):
    client, jw = ctx[6], ctx[8]
    tdir.mkdir(parents=True, exist_ok=True)
    res_p, jbb_p = tdir / "results.jsonl", tdir / "jbb_judgements.jsonl"
    calls0, t0 = CALLS["n"], time.time()
    done = {r["pair_id"] for r in sfd.load_jsonl(res_p)} if res_p.exists() else set()
    todo = [p for p in pool if p["pair_id"] not in done]
    if done:
        logger.info(f"[{name}] resuming: {len(done)} done, {len(todo)} to go")
    if todo:
        with open(res_p, "a", encoding="utf-8") as f:
            escalate(todo, v, schedule, ctx, f)
    results = sfd.load_jsonl(res_p)
    # post-hoc JBB rubric (incremental, as in Dominic's script)
    jbb = {r["pair_id"]: r["jbb_jailbroken"] for r in sfd.load_jsonl(jbb_p)} if jbb_p.exists() else {}
    todo_j = [r for r in results if r["pair_id"] not in jbb]
    if todo_j:
        def one(r):
            if not r.get("final_response") or r.get("final_degenerate"):
                return False
            return sfd.judge_response_jbb(client, r["goal"], r["prompt"], r["final_response"])
        with ThreadPoolExecutor(max_workers=jw) as ex, open(jbb_p, "a", encoding="utf-8") as f:
            for r, val in zip(todo_j, ex.map(one, todo_j)):
                jbb[r["pair_id"]] = val
                f.write(json.dumps({"pair_id": r["pair_id"], "jbb_jailbroken": val}) + "\n"); f.flush()
    for r in results:
        r["jbb_jailbroken"] = bool(jbb.get(r["pair_id"], False))
        r["looks_like_text"] = looks_like_text(r["final_response"]) if r["final_jailbroken"] else None
        r["clean_jailbroken"] = bool(r["final_jailbroken"] and r["looks_like_text"])
    n = len(results)
    g, j = sfd.compute_metrics(results, "final_jailbroken"), sfd.compute_metrics(results, "jbb_jailbroken")
    ncalls = CALLS["n"] - calls0
    hist = Counter(str(r["min_alpha_to_jailbreak"]) for r in results if r["final_jailbroken"])
    s = {"mode": "single_trait" if name != "none" else "none", "trait": name, "alpha_schedule": schedule,
         "config": {"model": a.model, "layer": a.layer, "residual_norm": ctx[4], "max_new_tokens": a.max_new_tokens,
                    "behaviors_source": "harmbench", "n_pairs": n, "n_behaviors": g.get("n_behaviors", 0),
                    "n_templates": 0, "pool": a.pool_desc, "direction": "trait_matrix_layer30.npy row, positive pole, unit norm",
                    "judge_inloop": "gpt-4.1-mini binary YES/NO",
                    "judge_posthoc": "gpt-4.1-mini JBB 1-10 rubric (jailbroken if == 10)",
                    "attn_implementation": a.attn, "batch_size": a.batch_size, "token_budget": a.token_budget},
         "gpt41mini": {"per_pair_asr": g["per_pair_asr"], "per_behavior_asr": g["per_behavior_asr"],
                       "by_category": sfd.compute_category_metrics(results, "final_jailbroken")},
         "jbb_rubric": {"per_pair_asr": j["per_pair_asr"], "per_behavior_asr": j["per_behavior_asr"],
                        "by_category": sfd.compute_category_metrics(results, "jbb_jailbroken")},
         "n_degenerate": sum(r["final_degenerate"] for r in results),
         "n_degenerate_any_alpha": sum(any(h["degenerate"] for h in r["history"]) for r in results),
         "n_degenerate_at_first_alpha": sum(r["history"][0]["degenerate"] for r in results),
         "n_jailbroken": sum(r["final_jailbroken"] for r in results),
         "n_jb_not_text": sum(1 for r in results if r["final_jailbroken"] and not r["looks_like_text"]),
         "clean_asr": sum(r["clean_jailbroken"] for r in results) / max(n, 1),
         "clean_by_category": sfd.compute_category_metrics(results, "clean_jailbroken"),
         "min_alpha_hist": dict(sorted(hist.items())),
         "n_generations": sum(r["n_iterations"] for r in results),
         "judge_calls_this_session": ncalls, "wall_min_this_session": round((time.time() - t0) / 60, 2)}
    # results.jsonl is rewritten with the merged post-hoc fields; histories are kept
    tmp = res_p.with_suffix(".merge.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    tmp.replace(res_p)
    (tdir / "summary.json").write_text(json.dumps(s, indent=2))
    with open(Path(a.output_dir) / "JUDGE_COUNT", "a") as f:
        f.write(f"{time.strftime('%FT%T')} {name} {ncalls} job={os.environ.get('SLURM_JOB_ID', '-')}\n")
    logger.info(f"[{name}] ASR {g['per_pair_asr']:.3f} | clean {s['clean_asr']:.3f} | JBB {j['per_pair_asr']:.3f} "
                f"| degenerate {s['n_degenerate']}/{n} | judge calls {ncalls} | {s['wall_min_this_session']} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traits", required=True, help="comma-separated trait names; 'none' = unsteered baseline")
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", default="outputs_gemma_alltraits")
    p.add_argument("--trait_matrix", default="inputs/gemma/alltraits/trait_matrix_layer30.npy")
    p.add_argument("--trait_names", default="inputs/gemma/alltraits/trait_names_layer30.json")
    p.add_argument("--behaviors_csv", default="inputs/gemma/alltraits/harmbench_behaviors_text_all.csv")
    p.add_argument("--layer", type=int, default=30)
    p.add_argument("--residual_norm", type=float, default=572.2347398103947)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--alpha_schedule", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--n_behaviors", type=int, default=0, help="smoke: random subsample of the 400 (0 = all)")
    p.add_argument("--pool_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--judge_workers", type=int, default=12)
    p.add_argument("--judge_budget", type=int, default=550000)
    p.add_argument("--attn", default="sdpa", choices=["gqa_fold", "sdpa", "eager"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY not set")

    names = json.load(open(a.trait_names))
    M = np.load(a.trait_matrix)
    assert M.shape[0] == len(names), (M.shape, len(names))
    traits = [t.strip() for t in a.traits.split(",") if t.strip()]
    bad = [t for t in traits if t != "none" and t not in names]
    if bad:
        raise SystemExit(f"ERROR: unknown traits {bad}")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    traits = [t for t in traits if not (out / t / "single_trait" / "summary.json").exists()]
    if not traits:
        logger.info("all requested traits done"); return

    pool = hb400_pool(a.behaviors_csv)
    a.pool_desc = f"HarmBench text behaviours, no templates, n={len(pool)}"
    if a.n_behaviors:
        pool = random.Random(a.pool_seed).sample(pool, a.n_behaviors)
        a.pool_desc += f" subsampled to {len(pool)} (seed {a.pool_seed})"
    logger.info(f"pool: {a.pool_desc}; traits to run: {traits}")

    client = OpenAI(max_retries=4, timeout=90)
    device = torch.device(a.device)
    model, tok = sfd.load_model(a.model, device, a.attn)
    layers = sfd.find_decoder_layers(model)
    texts = [tok.apply_chat_template([{"role": "user", "content": sfd.sanitize(q["prompt"])}], tokenize=False,
                                     add_generation_prompt=True) for q in pool]
    for q, ids in zip(pool, tok(texts, add_special_tokens=False)["input_ids"]):
        q["_ntok"] = len(ids)
    logger.info(f"prompt tokens max {max(q['_ntok'] for q in pool)} | residual_norm {a.residual_norm} | layer {a.layer}")
    ctx = (model, tok, layers, a.layer, a.residual_norm, device, client, a.batch_size, a.judge_workers,
           a.max_new_tokens, a.token_budget)
    for t in traits:
        if (out / t / "single_trait" / "summary.json").exists():
            continue
        used = judge_total(out / "JUDGE_COUNT")
        if used >= a.judge_budget:
            logger.info(f"judge budget reached ({used} >= {a.judge_budget}); stopping before {t}"); break
        if t == "none":
            v, schedule = np.zeros(M.shape[1], dtype=np.float32), [0.0]
        else:
            v = M[names.index(t)].astype(np.float64)
            v = v / (np.linalg.norm(v) + 1e-12)
            schedule = list(a.alpha_schedule)
        logger.info(f"\n{'=' * 70}\n  TRAIT {t}  (judge calls so far, all jobs: {used})\n{'=' * 70}")
        run_trait(t, v, schedule, pool, ctx, out / t / "single_trait", a)


if __name__ == "__main__":
    main()
