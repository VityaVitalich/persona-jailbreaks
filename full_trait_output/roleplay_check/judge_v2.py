#!/usr/bin/env python3
"""
Refusal judge for the v2 synthetic responses, exactly as MR-Eval's over-refusal bench runs it
(~/MR-Eval/overrefusal/run_eval.py): the OR-Bench classification prompt (Cui et al., 2024, verbatim
from justincui03/or-bench) -> direct_answer | direct_refusal | indirect_refusal, judged by
DeepSeek-V4-Flash (snapshot 0423) on OpenRouter through mreval.judge (client, snapshot check,
provider routing), temperature 0. max_tokens is 2000 instead of MR-Eval's 300: V4-Flash spends hidden reasoning tokens first and at 300 ~10% of calls end before the [[class]] answer. The prompt string and parser are read from
run_eval.py itself, so they cannot drift. QUESTION = the full prompt the model saw.

refused = direct_refusal or indirect_refusal; complied = direct_answer.
Writes synthetic_v2_judged.jsonl {id, harm, cls, judge_text}; cached, reruns only send missing ids.
"""
import ast, asyncio, json, re, sys
from pathlib import Path

MREVAL = Path.home() / "MR-Eval"
sys.path.insert(0, str(MREVAL))
from mreval.judge import build_judge_client, extra_body_for  # noqa: E402

HERE = Path(__file__).resolve().parent
src = (MREVAL / "overrefusal/run_eval.py").read_text()
PROMPT = next(n.value.value for n in ast.parse(src).body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "OR_BENCH_JUDGE_PROMPT")
LABELS = ("direct_answer", "indirect_refusal", "direct_refusal")  # = run_eval.CLASS_LABELS (order matters)
BRACKET = re.compile(r"\[\[(.*?)\]\]")


def parse(text):  # = run_eval.parse_judge_class
    m = BRACKET.findall(text or "")
    if not m: return None
    last = m[-1].strip().lower()
    return next((l for l in LABELS if l in last), None)


async def main():
    client, model = build_judge_client("openrouter", "deepseek/deepseek-v4-flash")
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "synthetic_v2_prompts.jsonl")}
    resp = [json.loads(l) for l in open(HERE / "synthetic_v2_responses.jsonl")]
    out_p = HERE / "synthetic_v2_judged.jsonl"
    done = {d["id"] for d in map(json.loads, open(out_p)) if d["cls"]} if out_p.exists() else set()
    todo = [r for r in resp if r["id"] not in done]
    print(f"{len(resp)} responses, {len(done)} judged, {len(todo)} to judge", flush=True)
    sem = asyncio.Semaphore(48)

    async def one(r):
        msg = PROMPT.format(question=prompts[r["id"]]["prompt"], response=r["response_text"])
        async with sem:
            for attempt in range(8):
                try:
                    c = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}],
                                                             temperature=0.0, max_tokens=2000, extra_body=extra_body_for(model))
                    t = c.choices[0].message.content or ""
                    cls = parse(t)
                    if cls: return {"id": r["id"], "harm": prompts[r["id"]]["harm"], "cls": cls, "judge_text": t}
                except Exception as e:
                    t = repr(e)
                await asyncio.sleep(min(60, 2 ** attempt))
        return {"id": r["id"], "harm": prompts[r["id"]]["harm"], "cls": None, "judge_text": t}

    with open(out_p, "a") as f:
        for i, fut in enumerate(asyncio.as_completed([one(r) for r in todo])):
            f.write(json.dumps(await fut, ensure_ascii=False) + "\n")
            if i % 1000 == 0: f.flush(); print(f"  {i}/{len(todo)}", flush=True)
    from collections import Counter
    rows = {d["id"]: d for d in map(json.loads, open(out_p))}
    print(Counter((d["harm"], d["cls"]) for d in rows.values()))


if __name__ == "__main__":
    asyncio.run(main())
