#!/usr/bin/env python3
"""Stage 1: greedy responses of Qwen3-32B (vLLM) for qwen_inputs.jsonl. Chat template, single user turn, no system
prompt, add_generation_prompt, enable_thinking=False (the Assistant Axis recipe for Qwen); temperature 0,
max_tokens 300 (the Llama collectors' MAX_NEW_TOKENS). Writes responses.jsonl {set, pair_id, response_text, n_prompt_tokens}.
usage: python qwen_gen.py MODEL inputs.jsonl responses.jsonl [--test]"""
import json, sys, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
def main():
    model, inp, out = sys.argv[1:4]; test = "--test" in sys.argv
    rows = [json.loads(l) for l in open(inp)]
    if test:
        import collections; c = collections.Counter(); rows = [r for r in rows if c.update([r["set"]]) or c[r["set"]] <= 4]
    tok = AutoTokenizer.from_pretrained(model)
    txt = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    print("example:", repr(txt[0][-200:]), flush=True)
    ids = [tok(t, add_special_tokens=False).input_ids for t in txt]
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
    t0 = time.time()
    res = llm.generate([{"prompt_token_ids": i} for i in ids], SamplingParams(temperature=0.0, max_tokens=300))
    print(f"generated {len(res)} in {time.time()-t0:.0f}s", flush=True)
    with open(out, "w") as f:
        for r, i, o in zip(rows, ids, res):
            f.write(json.dumps({"set": r["set"], "pair_id": r["pair_id"], "response_text": o.outputs[0].text, "n_prompt_tokens": len(i)}, ensure_ascii=False) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
