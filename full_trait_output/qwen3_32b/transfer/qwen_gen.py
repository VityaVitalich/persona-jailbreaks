#!/usr/bin/env python3
"""Greedy Qwen3-32B responses (vLLM) for one set of qwen_inputs.jsonl. Single user turn, no system prompt,
add_generation_prompt, enable_thinking=False (Assistant Axis recipe for Qwen); temperature 0, max_tokens 300 (the Llama
collectors' MAX_NEW_TOKENS). Atomic write: OUT.tmp then rename. usage: qwen_gen.py MODEL inputs.jsonl SET OUT [--test]"""
import json, os, sys, time
def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    model, inp, st, out = sys.argv[1:5]; test = "--test" in sys.argv
    rows = [r for r in map(json.loads, open(inp)) if r["set"] == st][: (4 if test else None)]
    tok = AutoTokenizer.from_pretrained(model)
    txt = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    print("example prompt tail:", repr(txt[0][-120:]), flush=True)
    ids = [tok(t, add_special_tokens=False).input_ids for t in txt]
    print(f"{st}: {len(rows)} prompts, max prompt tokens {max(map(len, ids))}", flush=True)
    # flashinfer JIT fails on B300 (sm_103a) -> Triton attention unless QWEN_ATTN overrides
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0,
              attention_backend=os.environ.get("QWEN_ATTN", "TRITON_ATTN"))
    t0 = time.time(); res = llm.generate([{"prompt_token_ids": i} for i in ids], SamplingParams(temperature=0.0, max_tokens=300))
    print(f"generated {len(res)} in {time.time()-t0:.0f}s", flush=True)
    with open(out + ".tmp", "w") as f:
        for r, i, o in zip(rows, ids, res):
            f.write(json.dumps({"set": st, "pair_id": r["pair_id"], "response_text": o.outputs[0].text, "n_prompt_tokens": len(i),
                                "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out); print("done", flush=True)
if __name__ == "__main__":
    main()
