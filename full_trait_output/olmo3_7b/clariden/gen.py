#!/usr/bin/env python3
"""Greedy OLMo-3-7B-Instruct responses (vLLM, serving image) for one set of the inputs. Single user turn through the model's
chat template (which fills its default system prompt), add_generation_prompt; temperature 0, max_tokens 300 (the Llama
collectors' MAX_NEW_TOKENS); stops at <|im_end|> as well as <|endoftext|>. Atomic write.
usage: gen.py MODEL inputs.jsonl SET OUT [--test]"""
import json, os, sys, time
def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    model, inp, st, out = sys.argv[1:5]; test = "--test" in sys.argv
    rows = [r for r in map(json.loads, open(inp)) if r["set"] == st][: (4 if test else None)]
    tok = AutoTokenizer.from_pretrained(model)
    txt = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in rows]
    print("template tail:", repr(txt[0][-40:]), flush=True)
    ids = [tok(t, add_special_tokens=False).input_ids for t in txt]
    print(f"{st}: {len(rows)} prompts, max prompt tokens {max(map(len, ids))}", flush=True)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=0)
    stop = tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    t0 = time.time(); res = llm.generate([{"prompt_token_ids": i} for i in ids], SamplingParams(temperature=0.0, max_tokens=300, stop_token_ids=stop))
    print(f"generated {len(res)} in {time.time()-t0:.0f}s", flush=True)
    with open(out + ".tmp", "w") as f:
        for r, i, o in zip(rows, ids, res):
            f.write(json.dumps({"set": st, "pair_id": r["pair_id"], "response_text": o.outputs[0].text, "n_prompt_tokens": len(i),
                                "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out); print("done", flush=True)
if __name__ == "__main__":
    main()
