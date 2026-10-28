"""Validate + time the fold-based GQA attention against HF's sdpa path on Gemma-4-31B.
One model load; the attention implementation is switched at runtime."""
import json, os, sys, time, torch
N_CORR=int(os.environ.get("VAL_N_CORR","8")); BS_LIST=[int(x) for x in os.environ.get("VAL_BS_LIST","32,64,96").split(",") if x]; SHORT_BS=int(os.environ.get("VAL_SHORT_BS","96")); N_NEW=int(os.environ.get("VAL_N_NEW","300"))
sys.path.insert(0, "/mnt/dlabscratch1/moskvore/persona_steer/full_trait_tools")
from gqa_fold_attention import register; register()
from chunked_sdpa import chunked_sdpa_scope
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "/mnt/dlabscratch1/bazina/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f"
tpl = {int(k): v for k, v in json.load(open("/mnt/dlabscratch1/moskvore/persona_steer/inputs/gemma/hb_templates.json")).items()}
rows = [json.loads(l) for l in open("/mnt/dlabscratch1/moskvore/persona_steer/inputs/gemma/hb_classified_responses.jsonl")]
pool = sorted([tpl[r["jailbreak_idx"]].format(r["behavior_text"]) for r in rows if r["attack_type"] == "human_jailbreak" and r["jailbroken"] is False], key=len)
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True); tok.padding_side = "left"
def enc(prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p.encode("utf-8", "replace").decode("utf-8", "replace")}], tokenize=False, add_generation_prompt=True) for p in prompts]
    return tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"}, trust_remote_code=True).eval()
print("loaded; attn =", model.config._attn_implementation, flush=True)
def gen(e, n):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad(), chunked_sdpa_scope():
        out = model.generate(**e, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); return out[:, e["input_ids"].shape[1]:], time.perf_counter() - t
# 1) correctness: 8 prompts spanning lengths, 64 new tokens, sdpa vs fold
mix = [pool[i] for i in [0, len(pool)//4, len(pool)//2, 3*len(pool)//4, -1, -2, -3, -4][:N_CORR]]
e = enc(mix); print(f"correctness batch: {len(mix)} prompts, in_len={e['input_ids'].shape[1]}", flush=True)
ref, t_ref = gen(e, 64)
model.set_attn_implementation("gqa_fold"); print("switched; attn =", model.config._attn_implementation, flush=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
new, t_new = gen(e, 64)
agree = [(int((ref[i] == new[i]).sum()), int((ref[i] != new[i]).nonzero()[0, 0]) if (ref[i] != new[i]).any() else 64) for i in range(len(mix))]
print(f"correctness (8 prompts, in_len={e['input_ids'].shape[1]}, 64 tokens): tokens equal / first divergence:", agree, flush=True)
print("  sample sdpa :", repr(tok.decode(ref[-1], skip_special_tokens=True)[:150])); print("  sample fold :", repr(tok.decode(new[-1], skip_special_tokens=True)[:150]), flush=True)
print(f"  time sdpa {t_ref:.1f}s vs fold {t_new:.1f}s | fold peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
# 2) throughput: longest 32 prompts, 300 tokens, both implementations; then fold at 64 and 96
b0 = BS_LIST[0]; long0 = enc(pool[-b0:])
for impl in ["gqa_fold", "sdpa"]:
    model.set_attn_implementation(impl)
    o, dt = gen(long0, N_NEW); print(f"{impl:9s} B={b0} long (in_len {long0['input_ids'].shape[1]}) {N_NEW} tok: {dt:.1f}s -> {b0/dt*3600:.0f} gen/h, {o.numel()/dt:.0f} tok/s | peak mem {torch.cuda.max_memory_allocated()/2**30:.0f} GiB", flush=True)
model.set_attn_implementation("gqa_fold")
for bs in BS_LIST[1:]:
    e2 = enc(pool[-bs:]); o, dt = gen(e2, N_NEW); print(f"gqa_fold  B={bs} long (in_len {e2['input_ids'].shape[1]}) {N_NEW} tok: {dt:.1f}s -> {bs/dt*3600:.0f} gen/h | peak mem {torch.cuda.max_memory_allocated()/2**30:.0f} GiB", flush=True)
short = enc(pool[:SHORT_BS]); o, dt = gen(short, N_NEW); print(f"gqa_fold  B={SHORT_BS} short (in_len {short['input_ids'].shape[1]}) {N_NEW} tok: {dt:.1f}s -> {SHORT_BS/dt*3600:.0f} gen/h", flush=True)
