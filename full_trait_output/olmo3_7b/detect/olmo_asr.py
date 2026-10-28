"""ASR (% jailbroken, GPT-4.1-mini judge labels; HB = human-jailbreak templates only, parse failures dropped) per set for
Llama / Gemma (shared data, via qwen3_32b/paper_appendix/refusal_rates.jsonl), Qwen3-32B (the appendix label file) and
OLMo-3-7B-Instruct (olmo3_7b/transfer/classified_*.jsonl). Prints a markdown table, models as rows; writes olmo_asr.json."""
import json, numpy as np
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; TR = "/Users/viktor/persona-vector-jbing/full_trait_output/olmo3_7b/transfer"
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]; key = {"harmbench": "hb", "wildjailbreak": "wjb"}
asr, n = {}, {}
for r in map(json.loads, open(f"{Q}/paper_appendix/refusal_rates.jsonl")):
    m = "llama" if r["model"].startswith("llama") else "gemma"; s = key.get(r["set"], r["set"]); asr[(m, s)] = 100 * r["jailbroken"]; n[(m, s)] = r["n"]
A = np.load(f"{Q}/transfer/diag/scores_audit_L48_last.npz")
for s in SETS: asr[("qwen", s)] = 100 * A[f"labels_{s}"].mean(); n[("qwen", s)] = len(A[f"labels_{s}"])
for s in SETS:
    try: R = [r for r in map(json.loads, open(f"{TR}/classified_{s}.jsonl")) if r["jailbroken"] is not None and (s != "hb" or r["attack_type"] == "human_jailbreak")]
    except FileNotFoundError: continue
    asr[("olmo", s)] = 100 * np.mean([r["jailbroken"] for r in R]); n[("olmo", s)] = len(R)
MOD = {"llama": "Llama-3.1-8B", "gemma": "Gemma-4-31B", "qwen": "Qwen3-32B", "olmo": "OLMo-3-7B"}
print("| model | " + " | ".join(SETS) + " |\n|---|" + "---:|" * len(SETS))
for m in MOD: print(f"| {MOD[m]} | " + " | ".join(f"{asr[(m, s)]:.1f} (n={n[(m, s)]})" if (m, s) in asr else "–" for s in SETS) + " |")
json.dump({f"{m}__{s}": {"asr": asr[(m, s)], "n": n[(m, s)]} for (m, s) in asr}, open("olmo_asr.json", "w"), indent=1)
