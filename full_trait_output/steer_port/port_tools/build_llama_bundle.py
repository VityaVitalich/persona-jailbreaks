# steer_traits_llama_layer16.pt for the Llama figure run (official snapshot, L16, norm 22 = runner default):
#   persona, random_full_s0  copied from steer_bench_layer16.pt (identical in every Llama bundle)
#   trait_<t>  positive pole, neg_<t> negative pole: llama_traits.npy layer 16 rows, unit (checked against Dominic's matrix)
#   = the transfer top-10 predictive traits (toward the jailbreak side) + the top-10 of Dominic's all-229 sweep (positive pole)
import json, os, numpy as np, torch
U = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb"; P = f"{U}/steer_port"
TA = json.load(open(f"{U}/q1_detection/out/trait_transfer_auc.json"))["llama"]["traits"]
R = np.load(f"{U}/qwen3_32b/probe_methods/llama/trait_matrix_layer16.npy").astype(np.float64); RN = json.load(open(f"{U}/qwen3_32b/probe_methods/llama/trait_names_layer16.json"))
T = np.asarray(np.load(f"{U}/hb_select/llama_traits.npy", mmap_mode="r")[:, 16], np.float64)
cs = (T / np.linalg.norm(T, axis=1, keepdims=True)) @ (R / np.linalg.norm(R, axis=1, keepdims=True)).T; j = cs.argmax(1)
assert (cs.max(1) > 0.999).all(); names = [RN[k] for k in j]
pred = sorted(TA, key=lambda t: -TA[t]["transfer_avg"])[:10]
steer = [r["trait"] for r in sorted(json.load(open(f"{P}/inputs/gemma/alltraits/single_trait_all229_ranking.json")), key=lambda r: -r["asr_harmbench"])[:10]]
base = torch.load(f"{P}/inputs/steer_bench_layer16.pt", map_location="cpu", weights_only=False)["vectors"]
vec = {"persona": base["persona"].float(), "random_full_s0": base["random_full_s0"].float()}
arms_pred, arms_steer = [], []
for t in pred:
    k = f"{'trait' if TA[t]['sign'] > 0 else 'neg'}_{t}"; arms_pred.append(k)
for t in steer: arms_steer.append(f"trait_{t}")
for k in dict.fromkeys(arms_pred + arms_steer):
    t = k.split("_", 1)[1]; v = R[RN.index(t)] / np.linalg.norm(R[RN.index(t)]); vec[k] = torch.tensor(-v if k.startswith("neg_") else v, dtype=torch.float32)
for k, v in vec.items(): assert abs(float(v.norm()) - 1) < 1e-4, k
out = f"{P}/inputs/steer_traits_llama_layer16.pt"; assert not os.path.exists(out)
torch.save({"vectors": vec, "layer": 16, "meta": {k: {"description": k} for k in vec}}, out)
json.dump({"pred": arms_pred, "steer": arms_steer}, open(f"{P}/inputs/llama_traits_spec.json", "w"), indent=1)
print("arms", len(vec), "| pred", arms_pred, "| steer", arms_steer)
