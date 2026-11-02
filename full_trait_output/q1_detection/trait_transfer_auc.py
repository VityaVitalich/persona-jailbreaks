"""Single-trait predictiveness on the q1 protocol's sets: for every trait vector t (unit, at the model's STEERING layer)
the score is h.t at the last prompt token (tail position 4); folded AUC per set, averaged over the 5 transfer sets
(PAP, GCG, PAIR, GPTFuzz, PEZ); also HB (human_jailbreak rows, as q1) and the sign (+ = high score -> jailbreak on HB).
No fitting. Trait names: matched to a named reference matrix by cosine (asserted > 0.999), OLMo = sorted .pt basenames.
usage: python3 trait_transfer_auc.py OUT.json"""
import json, glob, os, sys, numpy as np
U = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb"; Q = f"{U}/q1_detection"
STEER_L = {"llama": 16, "olmo": 16, "gemma": 30}; POS = 4
REF = {"llama": (f"{U}/qwen3_32b/probe_methods/llama/trait_matrix_layer16.npy", f"{U}/qwen3_32b/probe_methods/llama/trait_names_layer16.json"),
       "gemma": (f"{U}/steer_port/inputs/gemma/alltraits/trait_matrix_layer30.npy", f"{U}/steer_port/inputs/gemma/alltraits/trait_names_layer30.json")}
def auc(y, s):
    r = np.empty(len(s)); o = np.argsort(s, kind="mergesort"); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s[np.r_[False, np.diff(s[o]) == 0]] if len(s) else []):  # average ties
        m = s == v; r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
out = {}
for m, L in STEER_L.items():
    c = json.load(open(f"{Q}/configs/q1_{m}_hbwjb.json")); meta = {s: json.load(open(c["meta"].format(set=s))) for s in ["hb"] + c["test"]}
    li = meta["hb"]["layers"].index(L); T = np.asarray(np.load(c["traits"], mmap_mode="r")[:, li], np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
    if m in REF:
        R = np.load(REF[m][0]).astype(np.float64); R /= np.linalg.norm(R, axis=1, keepdims=True); nm = json.load(open(REF[m][1]))
        cs = T @ R.T; j = cs.argmax(1); assert (cs.max(1) > 0.999).all() and len(set(j)) == len(j), (m, cs.max(1).min()); names = [nm[k] for k in j]
    else:
        names = [os.path.basename(p)[:-3] for p in sorted(glob.glob(f"{U}/olmo3_7b/trait_vectors/all_traits_no_filter/*.pt"))]
    assert len(names) == len(T), (m, len(names), len(T))
    res = {n: {} for n in names}
    for s in ["hb"] + c["test"]:
        C = {r["pair_id"]: r for r in map(json.loads, open(c["labels"][s]))}; ids = meta[s]["pair_ids"]
        keep = [i for i, p in enumerate(ids) if p in C and C[p]["jailbroken"] is not None and (s != "hb" or C[p]["attack_type"] == "human_jailbreak")]
        y = np.array([int(C[ids[i]]["jailbroken"]) for i in keep]); X = np.asarray(np.load(c["acts"].format(set=s, layer=L), mmap_mode="r")[keep, POS], np.float64)
        S = X @ T.T
        for k, n in enumerate(names):
            a = auc(y, S[:, k]); res[n][s] = max(a, 1 - a); res[n][s + "_signed"] = a
    for n in names:
        res[n]["transfer_avg"] = float(np.mean([res[n][s] for s in c["test"]])); res[n]["sign"] = 1 if res[n]["hb_signed"] >= 0.5 else -1
    out[m] = {"layer": L, "position": POS, "token": meta["hb"]["tail_tokens"][POS], "sets": c["test"], "traits": res}
    print(m, L, len(names), "top transfer:", sorted(names, key=lambda n: -res[n]["transfer_avg"])[:5], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
