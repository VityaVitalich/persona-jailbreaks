# OLMo: steer_traits_olmo_layer16.pt = all 240 trait rows (olmo_traits.npy at layer 16, unit, positive pole) as trait_<t>,
#       plus neg_<t> for the transfer-top-10 traits whose low score predicts jailbreak.
# Gemma: steer_traits572b_gemma_layer30.pt = the transfer-top-10 traits not yet run at norm 572, signed toward jailbreak.
# Checks: OLMo rows equal the existing olmo bundle's trait_accommodating / trait_introverted (up to its sign).
import glob, json, os, numpy as np, torch
U = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb"; P = f"{U}/steer_port"
TA = json.load(open(f"{U}/q1_detection/out/trait_transfer_auc.json"))
top = lambda m: sorted(TA[m]["traits"], key=lambda t: -TA[m]["traits"][t]["transfer_avg"])[:10]
# ---- OLMo
names = [os.path.basename(p)[:-3] for p in sorted(glob.glob(f"{U}/olmo3_7b/trait_vectors/all_traits_no_filter/*.pt"))]
T = np.asarray(np.load(f"{U}/hb_select/olmo_traits.npy", mmap_mode="r")[:, [4, 8, 12, 16, 20, 24, 28, 31].index(16)], np.float64)
T /= np.linalg.norm(T, axis=1, keepdims=True)
old = torch.load(f"{P}/inputs/olmo/steering_directions_olmo_layer16.pt", map_location="cpu", weights_only=False)["vectors"]
for t in ("accommodating", "introverted", "paranoid"):
    c = float(np.dot(T[names.index(t)], old[f"trait_{t}"].double().numpy())); print("olmo check", t, round(c, 4)); assert abs(abs(c) - 1) < 1e-3
vec = {f"trait_{t}": torch.tensor(T[i], dtype=torch.float32) for i, t in enumerate(names)}
neg = [t for t in top("olmo") if TA["olmo"]["traits"][t]["sign"] < 0]
for t in neg: vec[f"neg_{t}"] = -vec[f"trait_{t}"]
out = f"{P}/inputs/olmo/steer_traits_olmo_layer16.pt"; assert not os.path.exists(out)
torch.save({"vectors": vec, "layer": 16, "meta": {k: {"description": ("negative pole of " if k.startswith("neg_") else "") + k.split("_", 1)[1]} for k in vec}}, out)
print("olmo bundle", len(vec), "neg:", neg, "| top-10 transfer:", [(t, TA["olmo"]["traits"][t]["sign"]) for t in top("olmo")])
# ---- Gemma: transfer top-10 minus what outputs_gemma_norm572_traits already has (pred_<t>)
have = {d[5:] for b in ("harmbench", "jbb", "wildjb") for d in os.listdir(f"{P}/outputs_gemma_norm572_traits/{b}") if d.startswith("pred_")}
spec = json.load(open(f"{P}/inputs/gemma/traits572_spec.json")); hsign = dict(spec["pred"])
M = np.load(f"{P}/inputs/gemma/alltraits/trait_matrix_layer30.npy"); gn = json.load(open(f"{P}/inputs/gemma/alltraits/trait_names_layer30.json"))
gv, info = {}, []
for t in top("gemma"):
    sg = TA["gemma"]["traits"][t]["sign"]
    if t in have: assert hsign[t] == sg, (t, hsign[t], sg); continue
    v = sg * M[gn.index(t)].astype(np.float64); gv[f"pred_{t}"] = torch.tensor(v / np.linalg.norm(v), dtype=torch.float32); info.append((t, sg))
out = f"{P}/inputs/gemma/steer_traits572b_gemma_layer30.pt"; assert not os.path.exists(out)
torch.save({"vectors": gv, "layer": 30, "meta": {k: {"description": "transfer-top-10 predictive trait, toward the jailbreak side"} for k in gv}}, out)
print("gemma new pred arms:", info)
