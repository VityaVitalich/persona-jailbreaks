"""Inputs for hb_select.py on clariden (run in the harmbench image):
  qwen: trait npy [240, 16, 5120] from the Assistant Axis vectors ([64, 5120] each, rows at the sweep layers), names in
        trait_names_layer32.json order; activations = the existing sweep/acts/{set}_L{l}.npy (9 tail tokens + user mean).
  olmo: per-layer npy [n, 5, 4096] + meta from transfer/acts_{set}.pt; trait npy [240, 8, 4096] (rows at the saved layers).
Writes configs hb_select_{model}.json next to the outputs."""
import glob, json, os, numpy as np, torch
U = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/persona_jb"; H = f"{U}/hb_select"; os.makedirs(f"{H}/olmo_acts", exist_ok=True)
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]
# qwen
Q = f"{U}/qwen3_32b"; layers = json.load(open(f"{Q}/sweep/acts/hb_meta.json"))["layers"]
names = json.load(open(f"{Q}/sweep/in/trait_names_layer32.json"))
T = np.stack([torch.load(f"{Q}/sweep/in/trait_vectors/{n}.pt", map_location="cpu")[layers].float().numpy() for n in names]); np.save(f"{H}/qwen_traits.npy", T)
json.dump({"acts": f"{Q}/sweep/acts/{{set}}_L{{layer}}.npy", "meta": f"{Q}/sweep/acts/{{set}}_meta.json", "traits": f"{H}/qwen_traits.npy",
           "labels": {s: (f"{Q}/sweep/in/classified_{s}.jsonl" if s in ("hb", "wjb", "pap") else f"{Q}/transfer/classified_{s}.jsonl") for s in SETS},
           "train": ["hb"], "test": [s for s in SETS if s != "hb"]}, open(f"{H}/hb_select_qwen.json", "w"), indent=1)
print("qwen traits", T.shape, flush=True)
# olmo
O = f"{U}/olmo3_7b"
for s in SETS:
    A = torch.load(f"{O}/transfer/acts_{s}.pt")
    for li, l in enumerate(A["layers"]): np.save(f"{H}/olmo_acts/{s}_L{l}.npy", A["X"][:, li].numpy())
    json.dump({"pair_ids": A["pair_ids"], "tail_tokens": A["tail_tokens"], "layers": A["layers"]}, open(f"{H}/olmo_acts/{s}_meta.json", "w"))
    ol = A["layers"]; print("olmo", s, tuple(A["X"].shape), flush=True)
P = sorted(glob.glob(f"{O}/trait_vectors/all_traits_no_filter/*.pt"))
T = np.stack([torch.load(p, map_location="cpu", weights_only=False)["vector"][ol].float().numpy() for p in P]); np.save(f"{H}/olmo_traits.npy", T)
json.dump({"acts": f"{H}/olmo_acts/{{set}}_L{{layer}}.npy", "meta": f"{H}/olmo_acts/{{set}}_meta.json", "traits": f"{H}/olmo_traits.npy",
           "labels": {s: f"{O}/transfer/classified_{s}.jsonl" for s in SETS}, "train": ["hb"], "test": [s for s in SETS if s != "hb"]}, open(f"{H}/hb_select_olmo.json", "w"), indent=1)
print("olmo traits", T.shape, flush=True)
