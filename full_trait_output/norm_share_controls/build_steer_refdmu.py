#!/usr/bin/env python3
"""
Refusal and jailbreak steering directions with their persona component removed (official-snapshot runs).
  refusal            -(Arditi refusal direction), layer 16 pos -1        (steer_controls_v2_layer16.pt, unchanged)
  dmu                mean(jailbroken) - mean(refused), all HarmBench pairs (steer_controls_v2_layer16.pt, unchanged)
  <x>_nopersona      <x> minus its projection onto the span of the 229 trait vectors, renormalized
  <x>_nonr           <x> minus its projection onto the span of 229 No Robots activations (nr_d4) instead
All unit norm. Prints the share removed and the folded HarmBench / transfer AUC of each vector as h.v.
Writes steer_refdmu_layer16.pt (steer_fixed_direction.py format).
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(OUT / "trait_pc_sweep")); sys.path.insert(0, str(HERE))
import trait_pc_sweep as S  # noqa: E402
from few_directions import rank_auc  # noqa: E402
unit = lambda v: v / np.linalg.norm(v)
cfg = S.CFG["llama"]
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True)
Qp = np.linalg.qr(T.T)[0]; Qn = np.linalg.qr(np.load(HERE / "extra_bases_llama.npz")["nr_d4"].astype(np.float64).T)[0]
old = torch.load(HERE / "steer_controls_v2_layer16.pt", weights_only=False)
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 16)
fam = {f: S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 16)[:2] for f, (rp, ap) in cfg["fams"].items()}
auc = lambda v, X, y: rank_auc(np.asarray(y), (np.asarray(X, np.float64) @ v)[:, None])[0]
fold = lambda a: max(a, 1 - a)
vec, meta = {}, {}
print(f"{'arm':20s} {'share removed':>13s} {'cos w/ orig':>11s} {'HB AUC':>7s} {'xfer':>6s}")
for n in ("refusal", "dmu"):
    v = old["vectors"][n].double().numpy()
    for tag, Q in (("", None), ("_nopersona", Qp), ("_nonr", Qn)):
        w = v if Q is None else unit(v - Q @ (Q.T @ v)); sh = 0.0 if Q is None else float(np.sum((Q.T @ v) ** 2))
        a = auc(w, Xh, yh); assert a > 0.5, n + tag  # oriented toward jailbroken
        xf = float(np.mean([fold(auc(w, X, y)) for X, y in fam.values()]))
        vec[n + tag] = torch.tensor(w, dtype=torch.float32)
        meta[n + tag] = {**old["meta"][n], "removed": tag or None, "share_removed": sh, "hb_auc_toward_jb_allrows": float(a), "transfer_auc_folded": xf}
        print(f"{n + tag:20s} {sh:13.3f} {w @ v:11.3f} {a:7.3f} {xf:6.3f}")
torch.save({"vectors": vec, "meta": meta, "layer": 16, "note": "refusal / dmu persona-removal arms, build_steer_refdmu.py"}, HERE / "steer_refdmu_layer16.pt")
