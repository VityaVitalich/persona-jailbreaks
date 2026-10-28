#!/usr/bin/env python3
"""Gemma steering arms: the refusal and jailbreak directions of the Gemma steering bundle (refusal_pos1_neg__full,
dmu__full; layer 30, both oriented toward jailbroken) split into their persona component v_in (projection onto the
span of the 240 trait vectors) and the remainder v_out, all unit norm. v_in / v_out are oriented toward jailbroken by
the sign of the mean projection gap (jailbroken - refused) on HarmBench human-jailbreak + WildJailbreak, the bundle's
rule; flips printed. Also prints share, cosine with the full direction, and folded HB / transfer AUC (h.v).
Writes steer_split_gemma_layer30.pt (steer_fixed_direction_gemma.py format)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score
G = Path(__file__).resolve().parent.parent; O = G.parent.parent; sys.path.insert(0, str(O / "trait_pc_sweep"))
import trait_pc_sweep as S  # noqa: E402
cfg = S.CFG["gemma"]; unit = lambda v: v / np.linalg.norm(v); fa = lambda y, s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
T = np.load(cfg["T"]).astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True); Qp = np.linalg.qr(T.T)[0]
B = torch.load(O / "gemma_steering_inputs/steering_directions_gemma_layer30.pt", weights_only=False)
hb = [r for r in S.load_jsonl(cfg["human"][0]) if r.get("attack_type") == "human_jailbreak"]
Xh, yh, _ = S.build_matrix(hb, torch.load(cfg["human"][1], map_location="cpu", weights_only=False), 30)
Xa, ya, _ = S.build_matrix(S.load_jsonl(cfg["aug"][0]), torch.load(cfg["aug"][1], map_location="cpu", weights_only=False), 30)
XF = np.vstack([Xh, Xa]).astype(np.float64); yF = np.r_[yh, ya]
fam = {f: S.build_matrix(S.load_jsonl(rp), torch.load(ap, map_location="cpu", weights_only=False), 30)[:2] for f, (rp, ap) in cfg["fams"].items()}
gap = lambda v: float((XF[yF == 1] @ v).mean() - (XF[yF == 0] @ v).mean())
vec, meta = {}, {}
print(f"{'arm':22s} {'share':>6s} {'cos full':>8s} {'flip':>5s} {'gap':>7s} {'HB AUC':>7s} {'xfer':>6s}")
for src, short in (("refusal_pos1_neg__full", "refusal"), ("dmu__full", "dmu")):
    v = unit(B["vectors"][src].double().numpy()); assert gap(v) > 0, src
    pin = Qp @ (Qp.T @ v); sh = float(pin @ pin)
    for n, w in ((short, v), (f"{short}_vin", unit(pin)), (f"{short}_vout", unit(v - pin))):
        fl = gap(w) < 0; w = -w if fl else w
        a = fa(yh, Xh @ w); x = float(np.mean([fa(y, X @ w) for X, y in fam.values()]))
        print(f"{n:22s} {sh:6.3f} {w @ v:8.3f} {str(fl):>5s} {gap(w):7.2f} {a:7.3f} {x:6.3f}")
        vec[n] = torch.tensor(w, dtype=torch.float32)
        meta[n] = {"source": src, "part": n.split("_")[-1] if "_v" in n else "full", "persona_share_of_source": sh,
                   "cos_source": float(w @ v), "sign_flipped": bool(fl), "mean_gap_hbwjb": gap(w), "hb_auc_folded": a, "transfer_auc_folded": x}
torch.save({"vectors": vec, "meta": meta, "layer": 30, "model": B.get("model"),
            "note": "refusal / dmu split into persona component and remainder, gemma/steer/build_steer_split_gemma.py"}, Path(__file__).resolve().parent / "steer_split_gemma_layer30.pt")
