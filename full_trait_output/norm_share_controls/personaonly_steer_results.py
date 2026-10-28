#!/usr/bin/env python3
"""
HarmBench 300-pool ASR of the persona-only steering arms (build_steer_personaonly.py) against the full direction and
the persona-removed direction on the same prompts (official snapshot). Paired differences with 95% intervals from a
bootstrap over behaviours (all prompts of a behaviour resampled together, 2000 draws). Intact refusal and jailbreak
directions were not rerun on the official snapshot, so for them only persona-only vs persona-removed is paired.
Writes steer_bench/personaonly_harmbench.json and personaonly_table.tex.
"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SB, OLD = HERE / "steer_bench", HERE / "steer_v2_results"
F = {"alpacajb_d2": OLD / "alpaca1_alpacajb_d2.jsonl", "alpacajb_d2_nopersona": OLD / "nopersona_det_alpacajb_d2_nopersona.jsonl",
     "sel_pca_1": OLD / "alpaca2_sel_pca_1.jsonl", "sel_pca_1_nopersona": OLD / "nopersona_sel_sel_pca_1_nopersona.jsonl"}
for a in ("persona", "random_full_s0", "refusal_nopersona", "dmu_nopersona", "refusal_personaonly", "dmu_personaonly",
          "alpacajb_d2_personaonly", "sel_pca_1_personaonly"):
    F[a] = SB / f"harmbench_{a}.jsonl"
L = {a: {json.loads(l)["pair_id"]: (json.loads(l)["behavior_id"], float(json.loads(l)["final_jailbroken"])) for l in open(p)} for a, p in F.items()}
ids = sorted(L["persona"]); assert all(set(v) == set(ids) for v in L.values())
beh = np.array([L["persona"][i][0] for i in ids]); ub = np.unique(beh); idx = [np.where(beh == b)[0] for b in ub]
y = {a: np.array([v[i][1] for i in ids]) for a, v in L.items()}
rng = np.random.default_rng(0); B = [np.concatenate([idx[j] for j in rng.integers(0, len(ub), len(ub))]) for _ in range(2000)]
def diff(a, b):
    d = y[a] - y[b]; bs = [100 * d[s].mean() for s in B]
    return dict(diff=100 * d.mean(), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
res = {"asr": {a: 100 * v.mean() for a, v in y.items()}, "paired": {}}
for po, full, rem in (("alpacajb_d2_personaonly", "alpacajb_d2", "alpacajb_d2_nopersona"), ("sel_pca_1_personaonly", "sel_pca_1", "sel_pca_1_nopersona"),
                      ("refusal_personaonly", None, "refusal_nopersona"), ("dmu_personaonly", None, "dmu_nopersona")):
    res["paired"][po] = {"vs_full": diff(po, full) if full else None, "vs_removed": diff(po, rem), "vs_random": diff(po, "random_full_s0")}
json.dump(res, open(SB / "personaonly_harmbench.json", "w"), indent=1)
for a, v in res["asr"].items(): print(f"{a:26s} {v:5.1f}")
for k, v in res["paired"].items():
    print(k, {kk: (f"{x['diff']:+.1f} [{x['lo']:+.1f},{x['hi']:+.1f}]" if x else None) for kk, x in v.items()})
