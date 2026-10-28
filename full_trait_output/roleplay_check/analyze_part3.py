#!/usr/bin/env python3
"""
Part 3: can the PersonaJB score tell apart (i) jailbroken = harmful request + model complied,
(ii) refused = harmful request + model refused, (iii) benign-complied = benign request + model
complied? Uses part2_scores.npz (50 seeds) and synthetic_judged.jsonl.
Pairwise folded AUC max(a,1-a), mean over seeds, 95% CI = bootstrap over JailbreakBench behaviours.
Strata: all prompts, plain only, role-play only. Writes part3_results.json and part3_tables.md.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from analyze_part1 import auc_rows

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
GROUPS = ["jailbroken", "refused", "benign_complied"]


def group(r, complied):
    if complied is None: return None
    if r["harm"] == "harmful": return "jailbroken" if complied else "refused"
    return "benign_complied" if complied else "benign_refused"


def pair_auc(S, g, beh, a, b, B=1000):
    m = (g == a) | (g == b); y = (g[m] == a); Sm = S[:, m]; bm = beh[m]
    if y.sum() == 0 or (~y).sum() == 0: return None
    point = float(np.nanmean(auc_rows(y, Sm)))
    ub = np.unique(bm); idx = {u: np.where(bm == u)[0] for u in ub}; vals = []
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])
        if 0 < y[ii].sum() < len(ii): vals.append(np.nanmean(auc_rows(y[ii], Sm[:, ii])))
    lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    if point < .5: point, lo, hi = 1 - point, 1 - hi, 1 - lo
    return {"auc": point, "ci": [float(lo), float(hi)], "n_a": int(y.sum()), "n_b": int((~y).sum())}


def main():
    d = np.load(HERE / "part2_scores.npz", allow_pickle=True)
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "synthetic_prompts.jsonl")}
    judged = {json.loads(l)["id"]: json.loads(l)["complied"] for l in open(HERE / "synthetic_judged.jsonl")}
    ids = list(d["ids"]); S = d["scores"]
    rows = [meta[i] for i in ids]
    g = np.array([group(r, judged.get(i)) for r, i in zip(rows, ids)], dtype=object)
    beh = np.array([r["jbb_index"] for r in rows]); rp = np.array([r["frame_type"] != "plain" for r in rows])
    counts = {s: dict(Counter(g[m])) for s, m in [("all", np.ones(len(g), bool)), ("plain", ~rp), ("roleplay", rp)]}
    res = {"counts": counts, "mean_score": {k: float(S[:, g == k].mean()) for k in GROUPS + ["benign_refused"] if (g == k).any()}}
    for s, m in [("all", np.ones(len(g), bool)), ("plain", ~rp), ("roleplay", rp)]:
        res[s] = {f"{a}__vs__{b}": pair_auc(S[:, m], g[m], beh[m], a, b)
                  for i, a in enumerate(GROUPS) for b in GROUPS[i + 1:]}
    json.dump(res, open(HERE / "part3_results.json", "w"), indent=1)
    f = lambda o: "--" if o is None else f"{o['auc']:.2f} [{o['ci'][0]:.2f}, {o['ci'][1]:.2f}] (n={o['n_a']}/{o['n_b']})"
    L = ["| Pair | All prompts | Plain | Role-play |", "|---|---|---|---|"]
    for k in res["all"]:
        L.append(f"| {k.replace('__vs__', ' vs ')} | {f(res['all'][k])} | {f(res['plain'][k])} | {f(res['roleplay'][k])} |")
    L += ["", "Counts: " + json.dumps(counts), "Mean P(jb): " + json.dumps({k: round(v, 3) for k, v in res["mean_score"].items()})]
    open(HERE / "part3_tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
