#!/usr/bin/env python3
"""
v2 three-group check: does the PersonaJB score separate
  jailbroken       harmful request, model answered (OR-Bench judge: direct_answer)
  refused          harmful request, model refused (direct_ or indirect_refusal)
  benign_complied  benign request, model answered
  (benign_refused  benign request, model refused; reported, small)
Detectors: the 50 published PersonaJB refits (analyze_part2.fit_scores). Separation = AUC folded to
max(a, 1-a) (direction is irrelevant here), mean over the 50 detectors; 95% CI = bootstrap over
requests (a request's plain and framed prompts stay together), 1000 resamples.
Strata: all prompts / plain / role-play / frame-matched (AUC computed inside each frame, then
averaged over frames weighted by the number of cross-group pairs, so frame offsets cannot help or hurt).
Writes v2_scores.npz (every prompt x seed), v2_results.json, v2_tables.md.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np, torch
from analyze_part1 import auc_rows
from analyze_part2 import fit_scores

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
G = ["jailbroken", "refused", "benign_complied", "benign_refused"]
PAIRS = [("jailbroken", "refused"), ("jailbroken", "benign_complied"), ("refused", "benign_complied"),
         ("benign_complied", "benign_refused"), ("refused", "benign_refused")]
fold = lambda a: np.where(a < .5, 1 - a, a)


def sep(S, g, a, b):
    m = (g == a) | (g == b); y = g[m] == a
    if y.sum() < 5 or (~y).sum() < 5: return np.nan
    return float(fold(np.nanmean(auc_rows(y, S[:, m]))))


def framematched(S, g, fr, a, b):
    num = den = 0.0
    for f in np.unique(fr):
        m = fr == f; na, nb = (g[m] == a).sum(), (g[m] == b).sum()
        if na >= 5 and nb >= 5:
            v = np.nanmean(auc_rows(g[m][(g[m] == a) | (g[m] == b)] == a, S[:, m][:, (g[m] == a) | (g[m] == b)]))
            num += v * na * nb; den += na * nb
    return float(fold(num / den)) if den else np.nan


def with_ci(fn, S, g, req, *args, B=1000):
    point = fn(S, g, *args)
    ur = np.unique(req); idx = {u: np.where(req == u)[0] for u in ur}; vals = []
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(ur, len(ur))])
        vals.append(fn(S[:, ii], g[ii], *[x[ii] if isinstance(x, np.ndarray) else x for x in args]))
    vals = np.array(vals, float); vals = vals[~np.isnan(vals)]
    return {"sep": point, "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else None}


def main():
    rows = [json.loads(l) for l in open(HERE / "synthetic_v2_prompts.jsonl")]
    acts = torch.load(HERE / "synthetic_v2_acts.pt", map_location="cpu")
    judged = {d["id"]: d["cls"] for d in map(json.loads, open(HERE / "synthetic_v2_judged.jsonl"))}
    X = np.stack([acts[r["id"]].float().numpy() for r in rows]).astype(np.float32)
    S, seeds, hb = fit_scores(X)
    np.savez_compressed(HERE / "v2_scores.npz", ids=np.array([r["id"] for r in rows]), seeds=np.array(seeds), scores=S.astype(np.float32))

    def grp(r):
        c = judged.get(r["id"])
        if c is None: return None
        ans = c == "direct_answer"
        return ("jailbroken" if ans else "refused") if r["harm"] == "harmful" else ("benign_complied" if ans else "benign_refused")
    g = np.array([grp(r) for r in rows], dtype=object)
    fr = np.array([r["frame"] for r in rows]); req = np.array([f"{r['harm']}_{r['req_idx']}" for r in rows])
    rp = np.array([r["frame_type"] != "plain" for r in rows])
    res = {"hb_test_safe_auc_reproduced": hb, "n_unjudged": int((g == None).sum()),  # noqa: E711
           "counts": {s: dict(Counter(g[m])) for s, m in [("all", np.ones(len(g), bool)), ("plain", ~rp), ("roleplay", rp)]},
           "mean_score": {k: float(S[:, g == k].mean()) for k in G if (g == k).any()},
           "judge_classes": {f"{h}|{c}": n for (h, c), n in Counter((r["harm"], judged.get(r["id"])) for r in rows).items()}}
    for a, b in PAIRS:
        k = f"{a} vs {b}"
        res[k] = {"all": with_ci(sep, S, g, req, a, b),
                  "plain": with_ci(lambda S_, g_, a_, b_, m_: sep(S_[:, m_], g_[m_], a_, b_), S, g, req, a, b, ~rp),
                  "roleplay": with_ci(lambda S_, g_, a_, b_, m_: sep(S_[:, m_], g_[m_], a_, b_), S, g, req, a, b, rp),
                  "frame_matched": with_ci(lambda S_, g_, a_, b_, f_: framematched(S_, g_, f_, a_, b_), S, g, req, a, b, fr, B=300)}
        print(k, {s: round(v["sep"], 3) for s, v in res[k].items()}, flush=True)
    json.dump(res, open(HERE / "v2_results.json", "w"), indent=1)
    c = lambda o: "--" if np.isnan(o["sep"]) else f"{o['sep']:.2f} [{o['ci'][0]:.2f}, {o['ci'][1]:.2f}]"
    L = ["| Pair | All | Plain | Role-play | Frame-matched |", "|---|---|---|---|---|"]
    L += [f"| {a} vs {b} | " + " | ".join(c(res[f'{a} vs {b}'][s]) for s in ("all", "plain", "roleplay", "frame_matched")) + " |" for a, b in PAIRS]
    L += ["", "Counts: " + json.dumps(res["counts"]), "Mean P(jb): " + json.dumps({k: round(v, 3) for k, v in res["mean_score"].items()})]
    open(HERE / "v2_tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
