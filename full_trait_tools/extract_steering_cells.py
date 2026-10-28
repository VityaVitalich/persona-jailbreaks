"""Emit, for every figure cell, the per-cluster (n_pairs, n_success) counts.

That is all a cluster bootstrap needs, and it is small enough to carry back to
the laptop, where numpy exists. Runs on the jumphost's bare python3.

HarmBench pairs are behaviour x jailbreak-template, so pairs sharing a
behaviour are not independent; keeping the cluster structure lets the interval
be resampled over behaviours rather than pairs.
"""
import glob
import json
import os

BAZ = ("/mnt/dlab/scratch/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
       "/full_trait_output")
MINE = "/mnt/dlab/scratch/dlabscratch1/moskvore/persona_steer/outputs"
S = BAZ + "/paper_data/steering"
CAP = 0.5
TRAITS = ["utilitarian", "closure_seeking", "edgy"]
CLASSIC = ["GPTFuzz", "GCG", "PAIR", "PAP", "PEZ"]


def first(*globs):
    for g in globs:
        h = sorted(glob.glob(g, recursive=True))
        if h:
            return h[0]
    return None


def sched(path):
    s = os.path.join(os.path.dirname(path), "summary.json")
    if not os.path.exists(s):
        return None
    d = json.load(open(s))
    return d.get("alpha_schedule") or d.get("config", {}).get("alpha_schedule")


def cell(path, refused_only=False, cap=CAP, field=None):
    if path is None or not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    sc = sched(path)
    if field is None:
        field = ("final_jailbroken" if "final_jailbroken" in rows[0]
                 else "jailbroken")
    if field not in rows[0]:
        return None
    if refused_only:
        key = next((k for k in ("started_jailbroken", "original_jailbroken",
                                "originally_jailbroken") if k in rows[0]), None)
        if key is None:
            return None
        rows = [r for r in rows if not r.get(key)]
    clusters = {}
    for r in rows:
        jb = bool(r.get(field))
        # escalation stops at the first success, so a longer schedule is recut
        # to alpha <= cap using the stored per-pair alpha (or the iteration
        # index through that run's own schedule)
        if jb and cap is not None and "min_alpha_to_jailbreak" in r:
            a = r.get("min_alpha_to_jailbreak")
            if a is None:
                k = r.get("n_iterations")
                a = sc[k - 1] if (k and sc and k <= len(sc)) else None
            jb = a is not None and a <= cap + 1e-9
        c = str(r.get("behavior_id") or r.get("pair_id"))
        n, s = clusters.get(c, (0, 0))
        clusters[c] = (n + 1, s + int(jb))
    return {"path": path.replace(BAZ + "/", "baz:").replace(MINE + "/", "mine:"),
            "field": field, "clusters": list(clusters.values())}


out = {}
for bench, hb in (("harmbench", True), ("jbb", False), ("wildjb", False)):
    out[f"persona|{bench}"] = cell(f"{S}/{bench}/persona_w/harmbench/results.jsonl"
                                   if bench != "harmbench" else
                                   f"{S}/harmbench/persona_w/results.jsonl", hb)
    out[f"refusal|{bench}"] = cell(f"{MINE}/{bench}/refusal_pos1_neg__full/results.jsonl", hb)
    out[f"dmu|{bench}"] = cell(f"{MINE}/{bench}/dmu__full/results.jsonl", hb)

for t in TRAITS:
    out[f"{t}|harmbench"] = cell(first(f"{BAZ}/single_trait_all229/{t}/**/results.jsonl",
                                       f"{BAZ}/single_trait_sweep/{t}/**/results.jsonl"))
    out[f"{t}|jbb"] = cell(first(f"{BAZ}/single_trait_transfer/{t}_jbb/**/results.jsonl"))
    out[f"{t}|wildjb"] = cell(first(f"{BAZ}/single_trait_transfer/{t}_wjb/**/results.jsonl"))

for a in CLASSIC:
    for bench in ("jbb", "wildjb"):
        p = f"{S}/{bench}/classic/{a}/results.jsonl"
        out[f"{a}|{bench}"] = cell(p, field="jailbroken")
        # the JBB classic column does not reproduce from `jailbroken`; dump the
        # rubric judge too so the right field can be identified rather than guessed
        alt = cell(p, field="jbb_jailbroken")
        if alt:
            out[f"{a}|{bench}|ALT_jbb_jailbroken"] = alt
    hbp = first(f"{BAZ}/**/harmbench*/classic/{a}/results.jsonl",
                f"{BAZ}/baselines_harmbench*/{a}/**/results.jsonl",
                f"{BAZ}/**/{a}/**/harmbench*/results.jsonl")
    out[f"{a}|harmbench"] = cell(hbp, field="jailbroken") if hbp else None

print("@@JSON@@" + json.dumps({k: v for k, v in out.items()}))
