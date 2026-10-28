"""Emit each pair's minimum steering coefficient to jailbreak, for the
alpha dose-response figure (steering strength on x, share of prompts flipped
on y).

Covers the six steering arms of fig_steering_panels (PersonaJB, refusal, dmu,
utilitarian, closure_seeking, edgy) on harmbench / jbb / wildjb. The standard
attacks have no steering coefficient, so they are not here. Per pair we keep
(cluster id, min_alpha_to_jailbreak or null); clusters are behaviours, so the
laptop-side plotter can bootstrap the curve the same way steering_cell_cis.json
was made. Small enough to carry back on stdout; runs on the jumphost's bare
python3, like extract_steering_cells.py:

    ssh jumphost python3 - < full_trait_tools/extract_steering_alpha_curves.py

HarmBench is restricted to the originally-refused pairs via the in-row flag.
The wildjb runs carry no such flag (the panels figure joins them to unsteered
labels by prompt text separately), so wildjb cells are emitted over all pairs
and marked refused_only false -- restrict downstream if needed.
"""
import glob
import json
import os

BAZ = ("/mnt/dlab/scratch/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
       "/full_trait_output")
MINE = "/mnt/dlab/scratch/dlabscratch1/moskvore/persona_steer/outputs"
S = BAZ + "/paper_data/steering"
TRAITS = ["utilitarian", "closure_seeking", "edgy"]
REFUSED_KEYS = ("started_jailbroken", "original_jailbroken",
                "originally_jailbroken")


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


def cell(path, refused_only=False):
    if path is None or not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    sc = sched(path)
    field = ("final_jailbroken" if "final_jailbroken" in rows[0]
             else "jailbroken")
    key = next((k for k in REFUSED_KEYS if k in rows[0]), None)
    if refused_only:
        if key is None:
            refused_only = False          # flag absent: emit all, say so
        else:
            rows = [r for r in rows if not r.get(key)]
    pairs = []
    for r in rows:
        a = None
        if r.get(field):
            a = r.get("min_alpha_to_jailbreak")
            if a is None:                 # older runs: iteration index only
                k = r.get("n_iterations")
                a = sc[k - 1] if (k and sc and k <= len(sc)) else None
        pairs.append([str(r.get("behavior_id") or r.get("pair_id")), a])
    return {"path": path.replace(BAZ + "/", "baz:").replace(MINE + "/", "mine:"),
            "field": field, "schedule": sc, "refused_only": refused_only,
            "pairs": pairs}


out = {}
for bench, refused in (("harmbench", True), ("jbb", False), ("wildjb", True)):
    out[f"persona|{bench}"] = cell(f"{S}/{bench}/persona_w/harmbench/results.jsonl"
                                   if bench != "harmbench" else
                                   f"{S}/harmbench/persona_w/results.jsonl", refused)
    out[f"refusal|{bench}"] = cell(f"{MINE}/{bench}/refusal_pos1_neg__full/results.jsonl",
                                   refused)
    out[f"dmu|{bench}"] = cell(f"{MINE}/{bench}/dmu__full/results.jsonl", refused)

for t in TRAITS:
    out[f"{t}|harmbench"] = cell(first(f"{BAZ}/single_trait_all229/{t}/**/results.jsonl",
                                       f"{BAZ}/single_trait_sweep/{t}/**/results.jsonl"),
                                 True)
    out[f"{t}|jbb"] = cell(first(f"{BAZ}/single_trait_transfer/{t}_jbb/**/results.jsonl"))
    out[f"{t}|wildjb"] = cell(first(f"{BAZ}/single_trait_transfer/{t}_wjb/**/results.jsonl"),
                              True)

print("@@JSON@@" + json.dumps(out))
