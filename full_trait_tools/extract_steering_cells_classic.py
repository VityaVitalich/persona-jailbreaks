"""Per-cluster counts for the standard-attack cells, from paper_data/activations.

These are the attack response dumps used for detection; each row carries
attack_type, behavior_id and a jailbroken label, which is what the published
Table 4 classic rows are computed from. Both bases are emitted (per-pair and
per-behaviour "any variant succeeded") so the one matching the table can be
picked on evidence rather than assumed.
"""
import json
import os
from collections import defaultdict

B = ("/mnt/dlab/scratch/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
     "/full_trait_output")
PDA = f"{B}/paper_data/activations"
S = f"{B}/paper_data/steering"


def rd(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def clusters(rows, fam_field="attack_type", fam=None):
    per = defaultdict(list)
    for r in rows:
        if fam is not None and str(r.get(fam_field, "")).lower() != fam.lower():
            continue
        if "jailbroken" not in r:
            continue
        per[str(r.get("behavior_id"))].append(bool(r["jailbroken"]))
    return [(len(v), sum(v)) for v in per.values()]


out = {}

# --- HarmBench: one responses.jsonl per family, attack_type-filtered ---
for fam, path in (("GPTFuzz", f"{PDA}/gptfuzz/responses.jsonl"),
                  ("GCG", f"{PDA}/gcg/responses.jsonl"),
                  ("PAIR", f"{PDA}/pair/responses.jsonl"),
                  ("PAP", f"{PDA}/pap/responses.jsonl"),
                  ("PEZ", f"{PDA}/pez/responses.jsonl")):
    rows = rd(path)
    types = sorted({str(r.get("attack_type")) for r in rows})
    cl = clusters(rows, fam=fam)
    if not cl:                      # attack_type may be spelled differently
        cl = clusters([r for r in rows
                       if str(r.get("attack_type")) not in ("plain", "direct_request")])
    out[f"{fam}|harmbench"] = {"clusters": cl, "path": path.replace(B, "baz:"),
                               "attack_types": types}

# --- JBB: pooled activation dump + a separate GCG dump ---
ja = rd(f"{B}/jbb_attack_activations/classified_responses.jsonl")
jtypes = sorted({str(r.get("attack_type")) for r in ja})
for fam in ("GPTFuzz", "PAIR", "PAP", "PEZ"):
    out[f"{fam}|jbb"] = {"clusters": clusters(ja, fam=fam),
                         "path": "baz:jbb_attack_activations",
                         "attack_types": jtypes}
gcg = rd(f"{PDA}/jbb_gcg/classified_responses.jsonl")
out["GCG|jbb"] = {"clusters": clusters(gcg),
                  "path": "baz:paper_data/activations/jbb_gcg",
                  "attack_types": sorted({str(r.get("attack_type")) for r in gcg})}

# --- WildJB: the classic runs already reproduced; keep them for one source ---
for fam in ("GPTFuzz", "GCG", "PAIR", "PAP", "PEZ"):
    rows = rd(f"{S}/wildjb/classic/{fam}/results.jsonl")
    out[f"{fam}|wildjb"] = {"clusters": clusters(rows),
                            "path": f"baz:paper_data/steering/wildjb/classic/{fam}",
                            "attack_types": []}

print("@@JSON@@" + json.dumps(out))
