#!/usr/bin/env python3
"""Pairwise separation matrix between the 8 synthetic cells (harm x framing): folded AUC
max(a,1-a) of the PersonaJB score, mean over the 50 seeds. Reads part2_scores.npz; writes
part2_matrix.md and part2_matrix.json (signed + folded)."""
import json
from pathlib import Path
import numpy as np
from analyze_part1 import auc_rows

HERE = Path(__file__).resolve().parent
d = np.load(HERE / "part2_scores.npz", allow_pickle=True)
rows = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "synthetic_prompts.jsonl")}
meta = [rows[i] for i in d["ids"]]; S = d["scores"]
FT = [("plain", "plain"), ("neutral_persona", "persona"), ("neutral_fiction", "fiction"), ("jb_persona", "jb-persona")]
cells = [(h, ft, f"{h[0].upper()}:{short}") for h in ("harmful", "benign") for ft, short in FT]
mask = {lab: np.array([r["harm"] == h and r["frame_type"] == ft for r in meta]) for h, ft, lab in cells}
labs = [c[2] for c in cells]
signed = {a: {b: float(np.nanmean(auc_rows(np.r_[np.ones(mask[a].sum()), np.zeros(mask[b].sum())].astype(bool),
                                             np.concatenate([S[:, mask[a]], S[:, mask[b]]], 1)))) if a != b else float("nan")
              for b in labs} for a in labs}
folded = {a: {b: max(v, 1 - v) for b, v in row.items()} for a, row in signed.items()}
L = ["| | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
for a in labs:
    L.append(f"| **{a}** | " + " | ".join("–" if a == b else f"{folded[a][b]:.2f}" for b in labs) + " |")
L += ["", "Mean P(jb): " + ", ".join(f"{a} {S[:, mask[a]].mean():.2f}" for a in labs)]
open(HERE / "part2_matrix.md", "w").write("\n".join(L) + "\n")
json.dump({"signed_auc_row_vs_col": signed, "folded": folded}, open(HERE / "part2_matrix.json", "w"), indent=1)
print("\n".join(L))
