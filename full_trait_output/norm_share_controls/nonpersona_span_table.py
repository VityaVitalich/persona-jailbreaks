#!/usr/bin/env python3
"""LaTeX rows for the appendix table of remove_nonpersona_span.json: transfer AUC (mean over draws) after each removal,
with the mean share of each dictionary vector removed in small type, plus the share of HarmBench activation variance
held by each removed span. Writes nonpersona_span_table.tex."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "remove_nonpersona_span.json"))
COLS = ["persona", "No Robots - P", "Alpaca - P", "Wikipedia - P", "random"]
ROWS = [("Wikipedia acts", "Wikipedia activations"), ("No Robots acts", "No Robots activations"), ("Alpaca acts", "Alpaca activations"),
        ("Shuffled contrasts", "Shuffled-label Alpaca contrasts"), ("Alpaca PCA-229", "Alpaca principal components"), ("Random Gaussian", "Random Gaussian directions")]
L = []
for k, lab in ROWS:
    o = R["dicts"][k]; w = np.mean([x["whole"][1] for x in o])
    cells = [f"{np.mean([x[c][2] for x in o]):.2f} {{\\scriptsize ({np.mean([x[c][0] for x in o]):.2f})}}" for c in COLS]
    L.append(f"    {lab} & {w:.2f} & " + " & ".join(cells) + r" \\")
L.append(r"    \midrule")
L.append(r"    Variance share of the span & -- & " + " & ".join(f"{R['spans'][c]['var_share']:.2f}" for c in COLS) + r" \\")
(HERE / "nonpersona_span_table.tex").write_text("\n".join(L) + "\n"); print("\n".join(L))
