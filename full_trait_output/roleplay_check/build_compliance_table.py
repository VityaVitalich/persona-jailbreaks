#!/usr/bin/env python3
"""Emit tables/compliance_separation_llama.tex from v2_results.json (analyze_v2.py): separation =
max(AUC, 1-AUC) of the PersonaJB score, mean over the 50 detectors."""
import json
from pathlib import Path
r = json.load(open(Path(__file__).resolve().parent / "v2_results.json"))
n = r["counts"]["all"]
ROWS = [("jailbroken vs refused", "Jailbroken vs.\\ refused"),
        ("refused vs benign_complied", "Refused vs.\\ benign answered"),
        ("jailbroken vs benign_complied", "Jailbroken vs.\\ benign answered")]
L = [r"\begin{table}[t]", r"  \centering", r"  \small",
     r"  \caption{\claudewritten{Separation (AUC, either direction) of the PersonaJB score between three groups of Llama-3.1-8B "
     r"responses: harmful requests answered (jailbroken, $n=%d$) or refused ($n=%d$), and benign requests answered ($n=%d$). "
     r"Same framing compares prompts only within each framing and averages over framings.}}" % (n["jailbroken"], n["refused"], n["benign_complied"]),
     r"  \label{tab:compliance-separation}",
     r"  \begin{tabular}{lcccc}", r"    \toprule",
     r"    Groups & All prompts & Plain & Role-play & Same framing \\", r"    \midrule"]
for k, name in ROWS:
    L.append(f"    {name} & " + " & ".join(f"{r[k][s]['sep']:.2f}" for s in ("all", "plain", "roleplay", "frame_matched")) + r" \\")
L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
out = Path.home() / "overleaf/persona-jailbreaks-iclr/tables/compliance_separation_llama.tex"
out.write_text("\n".join(L) + "\n"); print(out.read_text())
