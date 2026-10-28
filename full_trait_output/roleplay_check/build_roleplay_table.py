#!/usr/bin/env python3
"""Emit tables/roleplay_benchmarks_llama.tex from part1_results.json (role-play = persona|fiction,
rubric-v2 labels). AUCs folded max(a,1-a) as in tab:detection; subset AUCs shown only where the
subset has >=50 prompts. HarmBench test subsets: mean over the splits whose test set contains the subset."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "part1_results.json"))
fold = lambda a: max(a, 1 - a)
ROWS = [("HB_test", "HB", "HarmBench (test set)"), ("GCG", "GCG", "GCG"), ("PAIR", "PAIR", "PAIR"),
        ("PAP", "PAP", "PAP"), ("GPTFuzz", "GPTFuzz", "GPTFuzz"), ("PEZ", "PEZ", "PEZ")]
CAPTION = (r"Role-play in the Llama-3.1-8B detection benchmarks. Role-play share is the fraction of prompts that "
           r"GPT-4.1-mini labels as persona or fiction; role-play label AUC uses that label alone to detect jailbreaks. "
           r"PersonaJB AUC is reported on all prompts, as in Table~\ref{tab:detection}, and on role-play and "
           r"non-role-play subsets with at least $50$ prompts. Half of the PAIR set consists of unwrapped behaviors. "
           r"$^\dagger$Mean over the $%d$ of $50$ splits whose test set contains non-role-play prompts.")
L = [r"\begin{table}[t]", r"  \centering", r"  \small",
     r"  \caption{\claudewritten{" + CAPTION % r["HB_test"]["broad"]["n_seeds_within_nonrp"] + "}}",
     r"  \label{tab:roleplay-benchmarks}",
     r"  \begin{tabular}{lccccc}", r"    \toprule",
     r"    & & & \multicolumn{3}{c}{PersonaJB AUC} \\", r"    \cmidrule(lr){4-6}",
     r"    Attack family & Role-play share & Role-play label AUC & All & Role-play & Non-role-play \\",
     r"    \midrule"]
for key, lab_key, name in ROWS:
    o = r[key]["broad"]
    share = r["label_counts"][lab_key]; n = sum(share.values()); rp = share.get("persona", 0) + share.get("fiction", 0)
    sub = lambda which, cnt, dag="": f"{fold(o['auc_within_' + which]):.2f}{dag}" if cnt >= 50 else "--"
    within_rp = sub("rp", rp)
    within_non = sub("nonrp", n - rp, r"$^\dagger$" if key == "HB_test" else "")
    L.append(f"    {name} & {100 * rp / n:.0f}\\% & {fold(o['auc_rp_label_only']):.2f} & {fold(o['auc_all']):.2f} & {within_rp} & {within_non} \\\\")
L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
out = Path.home() / "overleaf/persona-jailbreaks-iclr/tables/roleplay_benchmarks_llama.tex"
out.write_text("\n".join(L) + "\n"); print(out.read_text())
