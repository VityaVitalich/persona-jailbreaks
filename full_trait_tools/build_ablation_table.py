#!/usr/bin/env python3
r"""Emit tables/ablation_llama.tex from results/ablation_cells.json.

Rows are the ablated directions in the order of fig:attack, then the pooled
random controls, then the no-intervention regeneration arm. Per benchmark:
refused share of the initially-complied pool, the paired gain over `none` with
its 95% cluster-bootstrap CI, and the degenerate share.

The NeurIPS fork uses bare \cite, the ICLR fork \citep (its bib style is
author-year), so the citation command is a flag rather than baked in.

  python full_trait_tools/build_ablation_table.py --cite-cmd cite \
      > ~/overleaf/persona-jailbreaks/tables/ablation_llama.tex
  python full_trait_tools/build_ablation_table.py --cite-cmd citep \
      > ~/overleaf/persona-jailbreaks-iclr/tables/ablation_llama.tex
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROWS = [("PersonaJB", "persona"), ("Refusal direction ($-r$)", "refusal"),
        ("Jailbreak direction ($\\Delta\\mu$)", "dmu"),
        ("utilitarian", "utilitarian"), ("closure\\_seeking", "closure_seeking"),
        ("edgy", "edgy"),
        ("Random (full space)", "random_full"), ("Random (persona space)", "random_persona")]
BENCHES = [("HarmBench", "harmbench"), ("WildJB", "wildjb")]

CAPTION = r"""Directional ablation of the six steering directions of Figure~\ref{fig:attack},
  on the prompts the unsteered Llama-3.1-8B complied with (HarmBench: 1245
  human-jailbreak pairs; WildJB: 1025 adversarial prompts). Each unit direction is
  projected out of the residual stream, $x \mapsto x - (x\cdot v)v$, at every layer
  and every token position~\CITE{arditi2024refusal}, with one greedy generation per
  prompt and no coefficient. \emph{Refused}: share of initially complied prompts that
  the GPT-4.1-mini judge which produced the complied labels now marks as not jailbroken. \emph{Gain}: paired difference
  to the \emph{none} row, which regenerates the same prompts with no intervention,
  with a 95\% bootstrap CI (HarmBench resampled by behavior, WildJB by prompt).
  \emph{Degen.}: degenerate responses (under 20 characters, or more than 30\% repeated
  word 5-grams), which are never sent to the judge and are counted separately from both. Random controls are unit directions drawn isotropically in
  the full space or uniformly in the 229-trait span, pooled over three seeds."""


def f3(x):
    return f"{x:.3f}"


PLACEMENTS = [("All layers", ""), ("Layer 16", "@16")]

COMPARE_CAPTION = r"""Where the direction has to be removed. Share of initially complied
  prompts that the model refuses after ablating each direction, for three placements of
  the same projection $x \mapsto x - (x\cdot v)v$: at every layer and token position
  (the recipe of~\CITE{arditi2024refusal}), and at layer 16's block output only. Layer
  16 is where the steering of Figure~\ref{fig:attack} \emph{added} the direction, so
  that column is the exact converse of the steering intervention; later layers can
  write the direction back into the residual stream, which an all-layer ablation
  prevents. \emph{None} regenerates the
  same prompts with no intervention and is placement-independent. Full detail for the
  all-layer setting, including paired gains with confidence intervals and degenerate
  rates, is in Table~\ref{tab:direction-ablation}."""


def compare_table(get, label, caption_cite="citep"):
    """Rows = directions, columns = benchmark x placement, refused rate only."""
    cap = COMPARE_CAPTION.replace("\\CITE{", "\\" + caption_cite + "{")
    n = len(PLACEMENTS)
    out = [r"\begin{table}", r"  \caption{" + cap + "}", f"  \\label{{{label}}}",
           r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{5pt}",
           r"  \begin{tabular}{l " + " ".join(["c" * n] * len(BENCHES)) + "}",
           r"    \toprule",
           "    & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{b}}}" for b, _ in BENCHES)
           + r" \\",
           "    " + " ".join(f"\\cmidrule(lr){{{2 + n * i}-{1 + n * (i + 1)}}}"
                             for i in range(len(BENCHES))),
           "    Ablated direction & "
           + " & ".join(" & ".join(p for p, _ in PLACEMENTS) for _ in BENCHES) + r" \\",
           r"    \midrule"]
    prev = None
    for lab, key in ROWS:
        grp = "ctrl" if key.startswith("random") else "dir"
        if prev == "dir" and grp == "ctrl":
            out.append(r"    \midrule")
        prev = grp
        cols = []
        for _, b in BENCHES:
            for _, place in PLACEMENTS:
                c = get(key, b, place)
                cols.append(f3(c["refused"]["rate"]) if c and "refused" in c else "--")
        body = " & ".join(f"\\textbf{{{v}}}" if key == "persona" and v != "--" else v
                          for v in cols)
        out.append(f"    {lab} & {body} \\\\")
    out.append(r"    \midrule")
    cols = []
    for _, b in BENCHES:
        c = get("none", b)
        r = f3(c["refused"]["rate"]) if c and "refused" in c else "--"
        cols += [r] * n
    out.append(f"    None (regeneration only) & {' & '.join(cols)} \\\\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(REPO / "results/ablation_cells.json"))
    ap.add_argument("--label", default="tab:direction-ablation")
    ap.add_argument("--cite-cmd", default="citep", choices=["cite", "citep"],
                    help="cite for the NeurIPS fork, citep for the ICLR fork")
    ap.add_argument("--placement", default="", choices=["", "@16"],
                    help="which placement the detail table reports")
    ap.add_argument("--compare", action="store_true",
                    help="emit the placement-comparison table (refused rate only, "
                         "all three placements side by side) instead")
    args = ap.parse_args()
    caption = CAPTION.replace("\\CITE{", "\\" + args.cite_cmd + "{")
    cells = json.load(open(args.cells))

    def get(key, bench, placement=None):
        # `none` has no hook, so one baseline serves every placement
        sfx = "" if key == "none" else (args.placement if placement is None else placement)
        return cells.get(f"{key}{sfx}|{bench}")

    if args.compare:
        print(compare_table(get, args.label if args.label != "tab:direction-ablation"
                            else "tab:ablation-placement", caption_cite=args.cite_cmd))
        return

    lines = [r"\begin{table}", r"  \caption{" + caption + "}", f"  \\label{{{args.label}}}",
             r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{tabular}{l ccc ccc}", r"    \toprule",
             "    & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{b}}}" for b, _ in BENCHES) + r" \\",
             "    " + " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                               for i in range(len(BENCHES))),
             "    Ablated direction & " + " & ".join("Refused & Gain [95\\% CI] & Degen."
                                                     for _ in BENCHES) + r" \\",
             r"    \midrule"]
    prev = None
    for label, key in ROWS:
        grp = "ctrl" if key.startswith("random") else "dir"
        if prev == "dir" and grp == "ctrl":
            lines.append(r"    \midrule")
        prev = grp
        cols = []
        for _, b in BENCHES:
            c = get(key, b)
            if c is None:
                cols += ["--", "--", "--"]
                continue
            if "refused" not in c:
                cols += ["--", "--", "--"]
                continue
            g = c.get("refused_gain_vs_none")
            gain = (f"{g['delta']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]" if g else "--")
            cols += [f3(c["refused"]["rate"]), gain, f3(c["degenerate"]["rate"])]
        bold = key == "persona"
        cells_tex = " & ".join(f"\\textbf{{{v}}}" if bold and i % 3 == 0 else v
                               for i, v in enumerate(cols))
        lines.append(f"    {label} & {cells_tex} \\\\")
    lines.append(r"    \midrule")
    cols = []
    for _, b in BENCHES:
        c = get("none", b)
        cols += ([f3(c["refused"]["rate"]), "--", f3(c["degenerate"]["rate"])]
                 if c and "refused" in c else ["--", "--", "--"])
    lines.append(f"    None (regeneration only) & {' & '.join(cols)} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
