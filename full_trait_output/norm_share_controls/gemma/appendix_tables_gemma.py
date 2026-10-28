#!/usr/bin/env python3
"""LaTeX tables for the Gemma-4-31B appendix section (sections/appendix/gemma_controls.tex), from the saved outputs
of the Gemma ports and audits in this directory. Writes into the Overleaf tables/ directory:
  controls_detection_gemma, controls_nonpersona_span_gemma, controls_refdmu_auc_gemma, controls_null_directions,
  jbrem_harmbench_gemma, jbrem_spans_gemma, steering_split_gemma, steering_dose_gemma."""
import json, sys
from pathlib import Path
import numpy as np
G = Path(__file__).resolve().parent; TAB = Path.home() / "overleaf/persona-jailbreaks-iclr/tables"
O = G.parents[2]; sys.path.insert(0, str(O / "full_trait_tools"))
from extract_gemma_steering_cells import looks_like_text  # noqa: E402
J = lambda p: json.load(open(G / p))
f2 = lambda x: f"{x:.2f}"
def write(name, s): (TAB / f"{name}.tex").write_text(s); print("wrote", name)

# ---- detection with control dictionaries ----
R, P = J("remove_persona_all_controls.json"), J("persona_row_check.json")
ROWS = [("Wikipedia acts", "Wikipedia activations"), ("No Robots acts", "No Robots activations"), ("Alpaca acts", "Alpaca activations"),
        ("Shuffled contrasts", "Shuffled-label Alpaca contrasts"), ("Alpaca PCA-240", "Alpaca principal components"), ("Random Gaussian", "Random Gaussian directions")]
body = [f"    Trait vectors (PersonaJB) & 1.00 & {f2(P['hb'])} & -- & {f2(P['transfer'])} & -- \\\\", "    \\midrule"]
for k, lab in ROWS:
    o = R[k]; w = np.mean([x["whole"] for x in o], 0); p = np.mean([x["persona"] for x in o], 0)
    body.append(f"    {lab} & {f2(p[0])} & {f2(w[0])} & {f2(p[1])} & {f2(w[1])} & {f2(p[2])} \\\\")
write("controls_detection_gemma", r"""\begin{table}[!htb]
  \centering
  \small
  \caption{\claudewritten{Jailbreak detection on Gemma-4-31B when the $240$ trait vectors of PersonaJB are replaced by $240$ non-persona directions (layer $30$); the rest of the detector is unchanged, including the WildJailbreak rows added to training. AUC in either direction on the HarmBench test set, mean over the $50$ splits that contain jailbroken test prompts (and over three draws of each random dictionary); transfer is the mean over the five attack families. Columns as in Table~\ref{tab:controls-detection}.}}
  \label{tab:controls-detection-gemma}
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{lccccc}
    \toprule
    & & \multicolumn{2}{c}{HarmBench} & \multicolumn{2}{c}{Avg.\ Transfer} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}
    Dictionary ($240$ directions) & \shortstack{Share in\\trait span} & Full & \shortstack{Persona\\removed} & Full & \shortstack{Persona\\removed} \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- removal of persona-free activation spans ----
N = J("remove_nonpersona_span.json"); COLS = ["persona", "No Robots - P", "Alpaca - P", "Wikipedia - P", "random"]
body = []
for k, lab in ROWS:
    o = N["dicts"][k]; w = np.mean([x["whole"][1] for x in o])
    body.append(f"    {lab} & {f2(w)} & " + " & ".join(f"{np.mean([x[c][2] for x in o]):.2f} {{\\scriptsize ({np.mean([x[c][0] for x in o]):.2f})}}" for c in COLS) + r" \\")
body += [r"    \midrule", r"    Variance share of the span & -- & " + " & ".join(f2(N["spans"][c]["var_share"]) for c in COLS) + r" \\"]
write("controls_nonpersona_span_gemma", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \caption{\claudewritten{Gemma-4-31B: transfer AUC of the PersonaJB detector on each dictionary after removing a $240$-dimensional span from every direction, as in Table~\ref{tab:controls-nonpersona-span}. In parentheses: mean share of each direction's squared norm that the removal takes away. Last row: share of the variance of the HarmBench activations in the removed span.}}
  \label{tab:controls-nonpersona-span-gemma}
  \begin{tabular}{lcccccc}
    \toprule
    & & & \multicolumn{3}{c}{Activation span without persona} & \\
    \cmidrule(lr){4-6}
    Dictionary ($240$ directions) & Full & \shortstack{Persona\\removed} & No Robots & Alpaca & Wikipedia & Random \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- refusal / jailbreak direction by family ----
A = J("refdmu_tables.json")["A"]; FAM = ["HarmBench", "GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
assert not any(A[k][f + "_reversed"] for k in A for f in FAM[1:])
body = []
for d, lab in (("refusal", "Refusal direction"), ("dmu", "Jailbreak direction")):
    for v, vl in (("intact", "intact"), ("persona removed", "removed")):
        a = A[f"{d}|{v}"]; body.append(f"    {lab if v == 'intact' else ''} & {vl} & " + " & ".join(f2(a[f]) for f in FAM) + f" & {f2(a['mean5'])} \\\\")
    if d == "refusal": body.append(r"    \midrule")
write("controls_refdmu_auc_gemma", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \caption{\claudewritten{Gemma-4-31B: jailbreak detection AUC of the refusal and jailbreak directions used as the score $h^\top v$, intact and with the persona component removed, as in Table~\ref{tab:controls-refdmu-auc}. HarmBench: test set, mean over the splits that contain jailbroken test prompts, with the jailbreak direction estimated on the training rows of each split. On Gemma both directions separate every family in the same direction as on HarmBench.}}
  \label{tab:controls-refdmu-auc-gemma}
  \begin{tabular}{llccccccc}
    \toprule
    & & HarmBench & GCG & PAIR & PAP & GPTFuzz & PEZ & Avg.\ (5) \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- single-direction transfer against a random-direction null, both models ----
V = J("audit/vin_vs_null.json")
body = []
for m, ml in (("llama", "Llama-3.1-8B"), ("gemma", "Gemma-4-31B")):
    nl = V[m]["_null"]
    body.append(f"    \\multicolumn{{5}}{{l}}{{\\emph{{{ml}}} (random direction: mean {f2(nl['mean'])}, $95$th percentile {f2(nl['p95'])})}} \\\\")
    for d, dl in (("refusal", "Refusal direction"), ("dmu", "Jailbreak direction")):
        for part, pl in (("", "full"), (" v_in", "persona component"), (" v_out", "persona removed")):
            q = V[m][d + part]
            body.append(f"    {dl if part == '' else ''} & {pl} & {f2(q['transfer'])} & ${q['d']:+.2f}$ {{\\scriptsize $[{q['lo']:+.2f}, {q['hi']:+.2f}]$}} & {q['pctG']:.0f} / {q['pctP']:.0f} \\\\".replace("+-", "-"))
    if m == "llama": body.append(r"    \midrule")
write("controls_null_directions", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \caption{\claudewritten{Transfer AUC of single directions used as the score $h^\top v$ (mean over the five attack families, AUC in either direction), against random directions. $\Delta$: difference from the mean over $1{,}000$ random unit directions, with a $95\%$ interval from resampling prompts within each family. Percentile: rank among the $1{,}000$ random directions in the full space / among $1{,}000$ random directions inside the trait span. The jailbreak direction is estimated on all HarmBench rows.}}
  \label{tab:controls-null-directions}
  \begin{tabular}{llccc}
    \toprule
    Direction & Part & Transfer & $\Delta$ vs.\ random & Percentile \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- what the jailbreak direction's remainder reads on HarmBench ----
B = J("refdmu_tables.json")["B"]; D = J("template_test/dilution_pressure.json"); U = J("template_test/unrestricted_persona.json")["c"]
row = lambda lab, b: f"    {lab} & {f2(b['within_request'])} & {f2(b['within_template'])} & {f2(b['rho_template'])} & {f2(b['cos_refusal'])} \\\\"
body = [row("Jailbreak direction", B["dmu intact"]), row(r"\quad persona component $w_{\mathrm{in}}$", B["dmu persona component"]),
        row(r"\quad remainder $w_{\mathrm{out}}$", B["dmu persona removed"]), row("Refusal direction", B["refusal intact"]),
        row(r"\quad persona component removed", B["refusal persona removed"]), row("Random Alpaca activation", B["random Alpaca activation (median of 240)"]),
        row("Random Gaussian direction", B["random Gaussian direction (median of 240)"]),
        f"    Template oracle & {f2(D['single']['template oracle']['within_request'])} & -- & 1.00 & -- \\\\", r"    \midrule"]
kw = max(["no_refusal", "amoral", "prefix_or_format", "dual_response", "ignore_rules", "stay_in_character"], key=lambda k: D["single"][k]["within_request"])
lo = lambda lab, x: f"    {lab} & {f2(x['within_request'])} & -- & {f2(x['rho_template'])} & -- \\\\"
body += [lo(r"$w_{\mathrm{out}}$, length features regressed out", D["residual"]["dilution"]), lo(r"$w_{\mathrm{out}}$, keyword features regressed out", D["residual"]["pressure"]),
         lo(r"$w_{\mathrm{out}}$, both regressed out", D["residual"]["dilution + pressure"]), lo(r"$w_{\mathrm{out}}$, jailbreak-persona span projected out", U["hb|w_out with U6 projected out"]),
         lo("Template text after the request (log length)", D["single"]["log_template_after"]), lo(f"Best keyword feature ({kw.replace('_', ' ')})", D["single"][kw]),
         lo(r"Jailbreak-persona direction $u_{\mathrm{jbp}}$", U["hb|u_jbp"])]
n_jb = 63
write("jbrem_harmbench_gemma", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \caption{\claudewritten{Gemma-4-31B: what scores read on the HarmBench human-jailbreak set, as in Table~\ref{tab:jbrem-harmbench}. Only $""" + str(n_jb) + r"""$ of the $3{,}180$ prompts are jailbroken, so these columns rest on $21$ requests and $13$ templates with any jailbroken prompt. The jailbreak direction is estimated on all of these prompts, so its rows and those of its parts are in-sample. Random rows: median over $240$ directions. AUC in either direction.}}
  \label{tab:jbrem-harmbench-gemma}
  \begin{tabular}{lcccc}
    \toprule
    Score & Within request & Within template & $\rho_{\mathrm{template}}$ & Cos \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- spans holding the remainder ----
Sh = J("w_out_in_spans.json")["share"]
SC = ["random Gaussian", "Wikipedia acts", "No Robots acts", "Alpaca acts", "HarmBench PCA-240", "random (1200)", "Alpaca acts x5 (1200)"]
body = [f"    {lab} & " + " & ".join(f2(Sh[c][k]) for c in SC) + r" \\" for k, lab in
        (("w_out", r"$w_{\mathrm{out}}$"), ("w_in", r"$w_{\mathrm{in}}$"), ("jailbreak direction", "Jailbreak direction"))]
write("jbrem_spans_gemma", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \caption{\claudewritten{Gemma-4-31B: share of squared norm of $w_{\mathrm{out}}$, $w_{\mathrm{in}}$ and the jailbreak direction in spans of activations, as in Table~\ref{tab:jbrem-spans}, with $240$ directions per span.}}
  \label{tab:jbrem-spans-gemma}
  \begin{tabular}{lccccccc}
    \toprule
    & \multicolumn{5}{c}{$240$ dimensions} & \multicolumn{2}{c}{$1{,}200$ dimensions} \\
    \cmidrule(lr){2-6}\cmidrule(lr){7-8}
    Direction & Random & Wikipedia & No Robots & Alpaca & \shortstack{HarmBench\\PCA} & Random & Alpaca $\times 5$ \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- steering: paired changes ----
St = json.load(open(G / "steer/steer_split_gemma.json"))
ext = [json.loads(l) for l in open(G / "steer/outputs_gemma_split/harmbench/persona_pattern/results.jsonl")]
tb = lambda x: x is True or str(x) == "True"
ext_asr = 100 * np.mean([tb(r["final_jailbroken"]) and not tb(r["final_degenerate"]) and looks_like_text(r["final_response"]) for r in ext])
cellA = lambda s: f"{s['asr']:.1f} {{\\scriptsize $[{s['lo']:.1f}, {s['hi']:.1f}]$}}"
cellD = lambda s: (f"${s['diff']:+.1f}$ {{\\scriptsize $[{s['diff_lo']:+.1f}, {s['diff_hi']:+.1f}]$}}".replace("+-", "-")) if "diff" in s else "--"
SR = [("PersonaJB direction|detector normal", r"PersonaJB direction, detector normal$^\ast$"), ("PersonaJB direction|activation pattern, persona space", r"\quad activation pattern in persona space"),
      ("Refusal direction|original", "Refusal direction"), ("Refusal direction|persona component only", r"\quad persona component only"), ("Refusal direction|persona component removed", r"\quad persona component removed"),
      ("Jailbreak direction|original", "Jailbreak direction"), ("Jailbreak direction|persona component only", r"\quad persona component only"), ("Jailbreak direction|persona component removed", r"\quad persona component removed"),
      ("Random direction", r"Random direction$^\ast$")]
body = [f"    {lab} & {cellA(St[k])} & {cellD(St[k])} \\\\" for k, lab in SR]
body.insert(2, f"    \\quad activation pattern, $\\alpha$ up to $0.8$ & {ext_asr:.1f} & -- \\\\")
write("steering_split_gemma", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \caption{\claudewritten{Steering Gemma-4-31B on $300$ originally refused HarmBench prompts (cumulative $\alpha$ from $0.1$ to $0.5$). ASR counts only coherent responses; $95\%$ intervals from a bootstrap over behaviors. $\Delta$: paired change against the original direction (for the activation pattern, against the detector normal). $^\ast$Earlier runs on the full refused pool, restricted to the same prompts. The $\alpha$ up to $0.8$ row extends the schedule for the prompts the activation pattern had not jailbroken.}}
  \label{tab:steering-split-gemma}
  \begin{tabular}{lcc}
    \toprule
    Direction & ASR (\%) & $\Delta$ (points) \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")

# ---- steering: single doses ----
SD = O / "full_trait_output/outputs_gemma_singledose/harmbench"; AL = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5"]
body = []
for arm, lab in (("w_persona__full", "PersonaJB direction"), ("refusal_pos1_neg__full", "Refusal direction"), ("dmu__full", "Jailbreak direction"), ("trait__emotional", "Emotional trait")):
    cells = []
    for a in AL:
        rr = [json.loads(l) for l in open(SD / f"{arm}__a{a}/results.jsonl")]
        deg = np.array([tb(r["final_degenerate"]) or not looks_like_text(r["final_response"]) for r in rr])
        jb = np.array([tb(r["final_jailbroken"]) for r in rr]) & ~deg
        cells.append(f"{100 * jb.mean():.0f} {{\\scriptsize ({100 * deg.mean():.0f})}}")
    body.append(f"    {lab} & " + " & ".join(cells) + r" \\")
write("steering_dose_gemma", r"""\begin{table}[!htb]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \caption{\claudewritten{Steering Gemma-4-31B at a single strength $\alpha$ on $300$ originally refused HarmBench prompts. ASR (\%) counting only coherent responses; in parentheses, the percentage of responses that are gibberish or repetitive text.}}
  \label{tab:steering-dose-gemma}
  \begin{tabular}{lccccccc}
    \toprule
    Direction & $\alpha=0.05$ & $0.1$ & $0.15$ & $0.2$ & $0.3$ & $0.4$ & $0.5$ \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
