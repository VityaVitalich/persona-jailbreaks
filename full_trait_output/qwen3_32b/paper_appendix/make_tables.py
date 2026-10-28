"""Builds the Qwen appendix tables from result files (no numbers typed by hand).
Sources: refusal_rates.jsonl (Llama/Gemma judge labels, shared data) + transfer/diag/scores_audit_L48_last.npz (Qwen labels);
transfer/diag/cross_family_tuned.json; transfer/diag/within_family_tuned.json; probe_methods/out/probe_methods.json."""
import json, numpy as np, sys
Q = "/Users/viktor/persona-vector-jbing/full_trait_output/qwen3_32b"; OUT = sys.argv[1]
SETS = ["hb", "wjb", "pap", "gcg", "pair", "gptfuzz", "pez"]; NAME = {"hb": "HarmBench", "wjb": "WildJB", "pap": "PAP", "gcg": "GCG", "pair": "PAIR", "gptfuzz": "GPTFuzz", "pez": "PEZ"}
MOD = {"llama": "Llama-3.1-8B", "gemma": "Gemma-4-31B", "qwen": "Qwen3-32B"}
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"{x:.3f}"
def sg(x, nd):
    v = round(x, nd); v = 0.0 if v == 0 else v
    return ("$-$" if v < 0 else "+") + f"{abs(v):.{nd}f}"
def ci(c): return f"[{c[0]:.2f}, {c[1]:.2f}]"
# ---- T1: ASR ----
rr = [json.loads(l) for l in open(sys.argv[2])]; key = {"harmbench": "hb", "wildjailbreak": "wjb"}
asr = {("llama" if r["model"].startswith("llama") else "gemma", key.get(r["set"], r["set"])): 100 * r["jailbroken"] for r in rr}
nn = {("llama" if r["model"].startswith("llama") else "gemma", key.get(r["set"], r["set"])): r["n"] for r in rr}
A = np.load(f"{Q}/transfer/diag/scores_audit_L48_last.npz")
for s in SETS: asr[("qwen", s)] = 100 * A[f"labels_{s}"].mean(); nn[("qwen", s)] = len(A[f"labels_{s}"])
rows = "\n".join(f"    {MOD[m]} & " + " & ".join(f"{asr[(m, s)]:.1f}" for s in SETS) + r" \\" for m in ("llama", "gemma", "qwen"))
ncol = " & ".join(f"{nn[('qwen', s)]}" for s in SETS)
open(f"{OUT}/qwen_asr.tex", "w").write(r"""\begin{table}[!htb]
  \centering
  \small
  \caption{\claudewritten{Attack success rate (\%, GPT-4.1-mini judge) of each set on each model. HarmBench is the human-jailbreak pool used for training. GCG, PAIR, PEZ and GPTFuzz test cases are optimized against each model itself; PAP and WildJailbreak prompts are shared. Qwen3-32B set sizes are in the last row (Llama has $800$ PAIR prompts).}}
  \label{tab:qwen-asr}
  \begin{tabular}{l""" + "c" * len(SETS) + r"""}
    \toprule
    Model & """ + " & ".join(NAME[s] for s in SETS) + r""" \\
    \midrule
""" + rows + r"""
    \midrule
    $n$ (Qwen3-32B) & """ + ncol + r""" \\
    \bottomrule
  \end{tabular}
\end{table}
""")
# ---- T2: cross-family transfer per training set (tuned C) ----
cf = json.load(open(f"{Q}/transfer/diag/cross_family_tuned.json"))["cells"]
DET = [("PersonaJB", "PersonaJB"), ("Gaussian", "Random Gaussian"), ("fullprobe", "Raw-activation probe")]
lines = []
for m in ("llama", "qwen"):
    for i, (d, dn) in enumerate(DET):
        c = [cf[f"{m}__{d}__{s}"]["auc"] for s in SETS]; av = cf[f"{m}__{d}__avg"]
        lines.append(f"    {MOD[m] if i == 0 else ''} & {dn} & " + " & ".join(f2(x) for x in c) + f" & {f2(av['auc'])} {ci(av['ci95'])}" + r" \\")
    dl = cf[f"{m}__delta_PersonaJB_minus_Gaussian__avg"]; dr = cf[f"{m}__delta_PersonaJB_minus_fullprobe__avg"]
    lines.append(f"    & PersonaJB $-$ Gaussian & " + " & ".join(sg(cf[f'{m}__delta_PersonaJB_minus_Gaussian__{s}']['delta'], 2) for s in SETS) + f" & {sg(dl['delta'], 3)} [{sg(dl['ci_bonf'][0], 3)}, {sg(dl['ci_bonf'][1], 3)}]" + r" \\")
    lines.append(f"    & PersonaJB $-$ raw probe & " + " & ".join(sg(cf[f'{m}__delta_PersonaJB_minus_fullprobe__{s}']['delta'], 2) for s in SETS) + f" & {sg(dr['delta'], 3)} [{sg(dr['ci_bonf'][0], 3)}, {sg(dr['ci_bonf'][1], 3)}]" + r" \\")
    if m == "llama": lines.append(r"    \midrule")
open(f"{OUT}/qwen_transfer.tex", "w").write(r"""\begin{table}[!htb]
  \centering
  \small
  \caption{\claudewritten{Transfer AUC (either direction) by training set. Each detector is trained on one set (columns) with its regularization strength chosen on a behaviour-disjoint validation split of that set (HarmBench: behaviour- and template-disjoint), refit on the whole set, and scored on the five attack families GCG, PAIR, PAP, GPTFuzz and PEZ, excluding the training set; cells are the mean over those families. Llama-3.1-8B at layer $16$, Qwen3-32B at layer $48$, last prompt token. Random Gaussian is the mean over three draws of as many unit directions as trait vectors. Avg.\ is the mean over training sets with a $95\%$ bootstrap interval over HarmBench behaviours; difference rows give the paired interval with a Bonferroni correction over $16$ comparisons. These are single fits on the full training set, not the $50$-split protocol of Table~\ref{tab:detection}.}}
  \label{tab:qwen-transfer}
  \setlength{\tabcolsep}{3pt}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{ll""" + "c" * len(SETS) + r"""c}
    \toprule
    & & \multicolumn{""" + str(len(SETS)) + r"""}{c}{Training set} & \\
    \cmidrule(lr){3-""" + str(2 + len(SETS)) + r"""}
    Model & Detector & """ + " & ".join(NAME[s] for s in SETS) + r""" & Avg. \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}}
\end{table}
""")
# ---- T3: within-family (nested tuned C) ----
wf = json.load(open(f"{Q}/transfer/diag/within_family_tuned.json"))
WD = [("LR_PersonaJB", "PersonaJB"), ("LR_Gaussian", "Random Gaussian"), ("LR_raw", "Raw-activation probe")]
lines = []
for m in ("llama", "qwen"):
    for i, (d, dn) in enumerate(WD):
        lines.append(f"    {MOD[m] if i == 0 else ''} & {dn} & " + " & ".join(f2(wf[f'{m}__{d}__{s}']['auc']) for s in SETS) + r" \\")
    if m == "llama": lines.append(r"    \midrule")
open(f"{OUT}/qwen_within.tex", "w").write(r"""\begin{table}[!htb]
  \centering
  \small
  \caption{\claudewritten{Within-set AUC (either direction): five-fold cross-validation inside each set with behaviour-disjoint folds. The regularization strength is chosen on a behaviour-disjoint validation split of each training fold. Models, layers and detectors as in Table~\ref{tab:qwen-transfer}.}}
  \label{tab:qwen-within}
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{ll""" + "c" * len(SETS) + r"""}
    \toprule
    Model & Detector & """ + " & ".join(NAME[s] for s in SETS) + r""" \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
# ---- T4: probe methods ----
pm = json.load(open(f"{Q}/probe_methods/out/probe_methods.json")); T = pm["table"]
METH = [("LR_PersonaJB", "LR, PersonaJB features"), ("LR_Gaussian", "LR, random Gaussian features"), ("LR_raw", "LR, raw activation"), ("PCA_LR", "PCA $\\to$ LR"),
        ("DiffMeans_raw", "Difference of means"), ("Concat_LR", "LR, PersonaJB features $\\oplus$ raw"), ("HGB_PersonaJB", "Gradient boosting, PersonaJB features"),
        ("HGB_PCA128", "Gradient boosting, $128$ PCs"), ("MLP_PersonaJB", "MLP, PersonaJB features"), ("MLP_raw", "MLP, raw activation")]
def cellv(c, base):
    lo, hi = c["delta_ci_bonf"]; d = c["delta_vs_LR_raw"]
    ds = "--" if base else (f"\\textbf{{{sg(d, 3)}}}" if (lo > 0 or hi < 0) else sg(d, 3))
    return f"{f2(c['auc'])} {ci(c['ci95'])} & {ds}"
lines = []
for m in ("llama", "qwen"):
    for i, (k, kn) in enumerate(METH):
        lines.append(f"    {MOD[m] if i == 0 else ''} & {kn} & {cellv(T[f'{m}__{k}__single_avg'], k == 'LR_raw')} & {cellv(T[f'{m}__{k}__lofo_avg'], k == 'LR_raw')}" + r" \\")
    if m == "llama": lines.append(r"    \midrule")
open(f"{OUT}/qwen_probe_methods.tex", "w").write(r"""\begin{table}[!htb]
  \centering
  \small
  \caption{\claudewritten{Transfer AUC (either direction) of other probes. Single: train on one set, score the attack families other than the training set, mean over the seven training sets (as Avg.\ in Table~\ref{tab:qwen-transfer}). Leave-one-out: train on all seven sets but one attack family, with each set weighted equally, score the held-out family, mean over the five families. Every hyperparameter (regularization strength, number of PCs, learning rate and leaves, MLP weight decay) is chosen on a behaviour-disjoint validation split of the training data. $\Delta$ is the difference to the raw-activation LR; bold marks a paired $95\%$ bootstrap interval over HarmBench behaviours, Bonferroni-corrected over $""" + str(pm["bonferroni_m"]) + r"""$ comparisons, that excludes zero. The MLP has one hidden layer of $128$ units.}}
  \label{tab:qwen-probe-methods}
  \setlength{\tabcolsep}{3pt}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{llcccc}
    \toprule
    & & \multicolumn{2}{c}{Single training set} & \multicolumn{2}{c}{Leave one family out} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}
    Model & Probe & AUC & $\Delta$ & AUC & $\Delta$ \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}}
\end{table}
""")
print("ok")
