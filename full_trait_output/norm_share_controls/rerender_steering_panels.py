#!/usr/bin/env python3
"""
Local re-render of fig_steering_panels with a fourth group, controls (HarmBench only; the
controls were not run on JBB / WildJB, so those panels leave the rows empty). Runs the paper generator
full_trait_tools/plot_steering_panels.py unchanged except for the patches below; output goes to
norm_share_controls/figs/, the paper figure is untouched.

Control rows (HarmBench originally-refused pool, n=300 seed 0, cumulative alpha 0.1-0.5):
  detector-normal arms on 229 random activations of Wikipedia / No Robots / Alpaca (3 draws each),
  top-3 single directions of each dictionary by HarmBench AUC, and the earlier controls
  (isotropic random, random trait mixtures, shuffled PersonaJB weights, GSM8k diff-of-means).
Multi-arm rows: dot = mean over the arms, trace = lowest Wilson lower bound to highest Wilson upper bound
of the arms. New arms ran on the official Llama snapshot; the persona / trait / refusal rows and the
earlier controls come from the earlier runs (see the steering README).
"""
import glob, json, math, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = REPO / "full_trait_tools/plot_steering_panels.py"
NEW = HERE / "steer_v2_results"
OLD = REPO / "full_trait_output/outputs_controls_cumulative/harmbench"


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def arm(path):
    R = [json.loads(l) for l in open(path)]
    return wilson(sum(bool(r["final_jailbroken"]) for r in R), len(R))


def family(paths):
    a = [arm(p) for p in paths]
    return sum(x[0] for x in a) / len(a), min(x[1] for x in a), max(x[2] for x in a)


FAM = [  # label, files
    ("Detector on Wikipedia", [NEW / f"extra_wikijb_d{i}.jsonl" for i in range(3)]),
    ("Detector on No Robots", [NEW / f"extra_nrjb_d{i}.jsonl" for i in range(3)]),
    ("Detector on Alpaca", [NEW / f"alpaca1_alpacajb_d{i}.jsonl" for i in range(3)]),
    ("Top Alpaca PCs", [NEW / f"alpaca2_sel_pca_{i}.jsonl" for i in (1, 2, 3)]),
    ("Top Alpaca Prompts", [NEW / f"alpaca2_sel_acts_{i}.jsonl" for i in (1, 2, 3)]),
    ("Top No Robots Prompts", [NEW / f"extra_sel_nr_{i}.jsonl" for i in (1, 2, 3)]),
    ("Top Wikipedia Passages", [NEW / f"extra_sel_wiki_{i}.jsonl" for i in (1, 2, 3)]),
    ("GSM8k Direction", [OLD / "offdomain_gsm8k__full/results.jsonl"]),
    ("Random Trait Mixture", [OLD / f"random_traitcombo_s{i}__full/results.jsonl" for i in range(3)]),
    ("Random Direction", [OLD / f"random_full_s{i}__full/results.jsonl" for i in range(3)]),
]
ctrl = {lab: family(fs) for lab, fs in FAM}
for lab, (m, lo, hi) in ctrl.items():
    print(f"  {lab:24s} {100*m:5.1f}  [{100*lo:.1f}, {100*hi:.1f}]")

src = GEN.read_text()
src = src.replace('OUT = REPO / "full_trait_output/plots/paper_figs"', f'OUT = Path(r"{HERE / "figs"}")')
src = src.replace('"std": ("Standard Attacks", C_STD)}',
                  '"std": ("Standard Attacks", C_STD),\n         "ctrl": ("Controls", "#7A5195")}')
inject = "CTRL = " + repr({k: v for k, v in ctrl.items()}) + "\n" + \
    "for _lab, (_m, _lo, _hi) in CTRL.items():\n" + \
    "    CIS[f'ctrl:{_lab}|harmbench'] = {'asr': _m, 'lo': _lo, 'hi': _hi}\n" + \
    "    ROWS.append((_lab, f'ctrl:{_lab}', _m * 100, None, None, 'ctrl'))\n"
src = src.replace("# (display name, cluster key, unsteered rate in percent)", inject + "# (display name, cluster key, unsteered rate in percent)")
src = src.replace('    d = CIS.get(f"{key}|{bench}")', '    if table_value is None:\n        return None, None, None, False\n    d = CIS.get(f"{key}|{bench}")')
src = src.replace("        vals = [cell(r[1], key, r[2 + k]) for r in rows]",
                  "        vals = [v for v in (cell(r[1], key, r[2 + k]) for r in rows) if v[0] is not None]")
src = src.replace("            x, lo, hi, ok = cell(key, bkey, v[k])", "            if v[k] is None:\n                continue\n            x, lo, hi, ok = cell(key, bkey, v[k])")
src = src.replace("ncol=3, fontsize=6.9*fs", "ncol=4, fontsize=6.9*fs")
start = src.index('build("fig_steering_panels"')
src = src[:start] + 'build("fig_steering_panels_controls", height=4.3, gap=0.55, left=0.235, legy=-0.117, fs=0.90)\n'
assert src.count("ctrl") >= 3
os.makedirs(HERE / "figs", exist_ok=True)
os.chdir(REPO)
exec(compile(src, str(GEN), "exec"), {"__name__": "__main__", "__file__": str(GEN)})
