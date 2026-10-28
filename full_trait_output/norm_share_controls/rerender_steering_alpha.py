#!/usr/bin/env python3
"""
Local re-render of fig_steering_alpha with control curves in the HarmBench panel (controls were not run
on JBB / WildJB). Control arms: HarmBench originally-refused pool n=300 seed 0, schedule 0.1-0.5; a family's
draws are pooled into one curve (= the mean of the draws). Paper arms are unchanged (their full refused pool).
Runs full_trait_tools/plot_steering_alpha_curve.py with the ARMS list extended and the data file augmented;
output figs/fig_steering_alpha_controls.pdf.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = REPO / "full_trait_tools/plot_steering_alpha_curve.py"
NEW = HERE / "steer_v2_results"
OLD = REPO / "full_trait_output/outputs_controls_cumulative/harmbench"
C1, C2 = "#7A5195", "#B39BC8"
FAM = [  # key, label, colour, ls, files
    ("ctrl_wikijb", "Detector on Wikipedia", C1, "-", [NEW / f"extra_wikijb_d{i}.jsonl" for i in range(3)]),
    ("ctrl_alpacajb", "Detector on Alpaca", C1, "--", [NEW / f"alpaca1_alpacajb_d{i}.jsonl" for i in range(3)]),
    ("ctrl_selwiki", "Top Wikipedia Passages", C2, "-", [NEW / f"extra_sel_wiki_{i}.jsonl" for i in (1, 2, 3)]),
    ("ctrl_gsm8k", "GSM8k Direction", C2, "--", [OLD / "offdomain_gsm8k__full/results.jsonl"]),
    ("ctrl_random", "Random Direction", C2, ":", [OLD / f"random_full_s{i}__full/results.jsonl" for i in range(3)]),
]
data = json.load(open(REPO / "results/steering_alpha_curves.json"))
for key, _, _, _, files in FAM:
    pairs = []
    for f in files:
        for l in open(f):
            r = json.loads(l)
            pairs.append([f"{Path(f).stem}:{r['behavior_id']}", r["min_alpha_to_jailbreak"] if r["final_jailbroken"] else None])
    data[f"{key}|harmbench"] = {"schedule": [0.1, 0.2, 0.3, 0.4, 0.5], "pairs": pairs}
aug = HERE / "figs/steering_alpha_curves_with_controls.json"; aug.parent.mkdir(exist_ok=True)
json.dump(data, open(aug, "w"))

src = GEN.read_text()
src = src.replace('    ("edgy", "Edgy", C_TRAIT, ":", False),\n]',
                  '    ("edgy", "Edgy", C_TRAIT, ":", False),\n' +
                  "".join(f'    ("{k}", "{lab}", "{c}", "{ls}", False),\n' for k, lab, c, ls, _ in FAM) + "]")
src = src.replace('loc="lower center", ncol=6, fontsize=7,', 'loc="lower center", ncol=4, fontsize=7,')
src = src.replace("fig.tight_layout(rect=(0, 0.10, 1, 1), pad=0.4, w_pad=0.8)", "fig.tight_layout(rect=(0, 0.20, 1, 1), pad=0.4, w_pad=0.8)")
sys.argv = [str(GEN), "--data", str(aug), "--figh", "2.6", "--out", str(HERE / "figs/fig_steering_alpha_controls.pdf")]
exec(compile(src, str(GEN), "exec"), {"__name__": "__main__", "__file__": str(GEN)})
