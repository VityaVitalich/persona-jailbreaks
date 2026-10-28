"""Markdown tables from out/q1_gemma.json: (1) every detector's selected readout and folded transfer AUC, (2) deltas vs full."""
import json, numpy as np
r = json.load(open("out/q1_gemma.json")); S, D, T, I = r["selected"], r["delta"], r["test"], r["info"]
NAMES = {"PersonaJB": "PersonaJB (240 traits)", "raw": "Linear probe (raw)", "PCA": "PCA-240 (train acts)", "gauss": "Random Gaussian 240 (3 draws)",
         "alpaca_pca": "Alpaca subspace (PCA-240)", "wiki_pca": "Wikipedia subspace (PCA-240)", "jb_dir": "JB direction (Δμ)", "refusal": "Refusal direction",
         "alpaca_mean": "Alpaca mean direction", "wiki_mean": "Wikipedia mean direction"}
print(f"train n={r['n_train']} ({r['n_pos_train']} jailbroken); test (n, jailbroken): {r['n_test']}\n")
print("| Detector | readout | C | val | " + " | ".join(T) + " | avg [95% CI] |\n|" + "---|" * (len(T) + 5))
for d, n in NAMES.items():
    ss = [S[f"gauss_d{k}__full__reselect"] for k in range(3)] if d == "gauss" else [S[f"{d}__full__reselect"]]
    sm = S[f"{d}__full__summary"]
    ro = "/".join(f"L{s['layer']} p{s['position']}" for s in ss); C = "/".join("—" if s["C"] is None else f"{s['C']:g}" for s in ss)
    print(f"| {n} | {ro} | {C} | {np.mean([s['val_auc'] for s in ss]):.3f} | " + " | ".join(f"{np.mean([s['test_auc'][t] for s in ss]):.3f}" for t in T)
          + f" | **{sm['avg']:.3f}** [{sm['avg_ci95'][0]:.3f}, {sm['avg_ci95'][1]:.3f}] |")
f = lambda k: (lambda v: f"{v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")(D[k]) if k in D else "—"
print("\n| Detector | full | − persona (same readout) | − persona (re-selected) | − random 240 (same readout) | only persona (same readout) | only persona (re-selected) |\n|" + "---|" * 7)
for d, n in NAMES.items():
    print(f"| {n} | {S[f'{d}__full__summary']['avg']:.3f} | " + " | ".join(f(f"{d}__{c}") for c in ("no_persona__fixed", "no_persona__reselect", "no_random__fixed", "only_persona__fixed", "only_persona__reselect")) + " |")
print("\npersona share of the selected readout's ingredients (full condition):")
for d in NAMES:
    if d in ("PersonaJB", "raw"): continue
    s = S["gauss_d0__full__reselect" if d == "gauss" else f"{d}__full__reselect"]; k = f"L{s['layer']}__p{s['position']}__full"
    print(f"  {d:12s} L{s['layer']} p{s['position']}: {I[k]['persona_share'][d]:.3f}  (train activations {I[k]['persona_share_train_acts']:.3f}, random span {I[k]['random_share_train_acts']:.3f})")
