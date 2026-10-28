#!/usr/bin/env python3
"""
extract_ablation_cells.py

Tabulates the directional-ablation runs (jailbreaking_tools/ablate_fixed_direction.py)
on the pulled result files: per arm and benchmark, the three exclusive outcomes on
the initially-complied pool -- still jailbroken / refused / degenerate -- with
bootstrap 95% CIs, and the refused-rate gain over the no-hook `none` arm as a
PAIRED per-prompt delta (same prompts in every arm, joined on pair_id).

Scored by the judge that produced the pool labels (judge_harmbench_responses.py:
behavior + response), the same one the complied/refused split is defined by, so the
`none` arm reproduces the pool by construction and is the reference row. Cell keys
are `<arm>|<bench>`.

Bootstrap follows steering_bootstrap_cis.py: 10,000 resamples, seed 0, HarmBench
resampled by behaviour (1245 pairs over ~159 behaviours; pairs would understate
the interval), WildJB by prompt. The three seeds of each random control are also
pooled into one `random_full` / `random_persona` row.

Runs on the synced copy (no GPU):
  rsync -a jumphost:/mnt/dlab/scratch/dlabscratch1/moskvore/persona_steer/outputs_ablation/ \
        full_trait_output/ablate_fixed_directions/
  python full_trait_tools/extract_ablation_cells.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 10000
BENCHES = ["harmbench", "wildjb"]
DIRS = ["persona", "refusal", "dmu", "utilitarian", "closure_seeking", "edgy",
        "random_full", "random_persona"]
PLACEMENTS = ["", "@16"]        # "" = all layers (Arditi); @16 = the steering layer


def split_arm(name):
    """'persona@16' -> ('persona', '@16'); 'persona' -> ('persona', '')."""
    return tuple(name.rsplit("@", 1)[0:1]) + (("@" + name.rsplit("@", 1)[1]) if "@" in name
                                              else "",)


def load(arm_dir: Path):
    rows = [json.loads(l) for l in (arm_dir / "results.jsonl").read_text().splitlines() if l]
    return {str(r["pair_id"]): r for r in rows}


def outcome(r, key):
    """1 = still jailbroken, 0 = refused, -1 = degenerate (never judged),
    None = judge parse failure.

    A parse failure is not a refusal: counting it as one would credit the
    ablation with an effect the judge never reported. It is excluded from the
    rates (so every rate shares the denominator n - parse failures) and reported
    as its own count.
    """
    if r.get("final_degenerate"):
        return -1
    v = r.get(key)
    if v is None:
        return None
    return 1 if v else 0


def cluster_ids(rows, bench):
    return [r["behavior_id"] if bench == "harmbench" else str(r["pair_id"]) for r in rows]


def boot_mean(values, clusters, rng):
    """Mean of `values` with a cluster bootstrap over `clusters`."""
    values = np.asarray(values, float)
    cl = np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    sizes = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    draws = sums[idx].sum(1) / sizes[idx].sum(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def cell(arm_rows, none_rows, bench, key, rng):
    ids = sorted(arm_rows)
    oc_all = {i: outcome(arm_rows[i], key) for i in ids}
    scored = [i for i in ids if oc_all[i] is not None]
    rows = [arm_rows[i] for i in scored]
    oc = np.array([oc_all[i] for i in scored])
    cl = cluster_ids(rows, bench)
    out = {"n": len(ids), "n_scored": len(scored), "parse_failures": len(ids) - len(scored)}
    if not scored:
        return out
    for name, val in (("still_jailbroken", (oc == 1)), ("refused", (oc == 0)),
                      ("degenerate", (oc == -1))):
        m, lo, hi = boot_mean(val.astype(float), cl, rng)
        out[name] = {"rate": m, "lo": lo, "hi": hi}
    if none_rows is not None:
        # a pair enters the paired delta only if BOTH arms were scored
        common = [i for i in scored
                  if i in none_rows and outcome(none_rows[i], key) is not None]
        if common:
            d = np.array([(oc_all[i] == 0) - (outcome(none_rows[i], key) == 0)
                          for i in common], float)
            m, lo, hi = boot_mean(d, cluster_ids([arm_rows[i] for i in common], bench), rng)
            out["refused_gain_vs_none"] = {"delta": m, "lo": lo, "hi": hi, "n_paired": len(common)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "full_trait_output/ablate_fixed_directions"))
    ap.add_argument("--out", default=str(ROOT / "results/ablation_cells.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    cells = {}
    for bench in BENCHES:
        bdir = Path(args.root) / bench
        if not bdir.exists():
            continue
        arms = {d.name: load(d) for d in sorted(bdir.iterdir()) if (d / "results.jsonl").exists()}
        # pool the seeds of each random control into one arm
        pooled = defaultdict(dict)
        for name, rows in arms.items():
            m = re.match(r"(random_\w+?)_s\d+(@\d+)?$", name)
            if m:
                key = m.group(1) + (m.group(2) or "")
                for pid, r in rows.items():
                    pooled[key][f"{name}:{pid}"] = dict(r, pair_id=f"{name}:{pid}")
        none_rows = arms.get("none")
        for name, rows in list(arms.items()) + list(pooled.items()):
            # the pooled controls carry seed-prefixed ids: pair them with `none`
            # by stripping the prefix
            nr = none_rows
            if name in pooled and none_rows is not None:
                nr = {pid: none_rows.get(pid.split(":", 1)[1]) for pid in rows}
                nr = {k: v for k, v in nr.items() if v is not None}
            cells[f"{name}|{bench}"] = cell(rows, nr if name != "none" else None,
                                            bench, "label_jailbroken", rng)

    Path(args.out).write_text(json.dumps(cells, indent=1, sort_keys=True))
    print(f"wrote {len(cells)} cells -> {args.out}\n")

    present = [b for b in BENCHES if any(k.endswith(f"|{b}") for k in cells)]
    for place in PLACEMENTS:
        if not any(f"{d}{place}|{b}" in cells for d in DIRS for b in present):
            continue
        title = {"": "ALL LAYERS (Arditi)"}.get(place, f"LAYER {place[1:]} ONLY")
        print(f"== {title}   rates on the initially-complied pool; "
              f"gain = refused - refused[none], paired 95% CI")
        hdr = f"{'arm':18s}" + "".join(
            f" | {b:>9s} {'still JB':>8s} {'refused':>8s} {'degen':>6s} {'NA':>3s}"
            f" {'gain vs none':>22s}" for b in present)
        print(hdr)
        names = [f"{d}{place}" for d in DIRS
                 if any(f"{d}{place}|{b}" in cells for b in present)]
        if place == "":
            names += [n for n in ["none"] if any(f"{n}|{b}" in cells for b in present)]
        for name in names:
            if re.search(r"random_\w+_s\d+(@\d+)?$", name):
                continue
            line = f"{name:18s}"
            for b in present:
                c = cells.get(f"{name}|{b}")
                if c is None or "refused" not in c:
                    line += f" | {'':>9s} {'--':>8s} {'--':>8s} {'--':>6s} {'--':>3s} {'':>22s}"
                    continue
                g = c.get("refused_gain_vs_none")
                gs = (f"{g['delta']:+.3f} [{g['lo']:+.3f},{g['hi']:+.3f}]" if g else "")
                line += (f" | {'n=' + str(c['n_scored']):>9s} {c['still_jailbroken']['rate']:8.3f} "
                         f"{c['refused']['rate']:8.3f} {c['degenerate']['rate']:6.3f} "
                         f"{c['parse_failures']:3d} {gs:>22s}")
            print(line)
        print()


if __name__ == "__main__":
    main()
