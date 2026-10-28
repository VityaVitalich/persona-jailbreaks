#!/usr/bin/env python3
"""
extract_controls_asr.py

Tabulates the steering-controls smoke: per-arm coherent-jailbreak ASR (alpha<=0.5,
HarmBench-refused) and the degeneracy rate, grouped across the 3 random seeds, next
to the published anchor ASR (PersonaJB / refusal / dmu). ASR here already excludes
gibberish -- steer_fixed_direction marks degenerate responses non-jailbroken -- so
a low ASR + high degeneracy is "the vector just breaks the model", not a jailbreak.

Runs on the pulled result files (no GPU); point --root at the synced controls dir.

  python full_trait_tools/extract_controls_asr.py \
      --root full_trait_output/steer_fixed_directions_controls
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANCHORS_JSON = ROOT / "results/steering_alpha_curves.json"


def arm_stats(arm_dir: Path):
    rows = [json.loads(l) for l in (arm_dir / "results.jsonl").read_text().splitlines() if l]
    n = len(rows)
    jb = sum(1 for r in rows if r.get("final_jailbroken"))
    deg = sum(1 for r in rows if r.get("final_degenerate"))
    return n, jb / n if n else float("nan"), deg / n if n else float("nan")


def anchor_asr(a=0.5):
    d = json.load(open(ANCHORS_JSON))
    out = {}
    for k in ("persona", "refusal", "dmu"):
        pairs = d[f"{k}|harmbench"]["pairs"]
        ma = [(e[1] if isinstance(e, (list, tuple)) else e) for e in pairs]
        n = len(ma)
        out[k] = (n, sum(1 for v in ma if v is not None and v <= a) / n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "full_trait_output/steer_fixed_directions_controls"))
    args = ap.parse_args()
    bench_dir = Path(args.root) / "harmbench"

    per_arm = {}
    for arm_dir in sorted(bench_dir.iterdir()):
        if (arm_dir / "results.jsonl").exists():
            per_arm[arm_dir.name] = arm_stats(arm_dir)

    # group the 3 seeds of each random type; keep singletons (gsm8k) as-is
    groups: dict[str, list] = {}
    for name, stats in per_arm.items():
        base = re.sub(r"_s\d+__full$", "", name).replace("__full", "")
        groups.setdefault(base, []).append(stats)

    print(f"\n{'arm':24s} {'n':>5s} {'ASR(coherent)':>14s} {'degenerate':>11s}")
    print("-" * 58)
    for base, arms in groups.items():
        ns = [n for n, _, _ in arms]
        asrs = [a for _, a, _ in arms]
        degs = [d for _, _, d in arms]
        n = ns[0]
        if len(arms) > 1:
            print(f"{base:24s} {n:5d} {sum(asrs)/len(asrs):8.3f} "
                  f"[{min(asrs):.3f},{max(asrs):.3f}] {sum(degs)/len(degs):11.3f}")
        else:
            print(f"{base:24s} {n:5d} {asrs[0]:14.3f} {degs[0]:11.3f}")

    print(f"\n{'published anchor':24s} {'n':>5s} {'ASR(a<=0.5)':>14s}")
    print("-" * 46)
    for k, (n, a) in anchor_asr().items():
        print(f"{k:24s} {n:5d} {a:14.3f}")


if __name__ == "__main__":
    main()
