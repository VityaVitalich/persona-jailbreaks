#!/usr/bin/env python3
"""
Larger synthetic set for the three-group check (jailbroken / refused / benign-complied).

Harmful requests: JailbreakBench harmful (100) + AdvBench (520) + StrongREJECT (313), deduplicated on
normalised text; StrongREJECT rows sourced from HarmBench are dropped (the detector is trained on
HarmBench). Benign requests: JailbreakBench benign (100) + Alpaca instructions without an input
field, sampled (seed 0) to match the harmful count.
Each request: the plain prompt + 6 of the 22 role-play frames of build_synthetic.py (2 per frame
type, sampled per request with seed 0).

Output: synthetic_v2_prompts.jsonl {id, harm, source, req_idx, frame, frame_type, prompt}
"""
import csv, json, random, re
from pathlib import Path
import pandas as pd
from build_synthetic import FRAMES

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def main():
    harmful, seen = [], set()
    def add(text, source):
        k = norm(text)
        if k and k not in seen:
            seen.add(k); harmful.append((text.strip(), source))
    for r in csv.DictReader(open(SRC / "jbb_harmful.csv", encoding="utf-8")): add(r["Goal"], "jbb")
    for r in csv.DictReader(open(SRC / "advbench.csv", encoding="utf-8")): add(r["goal"], "advbench")
    for r in csv.DictReader(open(SRC / "strongreject.csv", encoding="utf-8")):
        if "harmbench" not in r["source"].lower(): add(r["forbidden_prompt"], "strongreject")

    rng = random.Random(0)
    benign = [(r["Goal"].strip(), "jbb_benign") for r in csv.DictReader(open(SRC / "jbb_benign.csv", encoding="utf-8"))]
    alp = pd.read_parquet(SRC / "alpaca.parquet")
    alp = [t.strip() for t in alp.loc[alp["input"] == "", "instruction"] if 20 <= len(t) <= 300]
    benign += [(t, "alpaca") for t in rng.sample(alp, len(harmful) - len(benign))]

    by_type = {}
    for name, (ftype, _) in FRAMES.items(): by_type.setdefault(ftype, []).append(name)
    out = []
    for harm, reqs in [("harmful", harmful), ("benign", benign)]:
        for i, (req, source) in enumerate(reqs):
            base = dict(harm=harm, source=source, req_idx=i)
            out.append(dict(id=f"{harm}_{i}_plain", frame="plain", frame_type="plain", prompt=req, **base))
            for ftype in sorted(by_type):
                for name in rng.sample(sorted(by_type[ftype]), 2):
                    tpl = FRAMES[name][1]
                    mid = tpl[:tpl.index("{req}")].endswith((", ", "to: "))
                    out.append(dict(id=f"{harm}_{i}_{name}", frame=name, frame_type=ftype,
                                    prompt=tpl.format(req=req[0].lower() + req[1:] if mid else req), **base))
    with open(HERE / "synthetic_v2_prompts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    from collections import Counter
    print(len(out), "prompts;", Counter(o["harm"] for o in out if o["frame"] == "plain"),
          Counter((o["harm"], o["source"]) for o in out if o["frame"] == "plain"))


if __name__ == "__main__":
    main()
