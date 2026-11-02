"""Build gen_inputs.jsonl: SST-2 validation (872), AG News test (250 per class, seed 0), Geometry-of-Truth cities (all)."""
import sys, json, csv, collections, random
import pyarrow.parquet as pq
D, OUT = sys.argv[1], sys.argv[2]
rows = []
t = pq.read_table(f"{D}/sst2_val.parquet").to_pydict()
for i, (s, l) in enumerate(zip(t["sentence"], t["label"])): rows.append({"set": "sst2", "pair_id": f"sst2_{i}", "prompt": s.strip(), "label": int(l)})
t = pq.read_table(f"{D}/agnews_test.parquet").to_pydict()
by = collections.defaultdict(list)
for i, (s, l) in enumerate(zip(t["text"], t["label"])): by[int(l)].append((i, s))
rng = random.Random(0)
for l in sorted(by):
    for i, s in sorted(rng.sample(by[l], 250)): rows.append({"set": "agnews", "pair_id": f"agnews_{i}", "prompt": s.strip(), "label": l})
for i, r in enumerate(csv.DictReader(open(f"{D}/cities.csv"))): rows.append({"set": "cities", "pair_id": f"cities_{i}", "prompt": r["statement"].strip(), "label": int(r["label"])})
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
c = collections.Counter((r["set"], r["label"]) for r in rows); print(sorted(c.items())); print("total", len(rows))
