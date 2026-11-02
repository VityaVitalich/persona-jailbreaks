"""Build gen_inputs2.jsonl (harder general tasks): CoLA validation (all), MNLI validation-matched (400 per class),
MMLU answer correctness (750 questions x {correct, one random wrong option}), Geometry-of-Truth neg_cities and
common_claim_true_false (up to 750 per label), and synthetic addition correctness (750 true + 750 near-miss false)."""
import sys, json, csv, collections, random
import pyarrow.parquet as pq
D, OUT = sys.argv[1], sys.argv[2]
rng = random.Random(0); rows = []
t = pq.read_table(f"{D}/cola_val.parquet").to_pydict()
for i, (s, l) in enumerate(zip(t["sentence"], t["label"])): rows.append({"set": "cola", "pair_id": f"cola_{i}", "prompt": s.strip(), "label": int(l)})
t = pq.read_table(f"{D}/mnli_valm.parquet").to_pydict(); by = collections.defaultdict(list)
for i, (p, h, l) in enumerate(zip(t["premise"], t["hypothesis"], t["label"])):
    if l in (0, 1, 2): by[l].append((i, f"Premise: {p.strip()}\nHypothesis: {h.strip()}"))
for l in sorted(by):
    for i, s in sorted(rng.sample(by[l], 400)): rows.append({"set": "mnli", "pair_id": f"mnli_{i}", "prompt": s, "label": l})
t = pq.read_table(f"{D}/mmlu_test.parquet").to_pydict(); idx = sorted(rng.sample(range(len(t["question"])), 750))
for i in idx:
    q, ch, a = t["question"][i].strip(), t["choices"][i], int(t["answer"][i])
    w = rng.choice([j for j in range(len(ch)) if j != a])
    for lab, j in ((1, a), (0, w)): rows.append({"set": "mmlu", "pair_id": f"mmlu_{i}_{lab}", "prompt": f"Question: {q}\nAnswer: {str(ch[j]).strip()}", "label": lab})
for name in ("neg_cities", "common_claim_true_false"):
    by = collections.defaultdict(list)
    for i, r in enumerate(csv.DictReader(open(f"{D}/{name}.csv"))): by[int(float(r["label"]))].append((i, r["statement"].strip()))
    n = min(750, min(len(v) for v in by.values())); tag = "negcities" if name == "neg_cities" else "claims"
    for l in sorted(by):
        for i, s in sorted(rng.sample(by[l], n)): rows.append({"set": tag, "pair_id": f"{tag}_{i}", "prompt": s, "label": l})
for k in range(750):
    a, b = rng.randint(10, 999), rng.randint(10, 999); c = a + b
    rows.append({"set": "arith", "pair_id": f"arith_{k}_1", "prompt": f"{a} + {b} = {c}", "label": 1})
    a, b = rng.randint(10, 999), rng.randint(10, 999); d = rng.choice([1, 2, 9, 10, 11, 100]) * rng.choice([-1, 1])
    rows.append({"set": "arith", "pair_id": f"arith_{k}_0", "prompt": f"{a} + {b} = {a + b + d}", "label": 0})
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
c = collections.Counter((r["set"], r["label"]) for r in rows); print(sorted(c.items())); print("total", len(rows))
