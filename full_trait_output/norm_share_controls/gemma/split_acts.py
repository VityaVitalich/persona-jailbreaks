#!/usr/bin/env python3
"""Split gemma_acts.pt into the file layout the ported scripts read (same as the Llama files):
wiki/wiki_acts.pt {"raw": {...}}, wiki/norobots_acts.pt {"chat": {...}}, synthetic_v2_acts.pt {id: ...}."""
from pathlib import Path
import torch
G = Path(__file__).resolve().parent; a = torch.load(G / "gemma_acts.pt", map_location="cpu")
(G / "wiki").mkdir(exist_ok=True)
torch.save({"raw": {k: v for k, v in a.items() if k.startswith("wiki_")}}, G / "wiki/wiki_acts.pt")
torch.save({"chat": {k: v for k, v in a.items() if k.startswith("nr_")}}, G / "wiki/norobots_acts.pt")
torch.save({k[3:]: v for k, v in a.items() if k.startswith("rp:")}, G / "synthetic_v2_acts.pt")
for f in ("wiki_passages.jsonl", "norobots_prompts.jsonl"):
    if not (G / "wiki" / f).exists(): (G / "wiki" / f).symlink_to(G.parent / "wiki" / f)
print({k: sum(1 for x in a if x.startswith(k)) for k in ("wiki_", "nr_", "rp:", "verify:")})
