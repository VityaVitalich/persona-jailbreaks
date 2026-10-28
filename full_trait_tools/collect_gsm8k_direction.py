#!/usr/bin/env python3
"""
collect_gsm8k_direction.py  (GPU job)

Builds the off-domain steering control: a correct-vs-incorrect direction from
GSM8k, extracted with the *same recipe* as dmu/PersonaJB (Llama-3.1-8B-Instruct,
layer 16, last prompt token, single user turn, no system prompt) so it is
apples-to-apples with the real arms and differs only in the task it comes from.

For each GSM8k question we (1) grab the L16 activation at the last prompt token,
(2) greedily generate a solution and grade it against the gold answer, then form
    direction = unit( mean(act | correct) - mean(act | incorrect) ).
We also build a response-pooled variant (L16 mean-pooled over the generated
solution tokens), where correctness is actually expressed, as insurance against
the pre-answer prompt token carrying weak signal. Both are audited by AUC on a
held-out split; both are appended to the control bundle, but only the recipe-
matched one is named offdomain_gsm8k__full (the response-pooled one is
offdomain_gsm8k_resp__full). Pick between them from the printed AUCs.

Usage (on a GPU node):
  python full_trait_tools/collect_gsm8k_direction.py --n 1000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = ROOT / "full_trait_output/steering_directions/control_directions_layer16.pt"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = 16
MAX_NEW_TOKENS = 512


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def gold_answer(ans: str) -> str:
    return ans.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def prompt_ids(tok, question: str) -> torch.Tensor:
    conv = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="GSM8k test questions to run")
    ap.add_argument("--test_frac", type=float, default=0.3, help="held-out frac for AUC")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF id or local snapshot path")
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE),
                    help="control directions .pt to append the GSM8k arms to")
    args = ap.parse_args()
    bundle_path = Path(args.bundle)
    raw_out = bundle_path.parent / "gsm8k_direction_data.npz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device})
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))

    prompt_acts, resp_acts, labels = [], [], []
    for i, row in enumerate(ds):
        q, gold = row["question"], gold_answer(row["answer"])
        ids = prompt_ids(tok, q).to(device)
        plen = ids.shape[1]
        with torch.no_grad():
            # (1) last-prompt-token activation, recipe-matched to dmu
            fwd = model(ids, output_hidden_states=True)
            prompt_acts.append(fwd.hidden_states[LAYER + 1][0, -1, :].float().cpu().numpy())
            # (2) greedy solution for grading
            gen = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            resp_ids = gen[0, plen:]
            resp_text = tok.decode(resp_ids, skip_special_tokens=True)
            # (3) L16 mean-pooled over the generated solution tokens
            full = model(gen[:, :], output_hidden_states=True)
            resp_hs = full.hidden_states[LAYER + 1][0, plen:, :]
            resp_acts.append(resp_hs.float().mean(0).cpu().numpy())
        labels.append(int(extract_pred(resp_text) == gold))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)}  running acc={np.mean(labels):.3f}", flush=True)

    prompt_acts = np.asarray(prompt_acts, dtype=np.float64)
    resp_acts = np.asarray(resp_acts, dtype=np.float64)
    y = np.asarray(labels)
    print(f"\nGSM8k: n={len(y)}  correct={int(y.sum())}  incorrect={int((1-y).sum())}  "
          f"acc={y.mean():.3f}")
    if y.sum() < 10 or (1 - y).sum() < 10:
        raise SystemExit("ERROR: too few correct or incorrect to form a stable direction")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_te = int(args.test_frac * len(y))
    te, tr = idx[:n_te], idx[n_te:]

    def build(acts, name):
        d = unit(acts[tr][y[tr] == 1].mean(0) - acts[tr][y[tr] == 0].mean(0))
        auc = roc_auc_score(y[te], acts[te] @ d)
        print(f"  {name:26s} AUC(correct vs incorrect, held-out) = {auc:.3f}")
        return d, float(auc)

    print("\ndirection quality (does it separate correct from incorrect?):")
    prompt_dir, _ = build(prompt_acts, "offdomain_gsm8k (pos -1)")
    resp_dir, _ = build(resp_acts, "offdomain_gsm8k_resp")

    np.savez(raw_out, prompt_acts=prompt_acts, resp_acts=resp_acts, labels=y)

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    for name, d, desc in (
        ("offdomain_gsm8k__full", prompt_dir,
         "GSM8k correct-vs-incorrect diff-of-means, L16 last prompt token (dmu recipe)"),
        ("offdomain_gsm8k_resp__full", resp_dir,
         "GSM8k correct-vs-incorrect diff-of-means, L16 mean-pooled over solution tokens"),
    ):
        bundle["vectors"][name] = torch.tensor(d, dtype=torch.float32)
        bundle["meta"][name] = dict(base=name.replace("__full", ""),
                                    component="full", description=desc)
    torch.save(bundle, bundle_path)
    print(f"\nappended 2 GSM8k arms -> {bundle_path}")
    print(f"raw activations -> {raw_out}")


if __name__ == "__main__":
    main()
