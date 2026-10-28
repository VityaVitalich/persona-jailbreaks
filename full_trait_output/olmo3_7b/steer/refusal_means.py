"""OLMo-3-7B-Instruct refusal-direction means, Arditi recipe as used for Gemma (gemma_refusal_dir/out): random.seed(42),
128 harmful + 128 harmless instructions sampled from refusal_direction/dataset/splits/{harmful,harmless}_train.json (in that
order), one user turn through the OLMo chat template (its default system prompt included, as in detection), last prompt
token, every layer. Saves means_harmful.pt / means_harmless.pt ([n_layers+1, d], hidden_states index) + meta.json.
usage: refusal_means.py MODEL_PATH OUTDIR"""
import json, os, random, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
mp, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); here = os.path.dirname(os.path.abspath(__file__))
random.seed(42)
harm = random.sample([d["instruction"] for d in json.load(open(f"{here}/harmful_train.json"))], 128)
harmless = random.sample([d["instruction"] for d in json.load(open(f"{here}/harmless_train.json"))], 128)
tok = AutoTokenizer.from_pretrained(mp); model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16).to("cuda:0").eval()
def means(P):
    acc = None
    for p in P:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad(): hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        h = torch.stack([x[0, -1].float().cpu() for x in hs]); acc = h if acc is None else acc + h
    return acc / len(P)
torch.save(means(harm), f"{out}/means_harmful.pt"); torch.save(means(harmless), f"{out}/means_harmless.pt")
ids = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=True, add_generation_prompt=True)
json.dump({"model": mp, "n_harmful": 128, "n_harmless": 128, "seed": 42, "convention": "hidden_states[k]; layer L output == index L+1; last prompt token",
           "tail_tokens": tok.convert_ids_to_tokens(ids[-5:])}, open(f"{out}/meta.json", "w"), indent=1)
print("done", flush=True)
