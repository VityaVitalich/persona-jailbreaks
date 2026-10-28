#!/usr/bin/env python3
"""Does baselines/olmo3_compat (harmbench image, transformers 4.54) reproduce native OLMo-3 (serving image, 4.57)?
Run once per image on three benign prompts; saves the HarmBench get_template string (harmbench image) or the native chat
template string, layer-16 and final hidden states at the last prompt token, and next-token logits. With --compare, prints
the differences.   usage: compat_check.py OUT.pt | compat_check.py --compare A.pt B.pt"""
import sys, torch
if sys.argv[1] == "--compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    print("transformers", a["tv"], "vs", b["tv"]); print("templates equal:", a["prompts"] == b["prompts"])
    for k in ("h16", "hlast", "logits"):
        d = (a[k] - b[k]).abs().max().item(); r = d / b[k].abs().max().item(); print(f"{k}: max|diff| {d:.4g} (rel {r:.2e})")
    print("argmax equal:", torch.equal(a["logits"].argmax(-1), b["logits"].argmax(-1)), "top5 overlap", [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a["logits"].topk(5).indices, b["logits"].topk(5).indices)])
    sys.exit()
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
M = "allenai/Olmo-3-7B-Instruct"; Q = ["What is the capital of France?", "Write a haiku about autumn.", "Explain photosynthesis in two sentences."]
try:
    from baselines.model_utils import get_template, load_model_and_tokenizer  # harmbench image: registers olmo3
    t = get_template(M)["prompt"]; prompts = [t.format(instruction=q) for q in Q]
    model, tok = load_model_and_tokenizer(M, dtype="bfloat16", device_map="cuda:0")
except ImportError:
    tok = AutoTokenizer.from_pretrained(M); prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in Q]
    model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).to("cuda:0").eval()  # serving image has no accelerate
print(type(model).__module__, repr(prompts[0][-60:]))
H16, HL, LG = [], [], []
for p in prompts:
    ids = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.no_grad(): o = model(**ids, output_hidden_states=True)
    H16.append(o.hidden_states[17][0, -1].float().cpu()); HL.append(o.hidden_states[-1][0, -1].float().cpu()); LG.append(o.logits[0, -1].float().cpu())
torch.save({"tv": transformers.__version__, "prompts": prompts, "h16": torch.stack(H16), "hlast": torch.stack(HL), "logits": torch.stack(LG)}, sys.argv[1])
print("saved", sys.argv[1])
