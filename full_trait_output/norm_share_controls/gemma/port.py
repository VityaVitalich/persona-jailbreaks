#!/usr/bin/env python3
"""Make the Gemma copy of a norm_share_controls analysis script: python3.13 port.py <path relative to norm_share_controls>"""
import re, sys
from pathlib import Path
G = Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma"); NSC = G.parent
src = sys.argv[1]; s = (NSC / src).read_text()
PRE = 'exec(compile(open(Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py")).read(), "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/preamble_gemma.py", "exec"))'
s = re.sub(r'exec\(compile\(open\(HERE / "persona_part_test\.py"\)\.read\(\)\.split\([^)]*\)\[0\], str\(HERE / "persona_part_test\.py"\), "exec"\)\)', lambda m: PRE, s)
s = s.replace('S.CFG["llama"]', 'S.CFG["gemma"]')
s = re.sub(r'weights_only=False\), 16\)', 'weights_only=False), 30)', s)
s = s.replace('load_direction(Path.home() / "refusal_direction/runs/llama-3.1-8b-instruct/direction_layer16_lasttoken.pt")', '_gemma_arditi()')
s = s.replace('from llama_50seed_span import load_direction', 'sys.path.insert(0, "/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma"); from gemma_common import _gemma_arditi')
s = s.replace('_llama.npz', '_gemma.npz').replace('_llama.json', '_gemma.json')
s = s.replace('OUT / "roleplay_check/synthetic_v2_acts.pt"', 'Path("/Users/viktor/persona-vector-jbing/full_trait_output/norm_share_controls/gemma/synthetic_v2_acts.pt")')
s = s.replace('OUT = HERE.parent\n', 'OUT = HERE.parent.parent\n').replace('OUT = HERE.parent;', 'OUT = HERE.parent.parent;')
s = s.replace('OUT = NS.parent\n', 'OUT = NS.parent.parent\n')
s = re.sub(r'\b4096\b', '5376', s); s = re.sub(r'\b1145\b', '1200', s); s = re.sub(r'\b229\b', '240', s)
s = s.replace('layer 16', 'layer 30').replace('Llama-3.1-8B-Instruct', 'Gemma-4-31B-it').replace('Llama-3.1-8B', 'Gemma-4-31B-it')
s = '# GEMMA PORT of norm_share_controls/' + src + ' (gemma/port.py); model-specific inputs swapped, logic unchanged.\n' + s
(G / src).write_text(s); print("wrote", G / src)
