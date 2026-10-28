"""Gemma refusal direction in the orientation of llama_50seed_span.load_direction (Arditi: harmful - harmless),
layer 30, last token, from the steering bundle (which stores -(harmful - harmless))."""
from pathlib import Path
import numpy as np, torch
def _gemma_arditi():
    v = torch.load(Path.home() / "persona-vector-jbing/full_trait_output/gemma_steering_inputs/steering_directions_gemma_layer30.pt",
                   weights_only=False)["vectors"]["refusal_pos1_neg__full"].double().numpy()
    return -v
