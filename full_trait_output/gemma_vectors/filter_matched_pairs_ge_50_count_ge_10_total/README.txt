Folder purpose
==============

Activation source = pre_generation_last_token.
This uses the residual stream hidden state at the final prompt token before generation begins.
It is a prompt-side activation position, useful for early prediction before the model produces an answer.

Trait kept only if at least 10 matched (prompt_index, question_index) pairs satisfy:
positive_score - negative_score >= 50.
This is a paper-inspired interpretation of the ambiguous appendix wording.

Saved file format
=================

Each .pt file contains a dict with:
- vector: Tensor[n_layers, hidden_dim]
- trait: trait name
- activation_position
- filter_name
- passed_filter
- score statistics
- matched pair statistics
