PCA-based assistant-like axis
============================

This folder stores a per-layer PCA axis computed from a filtered set of trait vectors.
Method:
1. Load all trait vectors from the input directory.
2. Stack vectors into shape [n_traits, n_layers, hidden_dim].
3. For each layer independently, run PCA over the trait vectors.
4. Take PC1 at each layer as the saved axis.

Important note:
This is a PCA-based assistant-like axis (PC1 of trait space), not the paper's contrast-vector Assistant Axis.

Input directory: full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total
Output file: full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt
