PCA-based assistant-like axis
=============================

This folder stores a PCA axis computed from a set of trait vectors.

Method:
1. Load all trait vectors from the input directory.
2. Stack vectors into shape [n_traits, n_selected_layers, hidden_dim].
3. For each selected layer row independently, run PCA over the trait vectors.
4. Take PC1 at each selected layer row as the saved axis.

Important note:
This is a PCA-based assistant-like axis, PC1 of trait-vector space.
It is not necessarily the paper's exact contrast-vector Assistant Axis.

The saved axis rows correspond to selected original model layers:
[15, 20, 25, 30, 35, 40, 45]

For example, if vector_layers = [15,20,25,30,35,40,45], then axis row 3 is model layer 30.

Input directory: /scratch/gemma_trait_output/traits40_vectors/pre_generation_last_token/all_traits_no_filter
Output file: /scratch/gemma_trait_output/traits40_axes/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt
