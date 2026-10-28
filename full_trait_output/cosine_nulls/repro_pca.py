"""Reproduce the PCA-pole cosines of tab:pca-personas (pca_component_interpretation.py recipe)."""
import json, sys
import numpy as np, torch
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_rows(model, pop):
    base = {"llama": "full_trait_output/harmbench_activations", "gemma": "full_trait_output/harmbench_activations_gemma"}[model]
    layer = {"llama": "16", "gemma": "30"}[model]
    rows = [json.loads(l) for l in open(f"{base}/classified_responses.jsonl") if l.strip()]
    acts = torch.load(f"{base}/activations.pt", map_location="cpu", weights_only=False)
    rows = [r for r in rows if r["attack_type"] == "human_jailbreak"]
    if pop == "filtered":
        c = defaultdict(lambda: [0, 0])
        for r in rows:
            c[r["behavior_id"]][0] += 1; c[r["behavior_id"]][1] += r["jailbroken"]
        keep = {b for b, (t, j) in c.items() if 0.2 <= j / t <= 0.8}
        rows = [r for r in rows if r["behavior_id"] in keep]
    rows = [r for r in rows if r["pair_id"] in acts]
    X = np.stack([acts[r["pair_id"]][layer].float().numpy() for r in rows])
    y = np.array([int(r["jailbroken"]) for r in rows])
    return X, y, rows

def load_traits(model):
    if model == "llama":
        T = np.load("full_trait_output/trait_matrix_layer16.npy"); names = json.load(open("full_trait_output/trait_names_layer16.json"))
    else:
        T = np.load("full_trait_output/trait_pc_sweep/_gemma_ref/trait_matrix_layer30.npy"); names = json.load(open("full_trait_output/trait_pc_sweep/_gemma_ref/trait_names_layer30.json"))
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    return T.astype(np.float64), names

if __name__ == "__main__":
    model, pop = sys.argv[1], sys.argv[2]
    X, y, rows = load_rows(model, pop)
    T, names = load_traits(model)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=5, random_state=42).fit(Xs)
    print(model, pop, X.shape, y.mean().round(3), "var", (100 * pca.explained_variance_ratio_).round(1))
    for k in range(4):
        c = T @ pca.components_[k]
        o = np.argsort(-c)
        print(f"PC{k+1}", [(names[i], round(c[i], 2)) for i in o[:3]], [(names[i], round(c[i], 2)) for i in o[::-1][:3]])
