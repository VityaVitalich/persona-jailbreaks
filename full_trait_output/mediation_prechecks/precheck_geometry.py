# Pre-checks for MEDIATION_PLAN.md: geometry of the persona patch on stored L16 last-token activations.
import json, numpy as np, torch, warnings
warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
T = np.load('trait_matrix_layer16.npy').astype(np.float64)
Q = np.linalg.svd(T, full_matrices=False)[2].T
A = torch.load('harmbench_activations/activations.pt', weights_only=False)
rows = [json.loads(l) for l in open('harmbench_activations/classified_responses.jsonl')]
W, S = [18, 73, 30, 29, 71], [11, 3, 32, 77]
H = {(r['behavior_id'], r['jailbreak_idx']): A[r['pair_id']]['16'].double().numpy() for r in rows}
behs = sorted({r['behavior_id'] for r in rows})
def pairs(t0s, t1s):
    return [(b, a, c) for b in behs for a in t0s for c in t1s if a != c]
def stats(ps):
    D = np.stack([H[(b, c)] - H[(b, a)] for b, a, c in ps])
    PD = D @ Q @ Q.T
    return D, PD
D, PD = stats(pairs(W, S))
share = lambda D, B: np.mean(np.sum((D @ B) ** 2, 1) / np.sum(D ** 2, 1))
print('weak->strong n=%d  T share %.3f  median|PD| %.2f  median|D| %.2f' % (len(D), share(D, Q), np.median(np.linalg.norm(PD, axis=1)), np.median(np.linalg.norm(D, axis=1))))
for name, ps in [('weak->weak', pairs(W, W)), ('strong->strong', pairs(S, S))]:
    d, pd = stats(ps); print('%s n=%d  T share %.3f  median|PD| %.2f' % (name, len(d), share(d, Q), np.median(np.linalg.norm(pd, axis=1))))
Qr = np.linalg.qr(rng.standard_normal((4096, 229)))[0]; print('haar share %.3f' % share(D, Qr))
# template-pair mean persona offset vs per-pair offset
ps = pairs(W, S); keys = [(a, c) for _, a, c in ps]
M = {k: PD[[i for i, kk in enumerate(keys) if kk == k]].mean(0) for k in set(keys)}
Mt = np.stack([M[k] for k in keys])
print('template-pair-mean share of sum|PD|^2: %.3f' % (1 - np.sum((PD - Mt) ** 2) / np.sum(PD ** 2)))
# off-manifold: residual outside top-500 PCs fit on half the behaviours
half = set(behs[::2]); Hfit = np.stack([v for (b, t), v in H.items() if b in half]); mu = Hfit.mean(0)
V = np.linalg.svd(Hfit - mu, full_matrices=False)[2][:500].T
ev = [(b, a, c) for b, a, c in ps if b not in half]
res = lambda X: np.median(np.sum(((X - mu) - (X - mu) @ V @ V.T) ** 2, 1))
H0 = np.stack([H[(b, a)] for b, a, c in ev]); H1 = np.stack([H[(b, c)] for b, a, c in ev])
Hs = H0 + (H1 - H0) @ Q @ Q.T
print('off-manifold residual (top-500 PCs, held-out): h0 %.2f  h1 %.2f  h* %.2f' % (res(H0), res(H1), res(Hs)))
