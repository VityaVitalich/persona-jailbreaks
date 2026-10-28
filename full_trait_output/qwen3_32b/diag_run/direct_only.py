exec(open("analyse_diag.py").read().split("old = torch.load")[0])
import re
REF = re.compile(r"(i'?m sorry|i can(no|')t|i cannot|i won'?t|i'?m unable|i must decline|i apologi)", re.I)
dual = np.array([bool(y[i]) and bool(REF.search(r["response_text"][:300])) for i, r in enumerate(hv)])
keep = ~dual; print("dropped refuse-then-comply jailbreaks:", dual.sum(), "| remaining jb", y[keep].sum(), "of", keep.sum())
for s in ("think_L32_pos-1", "think_L48_pos-1", "think_L63_pos-1"):
    k = D["slots"].index(s); a = []
    for tr, te in S:
        tr, te = tr[keep[tr]], te[keep[te]]
        if 0 < y[te].sum() < len(te): a.append(fold(y[te], make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(X[tr, k], y[tr]).decision_function(X[te, k])))
    print(f"{s}: template-disjoint test AUC without refuse-then-comply jailbreaks {np.mean(a):.3f} (min {np.min(a):.2f}, {len(a)} splits)")
