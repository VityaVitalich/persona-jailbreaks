"""
gqa_fold_attention.py -- a grouped-query attention implementation for HF transformers
that never materialises the expanded key/value tensors.

Why: on the B300 (sm_103) with torch 2.8 only the SDPA *math* backend accepts Gemma-4's
shapes (head_dim 512 on global layers, boolean padding masks), and the math backend
implements `enable_gqa` by repeat_interleave-ing K and V -- ~1-2.5 GB of writes per layer
per decode step, i.e. 6-9 ms per attention call (measured), ~60 ms per decode step and
~840 prompt tokens/s at prefill for the 31B model. Folding the `g = H / KVH` query heads
that share a KV head into the query-length axis turns attention into two plain batched
matmuls of the unexpanded K/V: scores = Q'[B,KVH,g*Lq,D] @ K^T[B,KVH,D,Lk].

Numerics match the eager/math path: scores in bf16 matmul, additive finfo.min mask,
softmax in fp32, probs cast back to the input dtype before the PV matmul.
The query axis is chunked (QCHUNK) so the fp32 score tensor stays bounded at prefill.

Usage:
    from gqa_fold_attention import register; register()          # once, before from_pretrained
    AutoModelForCausalLM.from_pretrained(..., attn_implementation="gqa_fold")
"""
import os
import torch

QCHUNK = int(os.environ.get("GQA_FOLD_QCHUNK", "512"))    # folded query rows (g*Lq) per chunk
_MASK_CACHE = {}   # id(mask) -> (mask ref, additive mask). HF builds 2 boolean masks per forward (full,
                   # sliding) and hands one to every layer -> convert once per forward, not 60 times.
                   # Keyed by object identity while holding a reference, so the id cannot be recycled
                   # by a later mask while the entry lives (a data_ptr key could go stale that way).


def _additive_mask(m, dtype):
    if m.dtype != torch.bool:
        return m.to(dtype)
    hit = _MASK_CACHE.get(id(m))
    if hit is not None and hit[0] is m and hit[1].dtype == dtype:
        return hit[1]
    if len(_MASK_CACHE) >= 4:
        _MASK_CACHE.clear()
    add = torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill_(~m, torch.finfo(dtype).min)
    _MASK_CACHE[id(m)] = (m, add)
    return add


def gqa_fold_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                               scaling=None, sliding_window=None, **kwargs):
    B, H, Lq, D = query.shape
    KVH, Lk = key.shape[1], key.shape[2]
    g = H // KVH
    if scaling is None:
        scaling = D ** -0.5
    # head h = j*g + r shares kv head j (repeat_kv semantics) -> group rows (r, t) under kv head j
    q = query.reshape(B, KVH, g * Lq, D)
    kT = key.transpose(-1, -2)                                   # [B,KVH,D,Lk]
    mask = None
    if attention_mask is not None:
        mask = _additive_mask(attention_mask, query.dtype)
        # mask: [B,1,Lq|1,Lk] -> expand to folded rows [B,1,g,Lq|1,Lk] (view, no copy)
        mask = mask[:, :, None]                                  # [B,1,1,Lq|1,Lk]
    rows = g * Lq
    out = torch.empty(B, KVH, rows, D, dtype=query.dtype, device=query.device)
    # chunk along the ORIGINAL query axis (t) so the mask slices stay aligned
    tq = max(1, QCHUNK // g) if rows > QCHUNK else Lq
    for t0 in range(0, Lq, tq):
        t1 = min(Lq, t0 + tq)
        qc = query[:, :, t0:t1].reshape(B, KVH, g * (t1 - t0), D)           # [B,KVH,g*tc,D]
        s = torch.matmul(qc, kT)                                             # [B,KVH,g*tc,Lk]
        if scaling != 1.0:
            s.mul_(scaling)
        if mask is not None:
            mc = mask if mask.shape[-2] == 1 else mask[..., t0:t1, :]        # [B,1,1,tc|1,Lk]
            s.view(B, KVH, g, t1 - t0, Lk).add_(mc)                          # in place, broadcast
        p = torch.softmax(s, dim=-1, dtype=torch.float32).to(query.dtype)    # fp32 softmax, bf16 probs
        del s
        oc = torch.matmul(p, value)                                          # [B,KVH,g*tc,D]
        del p
        out[:, :, :].view(B, KVH, g, Lq, D)[:, :, :, t0:t1] = oc.view(B, KVH, g, t1 - t0, D)
    attn = out.view(B, H, Lq, D).transpose(1, 2).contiguous()               # [B,Lq,H,D]
    return attn, None


def register(name="gqa_fold"):
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register(name, gqa_fold_attention_forward)

    def _mask(*args, **kw):                 # always materialise the boolean mask (no is_causal skip)
        kw["allow_is_causal_skip"] = False
        return sdpa_mask(*args, **kw)
    AttentionMaskInterface.register(name, _mask)
    return name
