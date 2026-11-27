"""Numerical checks for the transformer anatomy Q&A (see transformer-anatomy-qa.md).

NumPy only, under a second. Each function answers one question from the note with numbers:

    swiglu_vs_moe            a SwiGLU gate is soft and per hidden feature, and every weight is used; MoE top-k skips experts
    causal_vs_bidirectional  change a later token: a decoder's earlier outputs do not move, an encoder's do
    decode_with_kv_cache     prefill + one-token-at-a-time decoding with a K/V cache equals a full recompute
    gqa_grouping             32 query heads over 8 K/V heads: which query heads share which K/V head
    llama3_8b                Llama 3 8B shapes, parameter count by part, and KV-cache sizes with and without GQA

Run:  python 17-sequence-models-and-transformers/transformer_anatomy.py
"""

from __future__ import annotations

import numpy as np


def silu(z):
    return z / (1.0 + np.exp(-z))


def softmax(s, axis=-1):
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------------------------------------- Q1: gate vs MoE

def swiglu_vs_moe(d=64, h=256, T=200, E=64, k=2, seed=0):
    """SwiGLU: every token multiplies all three matrices; the gate scales features softly and differently per token.

    MoE: a router picks k of E experts per token, so only k/E of the expert weights are multiplied by that token.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    Wg, Wu = rng.normal(size=(d, h)) / np.sqrt(d), rng.normal(size=(d, h)) / np.sqrt(d)
    gate = silu(X @ Wg)                                     # (T, h): one gate value per token per hidden feature
    closed = np.abs(gate) < 0.05                           # features this token nearly mutes
    # how much do two tokens agree on which features are nearly closed?  (Jaccard of the closed sets)
    a, b = closed[0], closed[1]
    jaccard = (a & b).sum() / max((a | b).sum(), 1)
    z = np.linspace(-10, 10, 20001)
    router = rng.normal(size=(d, E)) / np.sqrt(d)
    chosen = np.argsort(-(X @ router), axis=1)[:, :k]       # top-k experts per token
    return {
        "swiglu_weight_fraction_used": 1.0,                 # W_gate, W_up, W_down all multiply every token
        "swiglu_matmuls_per_token": 3 * d * h,
        "plain_mlp_matmuls_per_token": 2 * d * h,
        "gate_min": float(silu(z).min()),                   # SiLU dips to about -0.278, it is not a 0..1 switch
        "gate_at_plus_4": float(silu(np.array(4.0))),      # and grows without bound for large inputs
        "frac_gates_nearly_closed": float(closed.mean()),
        "jaccard_closed_sets_two_tokens": float(jaccard),
        "moe_weight_fraction_used": k / E,
        "moe_distinct_experts_used_by_batch": int(len(np.unique(chosen))),
    }


# ------------------------------------------------------------------------------------- Q4: encoder vs decoder mask

def attention(X, Wq, Wk, Wv, causal):
    q, kk, v = X @ Wq, X @ Wk, X @ Wv
    s = q @ kk.T / np.sqrt(q.shape[1])
    if causal:
        T = len(X)
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    return softmax(s) @ v


def causal_vs_bidirectional(T=8, d=16, change_at=5, seed=0):
    """Replace token `change_at`; measure how much the outputs at positions before it move."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    W = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3)]
    X2 = X.copy()
    X2[change_at] = rng.normal(size=d)
    out = {}
    for name, causal in (("decoder (causal)", True), ("encoder (bidirectional)", False)):
        before = np.abs(attention(X, *W, causal) - attention(X2, *W, causal))[:change_at].max()
        out[name] = float(before)
    return out


# -------------------------------------------------------------------------------- Q6: KV cache and GQA decoding

def gqa_attend(q, K, V, h, g, mask=None):
    """q: (Tq, h, dh); K, V: (Tk, g, dh). Query head i uses K/V head i // (h // g)."""
    dh = q.shape[-1]
    kv_of = np.arange(h) // (h // g)
    s = np.einsum("qhd,khd->hqk", q, K[:, kv_of]) / np.sqrt(dh)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    return np.einsum("hqk,khd->qhd", softmax(s), V[:, kv_of])


def decode_with_kv_cache(prompt=6, new=10, d=32, h=8, g=2, seed=0):
    """Same weights for every token; each token gets its own q, k, v. Cached decoding must match full recompute."""
    rng = np.random.default_rng(seed)
    dh = d // h
    Wq = rng.normal(size=(d, h * dh)) / np.sqrt(d)
    Wk = rng.normal(size=(d, g * dh)) / np.sqrt(d)
    Wv = rng.normal(size=(d, g * dh)) / np.sqrt(d)
    X = rng.normal(size=(prompt + new, d))                  # stand-in for the token vectors entering this layer
    T = len(X)

    def qkv(x):
        return (x @ Wq).reshape(-1, h, dh), (x @ Wk).reshape(-1, g, dh), (x @ Wv).reshape(-1, g, dh)

    # full recompute: all T tokens at once with a causal mask
    q, K, V = qkv(X)
    full = gqa_attend(q, K, V, h, g, np.tril(np.ones((T, T), bool))[None])

    # cached: prefill the prompt, then one token per step; only the new token's q, k, v are computed
    q0, Kc, Vc = qkv(X[:prompt])
    outs = [gqa_attend(q0, Kc, Vc, h, g, np.tril(np.ones((prompt, prompt), bool))[None])]
    projections_cached = prompt
    for t in range(prompt, T):
        qt, kt, vt = qkv(X[t:t + 1])
        Kc, Vc = np.concatenate([Kc, kt]), np.concatenate([Vc, vt])
        outs.append(gqa_attend(qt, Kc, Vc, h, g))             # the newest token may see everything in the cache
        projections_cached += 1
    cached = np.concatenate(outs)
    # without a cache, step t re-projects all t+1 tokens
    projections_no_cache = prompt + sum(t + 1 for t in range(prompt, T))
    return {
        "max_abs_diff": float(np.abs(full - cached).max()),
        "token_projections_with_cache": projections_cached,
        "token_projections_without_cache": projections_no_cache,
        "cache_shape_K": tuple(Kc.shape),                   # (tokens, K/V heads, head dim): no Q cache
    }


def gqa_grouping(h=32, g=8):
    kv_of = np.arange(h) // (h // g)
    return {f"K/V head {j}": [int(i) for i in np.where(kv_of == j)[0]] for j in range(g)}


# ------------------------------------------------------------------------------------------ Q7, Q8: Llama 3 8B

LLAMA3_8B = dict(  # values from the model's published config.json (meta-llama/Meta-Llama-3-8B)
    d_model=4096, n_layers=32, n_q_heads=32, n_kv_heads=8, head_dim=128, d_ff=14336, vocab=128256,
    tied_embeddings=False, rope_theta=500000.0, context=8192,
)


def llama3_8b(c=LLAMA3_8B, bytes_per_number=2):
    d, L, hq, hkv, dh, ff, V = (c[k] for k in ("d_model", "n_layers", "n_q_heads", "n_kv_heads", "head_dim", "d_ff",
                                                "vocab"))
    shapes = {  # (input, output), in the x · W convention; PyTorch stores the transpose (out, in)
        "W_Q": (d, hq * dh), "W_K": (d, hkv * dh), "W_V": (d, hkv * dh), "W_O": (hq * dh, d),
        "W_gate": (d, ff), "W_up": (d, ff), "W_down": (ff, d),
        "embedding": (V, d), "output (unembedding)": (d, V),
    }
    n = {k: a * b for k, (a, b) in shapes.items()}
    attn = n["W_Q"] + n["W_K"] + n["W_V"] + n["W_O"]
    mlp = n["W_gate"] + n["W_up"] + n["W_down"]
    norms = 2 * d
    per_layer = attn + mlp + norms
    embed = n["embedding"] + (0 if c["tied_embeddings"] else n["output (unembedding)"])
    total = L * per_layer + embed + d                       # + final norm

    def kv_bytes(kv_heads, tokens):
        return L * kv_heads * dh * 2 * bytes_per_number * tokens

    return {
        "shapes": shapes,
        "q_width_equals_d_model": hq * dh == d,
        "params_attention_per_layer": attn,
        "params_mlp_per_layer": mlp,
        "params_per_layer": per_layer,
        "params_embeddings": embed,
        "params_total": total,
        "mlp_share_of_layer": mlp / per_layer,
        "kv_bytes_per_token": {"GQA (8 K/V heads)": kv_bytes(hkv, 1), "MHA (32 K/V heads)": kv_bytes(hq, 1)},
        "kv_bytes_at_8k": {"GQA (8 K/V heads)": kv_bytes(hkv, 8192), "MHA (32 K/V heads)": kv_bytes(hq, 8192)},
        "weights_bytes": total * bytes_per_number,
    }


def main():
    print("Q1  SwiGLU gate vs MoE routing")
    for k, v in swiglu_vs_moe().items():
        print(f"    {k:38s} {v:.4g}" if isinstance(v, float) else f"    {k:38s} {v}")

    print("\nQ4  change token 5; largest change in the outputs at positions 0-4")
    for k, v in causal_vs_bidirectional().items():
        print(f"    {k:26s} {v:.3g}")

    print("\nQ6  KV-cache decoding (8 query heads, 2 K/V heads) vs full recompute")
    for k, v in decode_with_kv_cache().items():
        print(f"    {k:34s} {v}")

    print("\nQ7  Llama 3 8B: which query heads share each K/V head")
    for k, v in gqa_grouping().items():
        print(f"    {k}: query heads {v}")

    r = llama3_8b()
    print("\nQ8  Llama 3 8B matrix shapes (input x output)")
    for k, (a, b) in r["shapes"].items():
        print(f"    {k:22s} {a:>7,} x {b:<7,}  = {a * b / 1e6:7.1f} M")
    print(f"    per layer: attention {r['params_attention_per_layer'] / 1e6:.1f} M, MLP {r['params_mlp_per_layer'] / 1e6:.1f} M"
          f" ({r['mlp_share_of_layer']:.0%} of the layer)")
    print(f"    embeddings (untied, in + out) {r['params_embeddings'] / 1e9:.3f} B")
    print(f"    total {r['params_total'] / 1e9:.3f} B parameters, {r['weights_bytes'] / 2**30:.1f} GiB in bf16")
    for k in r["kv_bytes_per_token"]:
        print(f"    KV cache {k}: {r['kv_bytes_per_token'][k] / 1024:.0f} KiB per token,"
              f" {r['kv_bytes_at_8k'][k] / 2**30:.0f} GiB at 8,192 tokens")


if __name__ == "__main__":
    main()
