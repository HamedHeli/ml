"""Attention and a small Transformer in JAX (CPSC 440 lecture 22, CPSC 340 L35).

Attention: a query q looks up a set of (key, value) pairs,
    attn(q, K, V) = softmax(q K^T / sqrt(d_k)) V.
Self-attention: every token produces its own query, key and value with learned
projections Q = X W_q, K = X W_k, V = X W_v, so every position can read from every
other position in one step; no information has to survive a long recurrence.

Transformer block (pre-norm, as in GPT-2 and later):
    X = X + MultiHeadAttention(LayerNorm(X))     (heads run in parallel, outputs concatenated)
    X = X + MLP(LayerNorm(X))                    (the per-token nonlinearity)
Attention is permutation-equivariant, so position must be added explicitly:
here sinusoidal encodings PE[t, 2i] = sin(t / 10000^{2i/d}), PE[t, 2i+1] = cos(...).
A causal mask (token t only attends to tokens <= t) makes it a decoder (GPT);
no mask gives an encoder (BERT).

Cost: O(T^2 d) per layer, versus O(T d^2) for an RNN, but fully parallel over T.

Experiment: the same recall task as rnn.py at T = 150, where the RNN and LSTM
were at chance. We then look at where the final token's attention goes.

Run:  python 17-sequence-models-and-transformers/transformer.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn import V, recall_batch  # noqa: E402


def sinusoidal(T, d):
    t = np.arange(T)[:, None]
    i = np.arange(d // 2)[None]
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(t / 10000 ** (2 * i / d))
    pe[:, 1::2] = np.cos(t / 10000 ** (2 * i / d))
    return jnp.asarray(pe)


def init(key, d=32, heads=2, layers=2, d_ff=64):
    ks = iter(jax.random.split(key, 4 + 6 * layers))
    n = lambda shape, s: jax.random.normal(next(ks), shape) * s
    p = {"emb": n((V + 1, d), 0.5), "out": n((d, V), 1 / np.sqrt(d)), "blocks": [], "heads": heads,
         "lnf": (jnp.ones(d), jnp.zeros(d))}
    for _ in range(layers):
        p["blocks"].append({"Wq": n((d, d), 1 / np.sqrt(d)), "Wk": n((d, d), 1 / np.sqrt(d)),
                            "Wv": n((d, d), 1 / np.sqrt(d)), "Wo": n((d, d), 1 / np.sqrt(d)),
                            "W1": n((d, d_ff), 1 / np.sqrt(d)), "W2": n((d_ff, d), 1 / np.sqrt(d_ff)),
                            "ln1": (jnp.ones(d), jnp.zeros(d)), "ln2": (jnp.ones(d), jnp.zeros(d))})
    return p


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + eps) + b


def mha(blk, x, heads, causal=True):
    n, T, d = x.shape
    dh = d // heads
    split = lambda a: a.reshape(n, T, heads, dh).transpose(0, 2, 1, 3)      # (n, h, T, dh)
    Q, K, Vv = split(x @ blk["Wq"]), split(x @ blk["Wk"]), split(x @ blk["Wv"])
    s = Q @ K.transpose(0, 1, 3, 2) / jnp.sqrt(dh)
    if causal:
        s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -1e9)
    A = jax.nn.softmax(s, -1)
    out = (A @ Vv).transpose(0, 2, 1, 3).reshape(n, T, d)
    return out @ blk["Wo"], A


def forward(p, x, return_attn=False):
    """x: (n, T) tokens, with the marker bit folded in as token id V at position 0's embedding."""
    n, T = x.shape
    h = p["emb"][x] + p["emb"][V] * (jnp.arange(T) == 0)[None, :, None] + sinusoidal(T, p["emb"].shape[1])
    attns = []
    for blk in p["blocks"]:
        a, A = mha(blk, layer_norm(h, *blk["ln1"]), p["heads"])
        h = h + a
        h = h + jax.nn.gelu(layer_norm(h, *blk["ln2"]) @ blk["W1"]) @ blk["W2"]
        attns.append(A)
    logits = layer_norm(h[:, -1], *p["lnf"]) @ p["out"]
    return (logits, attns) if return_attn else logits


def train_recall(T=150, steps=600, seed=0):
    rng = np.random.default_rng(seed)
    params = init(jax.random.PRNGKey(seed))
    heads = params.pop("heads")
    opt = optax.adam(3e-3)
    state = opt.init(params)

    def loss(p, x, y):
        s = forward({**p, "heads": heads}, x)
        return jnp.mean(jax.nn.logsumexp(s, 1) - s[jnp.arange(len(y)), y])

    @jax.jit
    def step(p, st, x, y):
        g = jax.grad(loss)(p, x, y)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    for _ in range(steps):
        x, y = recall_batch(rng, 64, T)
        params, state = step(params, state, jnp.asarray(x), jnp.asarray(y))
    params["heads"] = heads
    return params


def demo():
    T = 150
    p = train_recall(T)
    x, y = recall_batch(np.random.default_rng(999), 1000, T)
    logits, attns = forward(p, jnp.asarray(x), return_attn=True)
    acc = float(np.mean(np.asarray(logits).argmax(1) == y))
    print(f"Transformer (2 layers, 2 heads, d=32) on recall with T={T}: test accuracy {acc:.3f} (RNN/LSTM: chance)")
    for l, A in enumerate(attns):
        w0 = np.asarray(A[:, :, -1, 0]).mean(0)          # last query's weight on position 0, per head
        print(f"  layer {l + 1}: attention from the last token to position 0, per head: {np.round(w0, 3)} "
              f"(uniform would be {1 / T:.3f})")
    n_par = sum(a.size for a in jax.tree_util.tree_leaves({k: v for k, v in p.items() if k != "heads"}))
    print(f"  parameters: {n_par}")


if __name__ == "__main__":
    demo()
