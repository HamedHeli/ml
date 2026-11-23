"""Small numerical checks for the mechanisms in a modern LLM block (see llm-block-mechanisms.md).

Each function isolates one mechanism and measures what happens with and without it, at initialization or on a
toy problem. Nothing here trains a language model; the point is to see the *mechanism* each choice relies on:

    signal_through_depth    residual stream: how much input signal and gradient survive 48 layers, with and without x + F(x)
    train_stack             plain vs post-norm vs pre-norm stacks trained with Adam at two learning rates, no warm-up
    last_layer_grad_vs_depth  why post-norm needs warm-up: its top-layer gradient does not shrink with depth
    depth_scaled_init       scaling the residual-writing projections by 1/sqrt(2L) keeps the stream's size bounded
    rmsnorm_properties      RMSNorm vs LayerNorm: what each removes, scale invariance, cost
    attention_scaling       why scores are divided by sqrt(d_head): softmax saturation without it
    qk_norm                 QK-norm keeps attention from collapsing when the projection weights grow
    rope_relative           RoPE makes q.k depend only on the offset m - n; an added absolute embedding does not
    swiglu_param_match      SwiGLU at hidden 8/3 d has the same parameter count as a 4d GELU MLP
    dropout_regimes         dropout helps when the same examples are seen many times, not in one pass over fresh data
    attention_biases        a key bias has no effect at all; a value bias is a constant shift; biases are ~0.01% of params
    gqa_equivalence         GQA with g = h is MHA, g = 1 is MQA; KV-cache bytes for a Llama-3-8B-shaped model

Run:  python 17-sequence-models-and-transformers/llm_block_mechanisms.py      (about 2 min on a CPU, mostly compiling the 24-layer stacks)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)


# ----------------------------------------------------------------------------------------------- building pieces

def rmsnorm(x, g=None, eps=1e-6):
    y = x / jnp.sqrt(jnp.mean(x * x, -1, keepdims=True) + eps)
    return y if g is None else y * g


def layernorm(x, eps=1e-6):
    m = jnp.mean(x, -1, keepdims=True)
    v = jnp.mean((x - m) ** 2, -1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


def mlp_params(key, L, d, h, out_scale=1.0):
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (L, d, h)) / np.sqrt(d)
    W2 = jax.random.normal(k2, (L, h, d)) / np.sqrt(h) * out_scale
    return W1, W2


def stack(params, x, mode):
    """A depth-L stack of MLP sub-layers F(x) = W2 gelu(W1 x), wired in one of four ways."""
    W1, W2 = params
    F = lambda l, z: jax.nn.gelu(z @ W1[l]) @ W2[l]
    for l in range(W1.shape[0]):
        if mode == "plain":
            x = F(l, x)
        elif mode == "residual":
            x = x + F(l, x)
        elif mode == "post":
            x = layernorm(x + F(l, x))
        elif mode == "pre":
            x = x + F(l, rmsnorm(x))
    return rmsnorm(x) if mode == "pre" else x


def per_layer_grad_norms(mode, L=24, d=64, n=256, seed=0, out_scale=1.0):
    key = jax.random.PRNGKey(seed)
    params = mlp_params(key, L, d, 4 * d, out_scale)
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (n, d))
    r = jax.random.normal(jax.random.PRNGKey(seed + 2), (n, d))
    loss = lambda p: jnp.mean(jnp.sum(stack(p, x, mode) * r, -1))
    g1, g2 = jax.grad(loss)(params)
    return np.asarray(jnp.sqrt(jnp.sum(g1 ** 2, (1, 2)) + jnp.sum(g2 ** 2, (1, 2))))


# ------------------------------------------------------------------------------------------------- the checks

def signal_through_depth(L=48, d=64):
    """How much of the input and of the gradient survive L layers, with and without the residual add.

    Returns |dLoss/dx_0| / |dLoss/dx_L| (the gradient that reaches the bottom) and RMS(x_L) / RMS(x_0).
    The residual branches use a 1/sqrt(L) output scale (see depth_scaled_init) so the sum stays bounded.
    """
    x0 = jax.random.normal(jax.random.PRNGKey(1), (256, d))
    r = jax.random.normal(jax.random.PRNGKey(2), (256, d))
    out = {}
    for mode in ("plain", "residual"):
        p = mlp_params(jax.random.PRNGKey(0), L, d, 4 * d, 1.0 if mode == "plain" else 1 / np.sqrt(L))
        g = jax.grad(lambda x: jnp.sum(stack(p, x, mode) * r))(x0)
        xL = stack(p, x0, mode)
        out[mode] = {"grad_ratio": float(jnp.linalg.norm(g) / jnp.linalg.norm(r)),
                     "signal_ratio": float(jnp.sqrt(jnp.mean(xL ** 2) / jnp.mean(x0 ** 2)))}
    return out


def train_stack(mode, lr, L=24, d=64, steps=400, seed=0):
    """Adam, constant learning rate, no warm-up, on y = sin(x W) with a deep stack wired as `mode`.

    Returns (initial loss, mean loss over the last 20 steps). The target's variance is about 0.43.
    """
    import optax
    X = jax.random.normal(jax.random.PRNGKey(seed + 5), (2048, d))
    Y = jnp.sin(X @ (jax.random.normal(jax.random.PRNGKey(seed + 6), (d, d)) / np.sqrt(d)))
    out_scale = 1 / np.sqrt(L) if mode == "residual" else 1.0
    params = (mlp_params(jax.random.PRNGKey(seed), L, d, 4 * d, out_scale), jnp.eye(d))
    opt = optax.adam(lr)
    st = opt.init(params)
    loss = lambda P, x, y: jnp.mean((stack(P[0], x, mode) @ P[1] - y) ** 2)

    @jax.jit
    def step(P, st, i):
        x = jax.lax.dynamic_slice_in_dim(X, i * 64 % 2048, 64)
        y = jax.lax.dynamic_slice_in_dim(Y, i * 64 % 2048, 64)
        l, g = jax.value_and_grad(loss)(P, x, y)
        u, st = opt.update(g, st, P)
        return optax.apply_updates(P, u), st, l

    ls = []
    for i in range(steps):
        params, st, l = step(params, st, i)
        ls.append(float(l))
    return ls[0], float(np.mean(ls[-20:]))


def last_layer_grad_vs_depth(depths=(6, 24, 96)):
    """Gradient norm on the last layer's weights at init: flat in depth for post-norm, ~1/sqrt(L) for pre-norm."""
    return {mode: {L: float(per_layer_grad_norms(mode, L=L)[-1]) for L in depths} for mode in ("post", "pre")}


def depth_scaled_init(L=48, d=64):
    """RMS of the pre-norm residual stream after L blocks, with the output projection at 1 or 1/sqrt(2L)."""
    x0 = jax.random.normal(jax.random.PRNGKey(1), (256, d))
    out = {}
    for name, s in (("standard", 1.0), ("scaled 1/sqrt(2L)", 1 / np.sqrt(2 * L))):
        W1, W2 = mlp_params(jax.random.PRNGKey(0), L, d, 4 * d, s)
        x = x0
        for l in range(L):
            x = x + jax.nn.gelu(rmsnorm(x) @ W1[l]) @ W2[l]
        out[name] = float(jnp.sqrt(jnp.mean(x * x)) / jnp.sqrt(jnp.mean(x0 * x0)))
    return out


def rmsnorm_properties(d=4096, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=d) + 0.05)                # a token vector with a small nonzero mean
    xc = x - x.mean()
    g = jax.grad(lambda z: jnp.sum(rmsnorm(z) * jnp.arange(d)))(x)
    return {
        "scale_invariant": float(jnp.max(jnp.abs(rmsnorm(7.3 * x) - rmsnorm(x)))),
        "shift_changes_rms": float(jnp.max(jnp.abs(rmsnorm(x + 3.0) - rmsnorm(x)))),
        "shift_changes_ln": float(jnp.max(jnp.abs(layernorm(x + 3.0) - layernorm(x)))),
        "ln_equals_rms_of_centred": float(jnp.max(jnp.abs(layernorm(x) - rmsnorm(xc)))),
        "rms_vs_ln_at_small_mean": float(jnp.max(jnp.abs(layernorm(x) - rmsnorm(x)))),
        "grad_dot_x": float(jnp.dot(g, x) / (jnp.linalg.norm(g) * jnp.linalg.norm(x))),
        # passes over the vector (reductions) and learned vectors
        "reductions": {"LayerNorm": 2, "RMSNorm": 1}, "learned_vectors": {"LayerNorm": 2, "RMSNorm": 1},
    }


def attention_scaling(T=512, dims=(16, 64, 128, 256), seed=0):
    """Score spread and attention entropy over T keys for random unit-variance q, k, with and without 1/sqrt(d)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dims:
        q, K = rng.normal(size=(200, d)), rng.normal(size=(T, d))
        s = q @ K.T
        def stats(z):
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            return float(np.mean(-np.sum(p * np.log(p + 1e-30), 1))), float(np.mean(p.max(1)))
        h_raw, m_raw = stats(s)
        h_sc, m_sc = stats(s / np.sqrt(d))
        rows.append({"d": d, "score_std": float(s.std()), "entropy_raw": h_raw, "maxw_raw": m_raw,
                     "entropy_scaled": h_sc, "maxw_scaled": m_sc})
    return {"rows": rows, "uniform_entropy": float(np.log(T))}


def qk_norm(T=256, d=64, dh=64, scales=(1, 4, 16), seed=0):
    """Attention entropy as W_q, W_k grow by a factor s (as weights do over training), with and without QK-norm."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    Wq, Wk = rng.normal(size=(d, dh)) / np.sqrt(d), rng.normal(size=(d, dh)) / np.sqrt(d)
    def entropy(s):
        s = s - s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        return float(np.mean(-np.sum(p * np.log(p + 1e-30), 1)))
    rms = lambda z: z / np.sqrt(np.mean(z * z, -1, keepdims=True))
    rows = []
    for s in scales:
        Q, K = X @ (s * Wq), X @ (s * Wk)
        rows.append({"scale": s, "max_logit_raw": float(np.abs(Q @ K.T).max() / np.sqrt(dh)),
                     "entropy_raw": entropy(Q @ K.T / np.sqrt(dh)),
                     "max_logit_qknorm": float(np.abs(rms(Q) @ rms(K).T).max() / np.sqrt(dh)),
                     "entropy_qknorm": entropy(rms(Q) @ rms(K).T / np.sqrt(dh))})
    return {"rows": rows, "uniform_entropy": float(np.log(T)), "qknorm_bound": float(np.sqrt(dh))}


def rope(x, pos, base=10000.0):
    """Rotate consecutive pairs (x[2i], x[2i+1]) of the last axis by angle pos * base^(-2i/d)."""
    d = x.shape[-1]
    theta = pos[:, None] * base ** (-np.arange(0, d, 2) / d)          # (T, d/2)
    c, s = np.cos(theta), np.sin(theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def rope_relative(d=64, T=64, shift=1000, seed=0):
    """Does score(m, n) depend only on m - n? Compare RoPE with an added sinusoidal absolute embedding."""
    rng = np.random.default_rng(seed)
    q, k = rng.normal(size=d), rng.normal(size=d)
    pos = np.arange(T, dtype=float)
    def scores(p):
        return rope(np.tile(q, (T, 1)), p) @ rope(np.tile(k, (T, 1)), p).T
    def sinus(p):
        i = np.arange(d // 2)
        pe = np.zeros((len(p), d))
        pe[:, 0::2] = np.sin(p[:, None] / 10000 ** (2 * i / d))
        pe[:, 1::2] = np.cos(p[:, None] / 10000 ** (2 * i / d))
        return pe
    abs_scores = lambda p: (q + sinus(p)) @ (k + sinus(p)).T
    return {
        "rope_shift_error": float(np.abs(scores(pos) - scores(pos + shift)).max()),
        "absolute_shift_error": float(np.abs(abs_scores(pos) - abs_scores(pos + shift)).max()),
        "rope_norm_change": float(np.abs(np.linalg.norm(rope(np.tile(q, (T, 1)), pos), axis=1)
                                         - np.linalg.norm(q)).max()),
        "score_scale": float(np.abs(scores(pos)).mean()),
    }


def swiglu_param_match(d=4096):
    gelu = 2 * d * (4 * d)
    h = int(round(8 * d / 3))
    swiglu = 3 * d * h
    return {"gelu_4d": gelu, "swiglu_hidden": h, "swiglu_8/3d": swiglu, "ratio": swiglu / gelu,
            "llama3_8b_hidden": 14336, "llama3_8b_hidden_over_d": 14336 / 4096}


def dropout_regimes(d=100, p=0.2, seed=0):
    """Linear regression with input dropout (inverted, rescaled by 1/(1-p)).

    small data: n = 80 < d examples, seen 2000 times (full-batch GD). Dropout acts like a ridge penalty.
    one pass:   20 000 fresh examples, each seen once (SGD, batch 32). There is nothing to overfit.
    Returns test MSE (noise variance = 0.25 is the floor).
    """
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=d) / np.sqrt(d)
    def data(n):
        X = rng.normal(size=(n, d))
        return X, X @ w_true + 0.5 * rng.normal(size=n)
    Xte, yte = data(5000)
    mse = lambda w: float(np.mean((Xte @ w - yte) ** 2))

    def train(X, y, drop, steps, batch, lr):
        w = np.zeros(d)
        r = np.random.default_rng(seed + 1)
        for t in range(steps):
            idx = np.arange(len(y)) if batch is None else np.arange(t * batch, (t + 1) * batch)
            Xb, yb = X[idx], y[idx]
            if drop > 0:
                Xb = Xb * (r.random(Xb.shape) > drop) / (1 - drop)
            w -= lr * Xb.T @ (Xb @ w - yb) / len(yb)
        return w

    Xs, ys = data(80)
    Xl, yl = data(20000)
    return {
        "small_no_dropout": mse(train(Xs, ys, 0.0, 2000, None, 0.05)),
        "small_dropout": mse(train(Xs, ys, p, 2000, None, 0.05)),
        "stream_no_dropout": mse(train(Xl, yl, 0.0, 20000 // 32, 32, 0.01)),
        "stream_dropout": mse(train(Xl, yl, p, 20000 // 32, 32, 0.01)),
        "noise_floor": 0.25,
    }


def attention_biases(T=16, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    bk, bv = rng.normal(size=d), rng.normal(size=d)
    mask = np.tril(np.ones((T, T), bool))
    def attn(bk_, bv_):
        s = (X @ Wq) @ (X @ Wk + bk_).T / np.sqrt(d)
        s = np.where(mask, s, -np.inf)
        a = np.exp(s - s.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        return a @ (X @ Wv + bv_)
    base = attn(0, 0)
    # biases vs weights in a Llama-3-8B-shaped model if every linear layer had a bias
    dm, ff, L, hq, hkv, dh, V = 4096, 14336, 32, 32, 8, 128, 128256
    weights = L * (dm * hq * dh + 2 * dm * hkv * dh + hq * dh * dm + 3 * dm * ff) + 2 * V * dm
    biases = L * (hq * dh + 2 * hkv * dh + dm + 2 * ff + dm)
    return {"key_bias_effect": float(np.abs(attn(bk, 0) - base).max()),
            "value_bias_minus_constant": float(np.abs(attn(0, bv) - base - bv).max()),
            "bias_fraction": biases / weights}


def gqa(X, Wq, Wk, Wv, h, g):
    """Causal attention with h query heads sharing g key/value heads (h % g == 0)."""
    T, d = X.shape
    dh = Wq.shape[1] // h
    Q = (X @ Wq).reshape(T, h, dh).transpose(1, 0, 2)
    K = np.repeat((X @ Wk).reshape(T, g, dh).transpose(1, 0, 2), h // g, 0)
    V = np.repeat((X @ Wv).reshape(T, g, dh).transpose(1, 0, 2), h // g, 0)
    s = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    return (a @ V).transpose(1, 0, 2).reshape(T, h * dh)


def gqa_equivalence(T=10, d=32, h=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    dh = d // h
    # MHA reference: each head has its own K, V
    ref = np.concatenate([gqa(X, Wq[:, i * dh:(i + 1) * dh], Wk[:, i * dh:(i + 1) * dh],
                              Wv[:, i * dh:(i + 1) * dh], 1, 1) for i in range(h)], 1)
    kv = lambda L, g, dh, ctx, bytes_=2: L * g * dh * 2 * bytes_ * ctx
    cfg = dict(L=32, dh=128)                                     # Llama 3 8B: 32 layers, 32 query heads, head dim 128
    return {"g=h_equals_mha": float(np.abs(gqa(X, Wq, Wk, Wv, h, h) - ref).max()),
            "kv_bytes_per_token": {"MHA (32 KV heads)": kv(g=32, ctx=1, **cfg),
                                   "GQA (8 KV heads)": kv(g=8, ctx=1, **cfg),
                                   "MQA (1 KV head)": kv(g=1, ctx=1, **cfg)},
            "kv_GiB_at_128k": {"MHA": kv(g=32, ctx=131072, **cfg) / 2 ** 30,
                               "GQA": kv(g=8, ctx=131072, **cfg) / 2 ** 30,
                               "MQA": kv(g=1, ctx=131072, **cfg) / 2 ** 30}}


# ------------------------------------------------------------------------------------------------------ report

def main():
    r = signal_through_depth()
    print("1. Residual stream (48 MLP layers): |grad at input| / |grad at output| and RMS(output) / RMS(input)")
    for k, v in r.items():
        print(f"   {k:<9} gradient {v['grad_ratio']:.1e}   signal {v['signal_ratio']:.1e}")

    print("2. Training a 24-layer stack, Adam, no warm-up: loss at step 0 -> mean of the last 20 of 400 steps")
    for mode, name in (("plain", "plain x = F(x)"), ("residual", "residual, no norm"),
                       ("post", "post-norm Norm(x + F(x))"), ("pre", "pre-norm x + F(Norm(x))")):
        res = [train_stack(mode, lr) for lr in (1e-3, 1e-2)]
        print(f"   {name:<26} lr 1e-3: {res[0][0]:.2f} -> {res[0][1]:.3g}    lr 1e-2: {res[1][0]:.2f} -> {res[1][1]:.3g}")
    lg = last_layer_grad_vs_depth()
    print("   gradient on the top layer at init, depth 6 / 24 / 96:  "
          + "   ".join(f"{m}-norm " + " / ".join(f"{v:.2f}" for v in lg[m].values()) for m in lg))

    di = depth_scaled_init()
    print("3. Depth-scaled init (48 pre-norm layers, stream RMS out / in)")
    print("   " + "   ".join(f"{k}: {v:.2f}" for k, v in di.items()))

    rp = rmsnorm_properties()
    print("4. RMSNorm vs LayerNorm (d = 4096)")
    print(f"   |RMSNorm(7.3x) - RMSNorm(x)| = {rp['scale_invariant']:.1e} (scale invariant)")
    print(f"   adding 3 to every feature changes RMSNorm by up to {rp['shift_changes_rms']:.2f}, "
          f"LayerNorm by {rp['shift_changes_ln']:.1e}")
    print(f"   LayerNorm(x) = RMSNorm(x - mean): max diff {rp['ln_equals_rms_of_centred']:.1e}; "
          f"RMSNorm vs LayerNorm when the mean is small: {rp['rms_vs_ln_at_small_mean']:.3f}")
    print(f"   cos(grad, x) = {rp['grad_dot_x']:.1e} (the gradient never changes the vector's length)")

    a = attention_scaling()
    print(f"5. Scaled dot product (T = 512 keys, uniform entropy {a['uniform_entropy']:.2f})")
    for row in a["rows"]:
        print(f"   d={row['d']:>3}: score std {row['score_std']:5.1f} | no scaling: entropy {row['entropy_raw']:.2f}, "
              f"max weight {row['maxw_raw']:.2f} | /sqrt(d): entropy {row['entropy_scaled']:.2f}, "
              f"max weight {row['maxw_scaled']:.3f}")

    q = qk_norm()
    print(f"6. QK-norm (W_q, W_k multiplied by s; uniform entropy {q['uniform_entropy']:.2f})")
    for row in q["rows"]:
        print(f"   s={row['scale']:>2}: max logit {row['max_logit_raw']:7.1f}, entropy {row['entropy_raw']:.2f}"
              f"  | QK-norm: max logit {row['max_logit_qknorm']:.1f} (bound {q['qknorm_bound']:.0f}), "
              f"entropy {row['entropy_qknorm']:.2f}")

    ro = rope_relative()
    print("7. RoPE: shift every position by 1000 and compare the 64x64 score matrix")
    print(f"   RoPE max change {ro['rope_shift_error']:.1e}   added absolute embedding {ro['absolute_shift_error']:.2f}"
          f"   (typical |score| {ro['score_scale']:.1f});  RoPE changes |q| by {ro['rope_norm_change']:.1e}")

    sw = swiglu_param_match()
    print(f"8. SwiGLU hidden {sw['swiglu_hidden']} for d=4096: {sw['swiglu_8/3d']:,} params vs "
          f"{sw['gelu_4d']:,} for a 4d GELU MLP (ratio {sw['ratio']:.3f}); Llama 3 8B uses "
          f"{sw['llama3_8b_hidden']} = {sw['llama3_8b_hidden_over_d']:.1f} d")

    dr = dropout_regimes()
    print("9. Dropout p = 0.2, linear regression, test MSE (noise floor 0.25)")
    print(f"   80 examples seen 2000 times: no dropout {dr['small_no_dropout']:.3f}, dropout {dr['small_dropout']:.3f}")
    print(f"   20 000 examples seen once:   no dropout {dr['stream_no_dropout']:.3f}, dropout {dr['stream_dropout']:.3f}")

    b = attention_biases()
    print("10. Biases in attention")
    print(f"   adding a key bias changes the output by {b['key_bias_effect']:.1e}; a value bias b_v adds exactly b_v "
          f"(residual {b['value_bias_minus_constant']:.1e})")
    print(f"   biases in every linear layer of a Llama-3-8B-shaped model: {100 * b['bias_fraction']:.4f}% of weights")

    g = gqa_equivalence()
    print(f"11. GQA with g = h equals MHA (max diff {g['g=h_equals_mha']:.1e}); KV cache for Llama 3 8B shapes:")
    for k, v in g["kv_bytes_per_token"].items():
        print(f"   {k:<18} {v // 1024:>4} KiB per token   {g['kv_GiB_at_128k'][k.split()[0]]:5.1f} GiB at 128k context")


if __name__ == "__main__":
    main()
