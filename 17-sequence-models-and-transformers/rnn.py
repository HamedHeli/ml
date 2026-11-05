"""Recurrent neural networks and LSTMs in JAX (CPSC 440 lecture 21).

An RNN is a discriminative cousin of the HMM: a deterministic hidden state
    h_t = tanh(W_h h_{t-1} + W_x x_t + b),   yhat_t = softmax(V h_t + c)
updated left to right. Inference is a single forward pass; training is
backpropagation through time. Repeated multiplication by W_h makes gradients
vanish or explode exponentially in the time lag, so plain RNNs struggle to carry
information over many steps.

LSTM: add a memory cell c_t with soft gates (input i, forget f, output o):
    c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t])
    h_t = o_t * tanh(c_t)
With f_t near 1 the cell is copied forward unchanged, so gradients survive long
lags. Gradient clipping handles the exploding case.

Experiment: the *recall* task. Input: T random tokens from {0..7}, plus a marker
bit that is 1 on the first token; the label is that first token. A model must
store one symbol and then protect it from T - 1 distracting inputs.

Run:  python 17-sequence-models-and-transformers/rnn.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

V = 8          # vocabulary size


def recall_batch(rng, n, T):
    x = rng.integers(0, V, (n, T))
    return x, x[:, 0]


def init(key, kind, H=64):
    ks = jax.random.split(key, 4)
    g = 4 if kind == "lstm" else 1
    p = {"Wx": jax.random.normal(ks[0], (V + 1, g * H)) * 0.3,
         "Wh": jax.random.normal(ks[1], (H, g * H)) / jnp.sqrt(H),
         "b": jnp.zeros(g * H),
         "Vo": jax.random.normal(ks[2], (H, V)) / jnp.sqrt(H), "c": jnp.zeros(V)}
    if kind == "lstm":
        p["b"] = p["b"].at[H:2 * H].set(3.0)      # forget-gate bias 3: start by remembering
    return p


def run(params, x, kind):
    H = params["Wh"].shape[0]
    X = jax.nn.one_hot(x, V)                      # (n, T, V)
    marker = jnp.zeros(x.shape + (1,)).at[:, 0].set(1.0)
    X = jnp.concatenate([X, marker], -1)          # extra input bit marks the token to remember

    def rnn_step(h, xt):
        h = jnp.tanh(xt @ params["Wx"] + h @ params["Wh"] + params["b"])
        return h, None

    def lstm_step(carry, xt):
        h, c = carry
        a = xt @ params["Wx"] + h @ params["Wh"] + params["b"]
        i, f, o, g = (jax.nn.sigmoid(a[:, :H]), jax.nn.sigmoid(a[:, H:2 * H]),
                      jax.nn.sigmoid(a[:, 2 * H:3 * H]), jnp.tanh(a[:, 3 * H:]))
        c = f * c + i * g
        return (o * jnp.tanh(c), c), None

    h0 = jnp.zeros((x.shape[0], H))
    Xs = jnp.swapaxes(X, 0, 1)
    if kind == "lstm":
        (h, _), _ = jax.lax.scan(lstm_step, (h0, h0), Xs)
    else:
        h, _ = jax.lax.scan(rnn_step, h0, Xs)
    return h @ params["Vo"] + params["c"]


def train_recall(kind, T, steps=1500, seed=0):
    rng = np.random.default_rng(seed)
    params = init(jax.random.PRNGKey(seed), kind)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-3))
    state = opt.init(params)

    def loss(p, x, y):
        s = run(p, x, kind)
        return jnp.mean(jax.nn.logsumexp(s, 1) - s[jnp.arange(len(y)), y])

    @jax.jit
    def step(p, st, x, y):
        l, g = jax.value_and_grad(loss)(p, x, y)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    for _ in range(steps):
        x, y = recall_batch(rng, 64, T)
        params, state, _ = step(params, state, jnp.asarray(x), jnp.asarray(y))
    x, y = recall_batch(np.random.default_rng(999), 2000, T)
    return float(np.mean(np.asarray(run(params, jnp.asarray(x), kind)).argmax(1) == y))


def demo():
    print("Recall the first of T tokens (chance = 1/8 = 0.125), test accuracy:")
    print(f"{'T':>4} {'RNN':>6} {'LSTM':>6}")
    for T in [20, 80, 150]:
        print(f"{T:4d} {train_recall('rnn', T):6.3f} {train_recall('lstm', T):6.3f}")


if __name__ == "__main__":
    demo()
