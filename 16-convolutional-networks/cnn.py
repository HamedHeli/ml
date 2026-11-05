"""A small convolutional neural network in JAX (CPSC 340 L33; CPSC 440 lectures 5, 14).

Architecture for 8x8 digits:
    input 8x8x1
    conv 3x3, 16 filters, ReLU   -> 8x8x16    (padding "SAME")
    max-pool 2x2                 -> 4x4x16
    conv 3x3, 32 filters, ReLU   -> 4x4x32
    global average pool          -> 32
    dense                        -> 10 scores
Convolutions give local connections + weight sharing (few parameters,
translation equivariance); pooling gives some translation invariance and a
growing receptive field.

We compare with an MLP with a similar parameter count on the original test set
and on test images shifted by one pixel, and with data augmentation (random
shifts during training), which the CPSC 340 slides recommend for invariance.

Run:  python 16-convolutional-networks/cnn.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_cnn(key):
    k1, k2, k3 = jax.random.split(key, 3)
    he = lambda k, shape, fan_in: jax.random.normal(k, shape) * jnp.sqrt(2 / fan_in)
    return {"c1": (he(k1, (3, 3, 1, 16), 9), jnp.zeros(16)),
            "c2": (he(k2, (3, 3, 16, 32), 144), jnp.zeros(32)),
            "fc": (he(k3, (32, 10), 32), jnp.zeros(10))}


def conv(x, w, b):
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + b


def cnn(params, x):
    x = x.reshape(-1, 8, 8, 1)
    x = jax.nn.relu(conv(x, *params["c1"]))
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    x = jax.nn.relu(conv(x, *params["c2"]))
    x = x.mean((1, 2))
    W, b = params["fc"]
    return x @ W + b


def init_mlp(key, sizes=(64, 80, 10)):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) * jnp.sqrt(2 / a), jnp.zeros(b)) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


def mlp(params, x):
    x = x.reshape(len(x), -1)
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def n_params(p):
    return sum(a.size for a in jax.tree_util.tree_leaves(p))


def shift(X, dy, dx):
    imgs = X.reshape(-1, 8, 8)
    out = np.zeros_like(imgs)
    ys, yd = (slice(0, 8 - dy), slice(dy, 8)) if dy >= 0 else (slice(-dy, 8), slice(0, 8 + dy))
    xs, xd = (slice(0, 8 - dx), slice(dx, 8)) if dx >= 0 else (slice(-dx, 8), slice(0, 8 + dx))
    out[:, yd, xd] = imgs[:, ys, xs]
    return out.reshape(len(X), 64)


def train(model, params, X, y, steps=3000, batch=64, lr=3e-3, augment=False, seed=0):
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss(p, xb, yb):
        s = model(p, xb)
        return jnp.mean(jax.nn.logsumexp(s, 1) - s[jnp.arange(len(yb)), yb])

    @jax.jit
    def step(p, st, xb, yb):
        g = jax.grad(loss)(p, xb, yb)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(y), batch)
        xb = X[idx]
        if augment:
            xb = shift(xb, *rng.integers(-1, 2, 2))
        params, state = step(params, state, jnp.asarray(xb), jnp.asarray(y[idx]))
    return params


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    Xs = shift(X[te], 1, 1)                       # test images moved down-right by one pixel
    key = jax.random.PRNGKey(0)
    err = lambda m, p, A: float(np.mean(np.asarray(m(p, jnp.asarray(A)).argmax(1)) != y[te]))
    print(f"{'model':>28} {'params':>7} {'test err':>9} {'shifted test err':>17}")
    for name, model, init, aug in [("MLP 64-80-10", mlp, init_mlp, False), ("CNN", cnn, init_cnn, False),
                                   ("MLP + shift augmentation", mlp, init_mlp, True), ("CNN + shift augmentation", cnn, init_cnn, True)]:
        p = train(model, init(key), X[tr], y[tr], augment=aug)
        print(f"{name:>28} {n_params(p):7d} {err(model, p, X[te]):9.3f} {err(model, p, Xs):17.3f}")


if __name__ == "__main__":
    demo()
