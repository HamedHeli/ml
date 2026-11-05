"""Self-supervised representation learning with a contrastive loss, in JAX
(CPSC 440 lecture 23).

Pretext task: make two random augmentations of the same image ("positive pair")
map to nearby embeddings and different images map far apart. With a batch of n
images, two views each, normalized embeddings u_i, v_i and temperature tau,
the InfoNCE / SimCLR loss for view u_i is

    -log  exp(u_i . v_i / tau) / sum_j exp(u_i . v_j / tau)

i.e. a softmax classification of "which of the n candidates is my partner".
Minimizing it maximizes a lower bound on the mutual information between views
(log n minus the loss). What the representation learns is decided by the
augmentations: it becomes invariant to whatever they change.

Evaluation (linear probe): freeze the encoder, fit logistic regression on the
embeddings of a *few* labelled examples, and test. Compared with raw pixels,
PCA, and a random (untrained) encoder of the same shape.

Run:  python 19-representation-learning-and-llms/contrastive.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


def augment(key, X):
    """Random shifts of up to 1 pixel, random pixel dropout and Gaussian noise on 8x8 images."""
    k1, k2, k3 = jax.random.split(key, 3)
    imgs = X.reshape(-1, 8, 8)
    pad = jnp.pad(imgs, ((0, 0), (1, 1), (1, 1)))
    off = jax.random.randint(k1, (len(X), 2), 0, 3)
    crop = jax.vmap(lambda im, o: jax.lax.dynamic_slice(im, (o[0], o[1]), (8, 8)))(pad, off)
    keep = jax.random.uniform(k2, crop.shape) > 0.2
    return (crop * keep + 0.1 * jax.random.normal(k3, crop.shape)).reshape(len(X), 64)


def init(key, sizes=(64, 128, 64, 32)):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) * np.sqrt(2 / a), jnp.zeros(b)) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


def encode(params, x):
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def info_nce(params, x, key, tau=0.2):
    k1, k2 = jax.random.split(key)
    u = encode(params, augment(k1, x))
    v = encode(params, augment(k2, x))
    u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    S = u @ v.T / tau
    labels = jnp.arange(len(x))
    l1 = jax.nn.logsumexp(S, 1) - S[labels, labels]
    l2 = jax.nn.logsumexp(S, 0) - S[labels, labels]
    return jnp.mean(l1 + l2) / 2


def train(X, steps=3000, batch=256, seed=0):
    params = init(jax.random.PRNGKey(seed))
    opt = optax.adam(1e-3)
    state = opt.init(params)

    @jax.jit
    def step(p, st, xb, key):
        l, g = jax.value_and_grad(info_nce)(p, xb, key)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    losses = []
    for _ in range(steps):
        key, sub = jax.random.split(key)
        params, state, l = step(params, state, jnp.asarray(X[rng.integers(0, len(X), batch)]), sub)
        losses.append(float(l))
    return params, losses


def linear_probe(Ztr, ytr, Zte, yte, lam=1e-2, iters=500):
    """Multinomial logistic regression with Adam on standardized features."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    C = int(ytr.max()) + 1
    W = jnp.zeros((Ztr.shape[1], C))
    b = jnp.zeros(C)
    opt = optax.adam(5e-2)
    st = opt.init((W, b))

    def loss(p):
        S = Ztr @ p[0] + p[1]
        return jnp.mean(jax.nn.logsumexp(S, 1) - S[jnp.arange(len(ytr)), ytr]) + lam * jnp.sum(p[0] ** 2)

    g = jax.jit(jax.grad(loss))
    p = (W, b)
    for _ in range(iters):
        u, st = opt.update(g(p), st)
        p = optax.apply_updates(p, u)
    return float(np.mean(np.asarray(Zte @ p[0] + p[1]).argmax(1) == yte))


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(y))
    unl, te = perm[:1400], perm[1400:]               # pretraining uses the 1400 images without labels
    params, losses = train(X[unl])
    print(f"InfoNCE loss: {losses[0]:.2f} -> {np.mean(losses[-100:]):.2f} "
          f"(log batch size = {np.log(256):.2f}; the gap is a lower bound on mutual information between views, in nats)")
    rand = init(jax.random.PRNGKey(42))
    Xc = X[unl] - X[unl].mean(0)
    Wp = np.linalg.svd(Xc, full_matrices=False)[2][:32]
    feats = {"raw pixels (64-d)": lambda A: A,
             "PCA (32-d)": lambda A: (A - X[unl].mean(0)) @ Wp.T,
             "random encoder (32-d)": lambda A: np.asarray(encode(rand, jnp.asarray(A))),
             "contrastive encoder (32-d)": lambda A: np.asarray(encode(params, jnp.asarray(A)))}
    rng = np.random.default_rng(1)
    print("Linear-probe test accuracy with n labelled examples per class (mean of 5 draws):")
    print(f"{'features':>28} " + " ".join(f"{'n=' + str(n):>7}" for n in [1, 5, 20]))
    for name, f in feats.items():
        accs = []
        for n in [1, 5, 20]:
            a = []
            for _ in range(5):
                lab = np.concatenate([rng.choice(unl[y[unl] == c], n, replace=False) for c in range(10)])
                a.append(linear_probe(jnp.asarray(f(X[lab])), jnp.asarray(y[lab]), jnp.asarray(f(X[te])), y[te]))
            accs.append(np.mean(a))
        print(f"{name:>28} " + " ".join(f"{a:7.3f}" for a in accs))


if __name__ == "__main__":
    demo()
