"""Neural networks and backpropagation, by hand in NumPy (CPSC 340 L29-L31;
CPSC 440 lectures 4-5).

A network with L hidden layers computes
    z_1 = h(W_1 x + b_1),  z_l = h(W_l z_{l-1} + b_l),  scores = V z_L + c
and is trained with the softmax (cross-entropy) loss. Hidden layers *learn* the
feature transformation that a linear model on fixed features would need.

Backpropagation = reverse-mode automatic differentiation: one forward pass that
stores activations, one backward pass that applies the chain rule from the loss
to every weight. Cost is about the same as the forward pass.

    delta_scores = softmax(scores) - onehot(y)
    dV = delta^T z_L,  delta_{L} = (delta V) * h'(a_L),  dW_L = delta_L^T z_{L-1}, ...

Activations: sigmoid gradients vanish in deep nets (h' <= 1/4 per layer); ReLU
h(a) = max(0, a) keeps gradient 1 where active. Skip connections (ResNets) add
z_{l-1} back to z_l so gradients have a short path. Training uses mini-batch SGD
with momentum (heavy ball), L2 weight decay, and early stopping on a validation set.

The gradient is verified against finite differences before training.

Run:  python 15-neural-networks/mlp.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class MLP:
    def __init__(self, sizes, activation="relu", residual=False, seed=0):
        rng = np.random.default_rng(seed)
        self.act, self.residual = activation, residual
        # He initialization keeps activations' scale stable through ReLU layers
        self.params = [(rng.standard_normal((a, b)) * np.sqrt(2 / a), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]

    def _h(self, a):
        return np.maximum(a, 0) if self.act == "relu" else 1 / (1 + np.exp(-a))

    def _dh(self, a, z):
        return (a > 0).astype(float) if self.act == "relu" else z * (1 - z)

    def forward(self, X):
        cache, z = [], X
        for l, (W, b) in enumerate(self.params):
            a = z @ W + b
            if l == len(self.params) - 1:
                cache.append((z, a, None))
                return a, cache
            z_new = self._h(a)
            if self.residual and z_new.shape == z.shape:
                z_new = z_new + z
            cache.append((z, a, z_new))
            z = z_new

    def loss_grad(self, X, y, lam=0.0):
        S, cache = self.forward(X)
        lse = logsumexp(S, 1)
        f = np.sum(lse - S[np.arange(len(y)), y]) + lam / 2 * sum(np.sum(W ** 2) for W, _ in self.params)
        delta = np.exp(S - lse[:, None])
        delta[np.arange(len(y)), y] -= 1                    # d f / d scores
        grads = [None] * len(self.params)
        for l in range(len(self.params) - 1, -1, -1):
            z_in, a, z_out = cache[l]
            W, _ = self.params[l]
            if l < len(self.params) - 1:                    # through the activation
                h_out = z_out - z_in if (self.residual and z_out.shape == z_in.shape) else z_out
                d_skip = delta if (self.residual and z_out.shape == z_in.shape) else 0
                delta_a = delta * self._dh(a, h_out)
            else:
                delta_a, d_skip = delta, 0
            grads[l] = (z_in.T @ delta_a + lam * W, delta_a.sum(0))
            delta = delta_a @ W.T + d_skip
        return f, grads

    def predict(self, X):
        return self.forward(X)[0].argmax(1)


def grad_check(model, X, y, eps=1e-6, n_checks=20, seed=0):
    rng = np.random.default_rng(seed)
    _, grads = model.loss_grad(X, y)
    worst = 0.0
    for _ in range(n_checks):
        l = rng.integers(len(model.params))
        W, _ = model.params[l]
        i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
        old = W[i, j]
        W[i, j] = old + eps
        fp, _ = model.loss_grad(X, y)
        W[i, j] = old - eps
        fm, _ = model.loss_grad(X, y)
        W[i, j] = old
        num = (fp - fm) / (2 * eps)
        worst = max(worst, abs(num - grads[l][0][i, j]) / max(1e-8, abs(num) + abs(grads[l][0][i, j])))
    return worst


def train(model, X, y, Xv, yv, epochs=60, batch=64, lr=0.05, momentum=0.9, lam=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    vel = [(np.zeros_like(W), np.zeros_like(b)) for W, b in model.params]
    best, best_params, history = np.inf, None, []
    for ep in range(epochs):
        perm = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            idx = perm[s:s + batch]
            _, grads = model.loss_grad(X[idx], y[idx], lam * len(idx) / len(y))
            for l, ((W, b), (gW, gb)) in enumerate(zip(model.params, grads)):
                vW, vb = vel[l]
                vW[:] = momentum * vW - lr * gW / len(idx)
                vb[:] = momentum * vb - lr * gb / len(idx)
                W += vW
                b += vb
        err_v = np.mean(model.predict(Xv) != yv)
        history.append(err_v)
        if err_v < best:                                   # early stopping: keep the best weights
            best, best_params = err_v, [(W.copy(), b.copy()) for W, b in model.params]
    model.params = best_params
    return history


def first_layer_grad_norm(model, X, y):
    _, grads = model.loss_grad(X, y)
    return np.linalg.norm(grads[0][0]) / len(y)


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    perm = np.random.default_rng(0).permutation(len(y))
    tr, va, te = perm[:1000], perm[1000:1300], perm[1300:]

    m = MLP([64, 32, 32, 10])
    print(f"Backprop vs. finite differences, max relative error over 20 weights: {grad_check(m, X[:50], y[:50]):.1e}")
    m_res = MLP([64, 32, 32, 32, 10], residual=True)
    print(f"Same check with skip connections: {grad_check(m_res, X[:50], y[:50]):.1e}")

    print("Vanishing gradients: norm of the first-layer gradient at initialization")
    for depth in [2, 6, 12]:
        for act in ["sigmoid", "relu"]:
            mm = MLP([64] + [32] * depth + [10], activation=act, seed=1)
            print(f"  {depth:2d} hidden layers, {act:>7}: {first_layer_grad_norm(mm, X[tr], y[tr]):.2e}")

    for name, model in [("linear softmax (no hidden layer)", MLP([64, 10])),
                        ("MLP 64-64-10, ReLU", MLP([64, 64, 10])),
                        ("MLP 64-64-64-10, ReLU", MLP([64, 64, 64, 10]))]:
        hist = train(model, X[tr], y[tr], X[va], y[va])
        print(f"{name:>34}: best validation error {min(hist):.3f} (epoch {int(np.argmin(hist)) + 1}), "
              f"test error {np.mean(model.predict(X[te]) != y[te]):.3f}")


if __name__ == "__main__":
    demo()
