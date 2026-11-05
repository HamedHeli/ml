"""Linear classifiers (CPSC 340 L18-L19, L22; CPSC 440 lectures 4-5).

All of these predict yhat = sign(w^T x) (binary) or argmax_c w_c^T x (multi-class).
They differ only in the loss used to train w:

    0-1 loss          [y w^T x <= 0]                 what we want; non-convex, flat
    perceptron        max(0, -y w^T x)               converges iff data separable
    hinge (SVM)       max(0, 1 - y w^T x)            convex; + L2 = maximize the margin
    logistic          log(1 + exp(-y w^T x))         convex, smooth; gives probabilities
    softmax           -w_y^T x + log sum_c exp(w_c^T x)   multi-class logistic

Logistic regression is also the MLE of p(y | x, w) = sigmoid(y w^T x); adding
lambda/2 ||w||^2 is MAP with a Gaussian prior (CPSC 440 lecture 4).

Stochastic gradient descent: use the gradient of one random example (or a
mini-batch) per step. Each step is O(d) instead of O(n d). It only converges with
a decreasing step size (e.g. alpha_t = alpha_0 / sqrt(t)) or with averaging.

Run:  python 06-linear-classifiers-and-kernels/linear_classifiers.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp


def perceptron(X, y, max_epochs=100):
    w = np.zeros(X.shape[1])
    for _ in range(max_epochs):
        mistakes = 0
        for i in range(len(y)):
            if y[i] * (X[i] @ w) <= 0:
                w += y[i] * X[i]
                mistakes += 1
        if mistakes == 0:
            break
    return w


def logistic_loss_grad(w, X, y, lam=0.0):
    m = y * (X @ w)
    f = np.sum(np.logaddexp(0, -m)) + lam / 2 * w @ w
    g = -X.T @ (y * expit(-m)) + lam * w
    return f, g


def hinge_loss_grad(w, X, y, lam=1.0):
    m = y * (X @ w)
    f = np.sum(np.maximum(0, 1 - m)) + lam / 2 * w @ w
    g = -X.T @ (y * (m < 1)) + lam * w      # a subgradient
    return f, g


def softmax_loss_grad(W, X, y, lam=0.0):
    """W has shape (d, C)."""
    S = X @ W
    P = np.exp(S - logsumexp(S, axis=1, keepdims=True))
    f = -S[np.arange(len(y)), y].sum() + logsumexp(S, axis=1).sum() + lam / 2 * np.sum(W ** 2)
    P[np.arange(len(y)), y] -= 1
    return f, X.T @ P + lam * W


def minimize(fun, w0, max_iter=500):
    """Gradient descent with backtracking (Armijo) line search."""
    w = w0.copy()
    f, g = fun(w)
    alpha = 1.0
    for _ in range(max_iter):
        while True:
            w_new = w - alpha * g
            f_new, g_new = fun(w_new)
            if f_new <= f - 1e-4 * alpha * np.sum(g * g) or alpha < 1e-12:
                break
            alpha /= 2
        w, f, g = w_new, f_new, g_new
        alpha *= 2
        if np.abs(g).max() < 1e-6:
            break
    return w


def sgd(fun_i, w0, n, epochs=10, alpha0=0.1, batch=1, seed=0, schedule="sqrt"):
    """fun_i(w, idx) -> (f, g) on the examples idx. Returns the averaged iterate."""
    rng = np.random.default_rng(seed)
    w, w_avg, t = w0.copy(), np.zeros_like(w0), 0
    for _ in range(epochs):
        for _ in range(n // batch):
            t += 1
            idx = rng.integers(0, n, batch)
            _, g = fun_i(w, idx)
            alpha = alpha0 / np.sqrt(t) if schedule == "sqrt" else alpha0
            w = w - alpha * g
            w_avg += (w - w_avg) / t
    return w, w_avg


def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = add_bias(X / 16)
    perm = np.random.default_rng(0).permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    yb = np.where(y == 8, 1, -1)              # binary task: is it an 8?
    print("Binary 'is it an 8?' on digits, test error")
    w = perceptron(X[tr], yb[tr])
    print(f"  perceptron          {np.mean(np.sign(X[te] @ w) != yb[te]):.3f}")
    w = minimize(lambda w: hinge_loss_grad(w, X[tr], yb[tr], 1.0), np.zeros(X.shape[1]))
    print(f"  SVM (hinge + L2)    {np.mean(np.sign(X[te] @ w) != yb[te]):.3f}")
    w = minimize(lambda w: logistic_loss_grad(w, X[tr], yb[tr], 1.0), np.zeros(X.shape[1]))
    print(f"  logistic + L2       {np.mean(np.sign(X[te] @ w) != yb[te]):.3f}")
    def softmax_flat(wflat):
        f, G = softmax_loss_grad(wflat.reshape(-1, 10), X[tr], y[tr], 1.0)
        return f, G.ravel()

    W = minimize(softmax_flat, np.zeros(X.shape[1] * 10)).reshape(-1, 10)
    print(f"Multi-class softmax regression, 10 digits: test error {np.mean((X[te] @ W).argmax(1) != y[te]):.3f}")

    print("SGD on L2-regularized logistic regression (10 epochs), final training objective")
    lam = 1.0
    f_full = lambda w: logistic_loss_grad(w, X[tr], yb[tr], lam)[0]
    w_star = minimize(lambda w: logistic_loss_grad(w, X[tr], yb[tr], lam), np.zeros(X.shape[1]))
    n = len(tr)

    def fun_i(w, idx):
        # unbiased estimate of the full gradient: n/|batch| times the batch gradient
        f, g = logistic_loss_grad(w, X[tr][idx], yb[tr][idx], lam * len(idx) / n)
        return f, g * n / len(idx)

    for sched, a0 in [("constant", 1e-3), ("sqrt", 1e-2)]:
        w_last, w_avg = sgd(fun_i, np.zeros(X.shape[1]), n, 10, a0, schedule=sched)
        print(f"  {sched:>8} step: last iterate {f_full(w_last):.2f}, averaged {f_full(w_avg):.2f}  (optimum {f_full(w_star):.2f})")


if __name__ == "__main__":
    demo()
