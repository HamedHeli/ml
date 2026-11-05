"""Feature selection and regularization (CPSC 340 L15-L18).

Feature selection: search and score
    Forward selection: start with no features; repeatedly add the feature that most
    lowers score(S) = training error + complexity penalty, stop when nothing helps.
    Penalties: AIC adds |S|, BIC adds (1/2) log(n) |S| (in units where the error is
    the negative log-likelihood, here n/2 log(SSE/n) for a Gaussian model).

L2 regularization (ridge)
    f(w) = 1/2 ||Xw - y||^2 + lambda/2 ||w||^2,  w = (X^T X + lambda I)^{-1} X^T y.
    Always has a unique solution, shrinks weights, lowers E_gap. Never sets a
    weight exactly to zero.

L1 regularization (lasso)
    f(w) = 1/2 ||Xw - y||^2 + lambda ||w||_1. Non-smooth at 0, so solutions are
    sparse: it does feature selection and regularization at once. Solved here by
    proximal gradient (ISTA): a gradient step on the smooth part, then
    soft-thresholding w_j <- sign(w_j) max(|w_j| - alpha lambda, 0).

Standardization
    Regularization penalizes all weights equally, so features must be on the same
    scale: subtract the mean and divide by the std, using *training* statistics only.

Run:  python 05-feature-selection-and-regularization/regularization.py
"""

from __future__ import annotations

import numpy as np


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def soft_threshold(w, t):
    return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)


def lasso(X, y, lam, max_iter=5000, tol=1e-9):
    L = np.linalg.eigvalsh(X.T @ X).max()
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        w_new = soft_threshold(w - X.T @ (X @ w - y) / L, lam / L)
        if np.abs(w_new - w).max() < tol:
            return w_new
        w = w_new
    return w


def forward_selection(X, y, penalty="bic"):
    n, d = X.shape

    def score(S):
        if not S:
            sse = np.sum(y ** 2)
        else:
            w = np.linalg.lstsq(X[:, S], y, rcond=None)[0]
            sse = np.sum((X[:, S] @ w - y) ** 2)
        nll = n / 2 * np.log(sse / n)
        return nll + (len(S) if penalty == "aic" else 0.5 * np.log(n) * len(S))

    S, best = [], score([])
    while True:
        cands = [(score(S + [j]), j) for j in range(d) if j not in S]
        if not cands:
            return S
        s, j = min(cands)
        if s >= best:
            return S
        S, best = S + [j], s


def standardize(X_train, X_test):
    mu, sd = X_train.mean(0), X_train.std(0)
    sd[sd == 0] = 1
    return (X_train - mu) / sd, (X_test - mu) / sd


def demo():
    rng = np.random.default_rng(0)
    n, d = 100, 50
    w_true = np.zeros(d)
    w_true[:5] = [3, -2, 1.5, 1, -1]
    X = rng.standard_normal((n, d))
    y = X @ w_true + 0.5 * rng.standard_normal(n)
    Xt = rng.standard_normal((2000, d))
    yt = Xt @ w_true
    mse = lambda w: np.mean((Xt @ w - yt) ** 2)
    print(f"Sparse truth: 5 relevant of {d} features, n = {n}")
    w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"  least squares            test MSE {mse(w_ls):.3f}, nonzeros {np.sum(np.abs(w_ls) > 1e-8)}")
    for lam in [1, 10, 100]:
        w = ridge(X, y, lam)
        print(f"  ridge   lambda={lam:<5}     test MSE {mse(w):.3f}, nonzeros {np.sum(np.abs(w) > 1e-8)}")
    for lam in [1, 10, 50]:
        w = lasso(X, y, lam)
        print(f"  lasso   lambda={lam:<5}     test MSE {mse(w):.3f}, nonzeros {np.sum(np.abs(w) > 1e-8)}, "
              f"recovered {sorted(np.nonzero(np.abs(w) > 1e-8)[0].tolist())[:8]}")
    for pen in ["aic", "bic"]:
        S = forward_selection(X, y, pen)
        print(f"  forward selection ({pen}): selected {sorted(S)}")


if __name__ == "__main__":
    demo()
