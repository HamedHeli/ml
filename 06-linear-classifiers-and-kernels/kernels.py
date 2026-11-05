"""The kernel trick (CPSC 340 L21).

L2-regularized least squares on features Z = phi(X) has two equivalent solutions:

    w = (Z^T Z + lambda I_k)^{-1} Z^T y      "normal equations",  k x k solve
    v = (Z Z^T + lambda I_n)^{-1} y,  w = Z^T v   "other normal equations",  n x n solve

Predictions only need inner products: yhat = Z_test Z^T v = K_test v with
K = Z Z^T. If k(x, x') = phi(x)^T phi(x') can be computed without forming phi,
we can use huge or infinite bases:

    polynomial   k(x, x') = (1 + x^T x')^p        all monomials up to degree p
    Gaussian RBF k(x, x') = exp(-||x - x'||^2 / (2 sigma^2))   infinite-dimensional

Any loss whose solution can be written w = Z^T u (representer theorem) can be
kernelized; here logistic regression is fit directly over u.

Run:  python 06-linear-classifiers-and-kernels/kernels.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def poly_kernel(X, Z, p=2):
    return (1 + X @ Z.T) ** p


def rbf_kernel(X, Z, sigma=1.0):
    d2 = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * X @ Z.T
    return np.exp(-np.maximum(d2, 0) / (2 * sigma ** 2))


def explicit_poly2(X):
    """phi such that phi(x)^T phi(x') = (1 + x^T x')^2, for checking the trick."""
    n, d = X.shape
    cols = [np.ones(n)] + [np.sqrt(2) * X[:, j] for j in range(d)] + [X[:, j] ** 2 for j in range(d)]
    cols += [np.sqrt(2) * X[:, i] * X[:, j] for i in range(d) for j in range(i + 1, d)]
    return np.stack(cols, 1)


class KernelRidge:
    def __init__(self, kernel, lam=1e-3):
        self.kernel, self.lam = kernel, lam

    def fit(self, X, y):
        self.X = X
        self.v = np.linalg.solve(self.kernel(X, X) + self.lam * np.eye(len(X)), y)
        return self

    def predict(self, Xt):
        return self.kernel(Xt, self.X) @ self.v


class KernelLogistic:
    """Minimize sum_i log(1 + exp(-y_i (K u)_i)) + lambda/2 u^T K u by gradient descent."""

    def __init__(self, kernel, lam=1e-2, iters=2000):
        self.kernel, self.lam, self.iters = kernel, lam, iters

    def fit(self, X, y):
        self.X = X
        K = self.kernel(X, X)
        u = np.zeros(len(X))
        step = 1.0 / (np.linalg.eigvalsh(K).max() * (0.25 * np.linalg.eigvalsh(K).max() + self.lam))
        for _ in range(self.iters):
            m = y * (K @ u)
            u -= step * (K @ (-y * expit(-m)) + self.lam * K @ u)
        self.u = u
        return self

    def predict(self, Xt):
        return np.sign(self.kernel(Xt, self.X) @ self.u)


def demo():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    y = rng.standard_normal(30)
    Z = explicit_poly2(X)
    w = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ y)
    kr = KernelRidge(lambda A, B: poly_kernel(A, B, 2), 1e-3).fit(X, y)
    Xt = rng.standard_normal((5, 3))
    print("Degree-2 kernel trick vs. explicit features, max prediction difference:",
          f"{np.abs(explicit_poly2(Xt) @ w - kr.predict(Xt)).max():.1e}")

    from sklearn.datasets import make_circles
    Xc, yc = make_circles(400, noise=0.1, factor=0.4, random_state=0)
    yc = 2 * yc - 1
    Xct, yct = make_circles(1000, noise=0.1, factor=0.4, random_state=1)
    yct = 2 * yct - 1
    print("Concentric circles, test error:")
    lin = KernelLogistic(lambda A, B: A @ B.T).fit(Xc, yc)
    print(f"  linear kernel logistic   {np.mean(lin.predict(Xct) != yct):.3f}")
    for p in [2, 4]:
        m = KernelLogistic(lambda A, B: poly_kernel(A, B, p)).fit(Xc, yc)
        print(f"  poly-{p} kernel logistic   {np.mean(m.predict(Xct) != yct):.3f}")
    m = KernelLogistic(lambda A, B: rbf_kernel(A, B, 0.5)).fit(Xc, yc)
    print(f"  RBF kernel logistic      {np.mean(m.predict(Xct) != yct):.3f}")


if __name__ == "__main__":
    demo()
