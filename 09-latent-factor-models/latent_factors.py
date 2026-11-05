"""Latent-factor models: PCA, robust PCA, matrix factorization for recommender
systems, and multi-dimensional scaling (CPSC 340 L25-L28).

PCA
    Approximate the centred data X (n x d) by Z W with Z (n x k), W (k x d):
        minimize ||Z W - X||_F^2.
    Solution: rows of W = top-k right singular vectors of X (orthonormal, ordered
    by variance explained); Z = X W^T. It is non-unique (any rotation of the
    factors gives the same span), so we fix it by orthonormality and ordering.
    Alternating minimization (solve for Z with W fixed, then W with Z fixed)
    reaches the same subspace and generalizes to other losses and missing data.

Robust PCA
    Replace the squared error by an absolute error, sum |<w_j, z_i> - x_ij|,
    (smoothed here), so sparse large errors (outliers, moving objects) are ignored.

Recommender systems: matrix factorization with missing entries
    ratings y_ij for the observed (user, item) pairs only:
        f = sum_{(i,j) observed} (b + b_i + b_j + z_i^T w_j - y_ij)^2
            + lambda (||Z||^2 + ||W||^2)
    Fit here by alternating least squares (ALS). Collaborative filtering uses
    only the rating matrix; content-based filtering uses item/user features.

Classical multi-dimensional scaling (MDS)
    Find low-dimensional points whose pairwise distances match the given ones.
    With Euclidean distances, double-centring the squared-distance matrix gives
    the Gram matrix and the solution is PCA on it. t-SNE replaces this with a
    probabilistic neighbourhood-matching objective (README).

Run:  python 09-latent-factor-models/latent_factors.py
"""

from __future__ import annotations

import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = Vt[: self.k]
        self.explained = (s[: self.k] ** 2).sum() / (s ** 2).sum()
        return self

    def compress(self, X):
        return (X - self.mu) @ self.W.T

    def expand(self, Z):
        return Z @ self.W + self.mu


def pca_alternating(X, k, iters=200, seed=0):
    """Alternating least squares for min ||ZW - X||^2 (X already centred)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((k, X.shape[1]))
    for _ in range(iters):
        Z = np.linalg.solve(W @ W.T, W @ X.T).T
        W = np.linalg.solve(Z.T @ Z, Z.T @ X)
    return Z, W


def robust_pca(X, k, iters=3000, eps=1e-3, lr=1e-3, seed=0):
    """Gradient descent on sum sqrt((ZW - X)^2 + eps): a smooth L1 loss."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    Z = 0.1 * rng.standard_normal((n, k))
    W = 0.1 * rng.standard_normal((k, d))
    for _ in range(iters):
        R = Z @ W - X
        G = R / np.sqrt(R ** 2 + eps)
        Z, W = Z - lr * G @ W.T, W - lr * Z.T @ G
    return Z, W


def subspace_angle(W1, W2):
    """Largest principal angle (degrees) between the row spaces of W1 and W2."""
    Q1, _ = np.linalg.qr(W1.T)
    Q2, _ = np.linalg.qr(W2.T)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s.min(), -1, 1)))


class MatrixFactorization:
    """ALS for ratings with biases. ``R`` is (n_users, n_items) with np.nan for missing."""

    def __init__(self, k=5, lam=5.0, iters=30):
        self.k, self.lam, self.iters = k, lam, iters

    def fit(self, R, seed=0):
        rng = np.random.default_rng(seed)
        n, m = R.shape
        M = ~np.isnan(R)
        self.b = np.nanmean(R)
        self.bu = np.zeros(n)
        self.bi = np.zeros(m)
        Z = 0.1 * rng.standard_normal((n, self.k))
        W = 0.1 * rng.standard_normal((m, self.k))
        I = self.lam * np.eye(self.k + 1)
        I[0, 0] = self.lam  # regularize biases too
        for _ in range(self.iters):
            for u in range(n):               # each user: ridge regression on [1, w_j]
                j = M[u]
                A = np.hstack([np.ones((j.sum(), 1)), W[j]])
                t = R[u, j] - self.b - self.bi[j]
                sol = np.linalg.solve(A.T @ A + I, A.T @ t)
                self.bu[u], Z[u] = sol[0], sol[1:]
            for i in range(m):               # each item: ridge regression on [1, z_u]
                u = M[:, i]
                A = np.hstack([np.ones((u.sum(), 1)), Z[u]])
                t = R[u, i] - self.b - self.bu[u]
                sol = np.linalg.solve(A.T @ A + I, A.T @ t)
                self.bi[i], W[i] = sol[0], sol[1:]
        self.Z, self.W = Z, W
        return self

    def predict(self):
        return self.b + self.bu[:, None] + self.bi[None] + self.Z @ self.W.T


def classical_mds(D, k=2):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J              # Gram matrix of the centred points
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0))


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    print("PCA on 8x8 digits (64 dimensions)")
    for k in [2, 10, 30]:
        p = PCA(k).fit(X)
        err = np.sum((p.expand(p.compress(X)) - X) ** 2) / np.sum((X - X.mean(0)) ** 2)
        print(f"  k={k:2d}: variance explained {p.explained:.3f}, relative squared reconstruction error {err:.3f}")
    Xc = X - X.mean(0)
    _, W_als = pca_alternating(Xc, 5)
    print(f"  alternating minimization vs. SVD subspace (k=5): largest angle {subspace_angle(W_als, PCA(5).fit(X).W):.2e} degrees")

    rng = np.random.default_rng(0)
    L = rng.standard_normal((200, 2)) @ rng.standard_normal((2, 20))
    Xo = L.copy()
    mask = rng.uniform(size=Xo.shape) < 0.05
    Xo[mask] += rng.choice([-20, 20], mask.sum())          # 5% gross corruptions
    _, W_pca = pca_alternating(Xo - Xo.mean(0), 2)
    _, W_rob = robust_pca(Xo, 2)
    _, W_true = pca_alternating(L, 2)
    print(f"Low-rank data with 5% gross outliers: angle to the true subspace, PCA {subspace_angle(W_pca, W_true):.1f} deg, "
          f"robust PCA {subspace_angle(W_rob, W_true):.1f} deg")

    # synthetic ratings: 300 users, 200 items, rank-3 tastes, 10% observed
    n, m = 300, 200
    U, V = rng.standard_normal((n, 3)), rng.standard_normal((m, 3))
    full = 3 + 0.5 * rng.standard_normal(n)[:, None] + 0.5 * rng.standard_normal(m)[None] + U @ V.T * 0.5
    obs = rng.uniform(size=(n, m)) < 0.10
    R = np.where(obs, full + 0.3 * rng.standard_normal((n, m)), np.nan)
    test = ~obs
    rmse = lambda P: np.sqrt(np.mean((P[test] - full[test]) ** 2))
    print("Recommender on synthetic ratings (10% observed), RMSE on unobserved entries:")
    print(f"  global mean                  {rmse(np.full((n, m), np.nanmean(R))):.3f}")
    mf0 = MatrixFactorization(k=0, lam=5).fit(R)
    print(f"  biases only (k=0)            {rmse(mf0.predict()):.3f}")
    mf = MatrixFactorization(k=3, lam=5).fit(R)
    print(f"  biases + rank-3 factors      {rmse(mf.predict()):.3f}")

    P = rng.standard_normal((50, 2)) @ rng.standard_normal((2, 10))
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    E = classical_mds(D, 2)
    D2 = np.sqrt(((E[:, None] - E[None]) ** 2).sum(-1))
    print(f"Classical MDS: 10-D points lying on a plane, max distance error after embedding in 2-D {np.abs(D - D2).max():.1e}")


if __name__ == "__main__":
    demo()
