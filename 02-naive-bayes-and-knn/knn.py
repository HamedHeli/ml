"""k-nearest neighbours (CPSC 340 L6-L7) and the curse of dimensionality.

kNN is the standard *non-parametric* model: there is no training, the model is
the training set, and its size grows with n. To classify x, find the k training
points closest to x and take the majority label.

Distances are computed in one vectorized step using
    ||x - z||^2 = ||x||^2 + ||z||^2 - 2 x.z      (L6 bonus "Vectorized Distance Calculation")
which turns the O(n t d) triple loop into one matrix product.

Run:  python 02-naive-bayes-and-knn/knn.py
"""

from __future__ import annotations

import numpy as np


def sq_dists(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, shape (len(X), len(Z))."""
    d = (X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * X @ Z.T
    return np.maximum(d, 0.0)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y          # "training" is memorizing
        self.n_classes = int(y.max()) + 1
        return self

    def predict(self, Xt, batch=1024):
        out = np.empty(len(Xt), dtype=int)
        for s in range(0, len(Xt), batch):
            D = sq_dists(Xt[s:s + batch], self.X)
            nn = np.argpartition(D, self.k - 1, axis=1)[:, : self.k]
            votes = np.zeros((len(nn), self.n_classes))
            np.add.at(votes, (np.arange(len(nn))[:, None], self.y[nn]), 1)
            out[s:s + batch] = votes.argmax(1)
        return out


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = X.astype(float)
    perm = np.random.default_rng(0).permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    print("kNN on 8x8 digits")
    for k in [1, 3, 10, 50]:
        m = KNN(k).fit(X[tr], y[tr])
        print(f"  k={k:2d}: train err {np.mean(m.predict(X[tr]) != y[tr]):.3f}  test err {np.mean(m.predict(X[te]) != y[te]):.3f}")

    print("Curse of dimensionality: ratio of nearest to farthest distance among 1000 uniform points")
    rng = np.random.default_rng(0)
    for d in [2, 10, 100, 1000]:
        P = rng.uniform(size=(1000, d))
        D = np.sqrt(sq_dists(P[:1], P[1:]))[0]
        print(f"  d={d:4d}: min/max = {D.min() / D.max():.3f}")


if __name__ == "__main__":
    demo()
