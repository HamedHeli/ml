"""Outlier detection (CPSC 340 L10): model-based, distance-based (global and local).

* z-score (model-based): fit a Gaussian, flag |x - mean| / std > threshold. The
  mean and std are themselves pulled by outliers, and it assumes one mode.
* kNN outlierness (global, distance-based): average distance to the k nearest
  neighbours. Good for "global" outliers far from everything.
* Outlierness ratio (local, distance-based): average kNN distance of x divided
  by the average kNN distance of its neighbours. Detects points that are far
  *relative to their neighbourhood*, so a point between a dense and a sparse
  cluster can be flagged even though its raw distance looks normal.

Run:  python 03-clustering-and-outliers/outliers.py
"""

from __future__ import annotations

import numpy as np


def sq_dists(X, Z):
    return np.maximum((X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * X @ Z.T, 0)


def z_scores(X):
    return np.abs((X - X.mean(0)) / X.std(0)).max(1)


def knn_outlierness(X, k=5):
    D = np.sqrt(sq_dists(X, X))
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    avg = np.take_along_axis(D, nn, 1).mean(1)
    return avg, nn


def local_outlierness(X, k=5):
    avg, nn = knn_outlierness(X, k)
    return avg / avg[nn].mean(1)


def demo():
    rng = np.random.default_rng(0)
    dense = rng.normal([0, 0], 0.1, (200, 2))
    sparse = rng.normal([3, 0], 1.0, (200, 2))
    local_out = np.array([[0.6, 0.0]])           # close to the dense cluster in absolute terms
    global_out = np.array([[8.0, 6.0]])
    X = np.vstack([dense, sparse, local_out, global_out])
    i_loc, i_glob = len(X) - 2, len(X) - 1
    for name, s in [("z-score", z_scores(X)), ("kNN distance", knn_outlierness(X)[0]),
                    ("local outlierness", local_outlierness(X))]:
        rank = (-s).argsort().argsort()
        print(f"{name:>18}: rank of global outlier {rank[i_glob] + 1:3d}, rank of local outlier {rank[i_loc] + 1:3d}  (of {len(X)})")


if __name__ == "__main__":
    demo()
