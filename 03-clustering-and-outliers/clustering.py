"""Clustering (CPSC 340 L8-L10): k-means, k-means++, DBSCAN, agglomerative clustering.

k-means
    alternate (1) assign each x_i to its nearest mean, (2) set each mean to the
    average of its points. Each step cannot increase sum_i ||x_i - w_{c_i}||^2, so
    it converges, but only to a local optimum; restart and keep the lowest cost.
    k-means++ initializes means by sampling points with probability proportional
    to the squared distance to the nearest mean chosen so far. Clusters are
    convex (Voronoi cells), so k-means cannot find non-convex shapes.

DBSCAN (density-based)
    a point with >= min_neighbours points within radius eps is a *core* point;
    clusters are connected components of core points (plus their non-core
    neighbours). Finds non-convex clusters and labels sparse points as noise (-1),
    but needs eps and struggles when clusters have different densities.

Agglomerative (hierarchical, bottom-up)
    start with n clusters, repeatedly merge the closest pair. "Closest" between
    clusters is single link (min distance), complete link (max) or average link.
    Produces a whole tree (dendrogram); cut it to get k clusters.

Run:  python 03-clustering-and-outliers/clustering.py
"""

from __future__ import annotations

from collections import deque

import numpy as np


def sq_dists(X, Z):
    return np.maximum((X ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * X @ Z.T, 0)


def kmeans_pp_init(X, k, rng):
    W = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = sq_dists(X, np.array(W)).min(1)
        W.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(W)


def kmeans(X, k, n_restarts=10, init="++", max_iter=100, seed=0):
    """Returns (means, labels, cost) of the best restart."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        W = kmeans_pp_init(X, k, rng) if init == "++" else X[rng.choice(len(X), k, replace=False)]
        labels = None
        for _ in range(max_iter):
            new = sq_dists(X, W).argmin(1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                if np.any(labels == c):
                    W[c] = X[labels == c].mean(0)
        cost = sq_dists(X, W)[np.arange(len(X)), labels].sum()
        if best is None or cost < best[2]:
            best = (W.copy(), labels, cost)
    return best


def dbscan(X, eps, min_neighbours):
    """Labels 0..K-1 for clusters, -1 for noise."""
    D = sq_dists(X, X) <= eps ** 2
    neigh = [np.nonzero(row)[0] for row in D]
    core = np.array([len(nb) >= min_neighbours for nb in neigh])
    labels = np.full(len(X), -1)
    cluster = 0
    for i in range(len(X)):
        if not core[i] or labels[i] != -1:
            continue
        labels[i] = cluster
        queue = deque([i])
        while queue:                       # breadth-first expansion through core points
            j = queue.popleft()
            for nb in neigh[j]:
                if labels[nb] == -1:
                    labels[nb] = cluster
                    if core[nb]:
                        queue.append(nb)
        cluster += 1
    return labels


def agglomerative(X, k, linkage="single"):
    """Naive O(n^3) agglomerative clustering, cut at k clusters."""
    D = np.sqrt(sq_dists(X, X))
    clusters = {i: [i] for i in range(len(X))}
    # cluster-to-cluster distance matrix, updated with the Lance-Williams rule
    C = D.copy()
    np.fill_diagonal(C, np.inf)
    active = np.ones(len(X), bool)
    while len(clusters) > k:
        i, j = np.unravel_index(np.argmin(np.where(active[:, None] & active[None], C, np.inf)), C.shape)
        ni, nj = len(clusters[i]), len(clusters[j])
        if linkage == "single":
            new = np.minimum(C[i], C[j])
        elif linkage == "complete":
            new = np.maximum(C[i], C[j])
        else:  # average
            new = (ni * C[i] + nj * C[j]) / (ni + nj)
        C[i], C[:, i] = new, new
        C[i, i] = np.inf
        active[j] = False
        C[j], C[:, j] = np.inf, np.inf
        clusters[i] += clusters.pop(j)
    labels = np.empty(len(X), int)
    for c, members in enumerate(clusters.values()):
        labels[members] = c
    return labels


def adjusted_rand(a, b):
    """Adjusted Rand index (1 = same partition up to relabelling, ~0 = random)."""
    from math import comb
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)), int)
    np.add.at(M, (ia, ib), 1)
    s = sum(comb(int(x), 2) for x in M.ravel())
    sa = sum(comb(int(x), 2) for x in M.sum(1))
    sb = sum(comb(int(x), 2) for x in M.sum(0))
    e = sa * sb / comb(len(a), 2)
    return (s - e) / (0.5 * (sa + sb) - e)


def demo():
    from sklearn.datasets import make_blobs, make_moons

    Xb, yb = make_blobs(600, centers=4, cluster_std=0.8, random_state=1)
    Xm, ym = make_moons(400, noise=0.06, random_state=0)
    print("Adjusted Rand index against the true groups (1.0 = perfect)")
    print(f"{'method':>24} {'blobs':>6} {'moons':>6}")
    W, lb, _ = kmeans(Xb, 4)
    _, lm, _ = kmeans(Xm, 2)
    print(f"{'k-means++':>24} {adjusted_rand(lb, yb):6.2f} {adjusted_rand(lm, ym):6.2f}")
    print(f"{'DBSCAN':>24} {adjusted_rand(dbscan(Xb, 0.6, 5), yb):6.2f} {adjusted_rand(dbscan(Xm, 0.2, 5), ym):6.2f}")
    for link in ["single", "complete", "average"]:
        print(f"{'agglomerative ' + link:>24} {adjusted_rand(agglomerative(Xb, 4, link), yb):6.2f} "
              f"{adjusted_rand(agglomerative(Xm, 2, link), ym):6.2f}")
    costs = [kmeans(Xb, 4, n_restarts=1, init="random", seed=s)[2] for s in range(20)]
    print(f"k-means, 20 single random restarts on blobs: cost ranges {min(costs):.0f} to {max(costs):.0f} (local optima)")


if __name__ == "__main__":
    demo()
