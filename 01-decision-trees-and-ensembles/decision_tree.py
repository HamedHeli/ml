"""Decision stumps and decision trees (CPSC 340 L3), plus regression trees (L23).

A decision stump is one rule "if x_j > t predict a else predict b". It is learned
by *search and score*: try every feature j and every threshold t, score each
split, keep the best. A decision tree is learned by *greedy recursive splitting*:
fit a stump, then fit a new tree on each side, until a depth limit or a pure leaf.

Scores implemented
------------------
* ``accuracy``   classification accuracy of the stump (the first score in L3).
* ``infogain``   reduction in label entropy, H(y) - [n_L H(y_L) + n_R H(y_R)] / n.
                 Accuracy can fail to make progress (L3 "Example Where Accuracy Fails")
                 because a split can be useful without changing any leaf's majority.
* ``squared``    squared error for regression trees: leaves predict the mean of y.

Thresholds are searched in O(n d log n) by sorting each feature once and sweeping,
the speed-up described in the L3 bonus slides.

Run:  python 01-decision-trees-and-ensembles/decision_tree.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    """Entropy (nats) of the label distribution in each row of ``counts``."""
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts, dtype=float), where=n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)
    return h


@dataclass
class Stump:
    feature: int | None = None
    threshold: float = 0.0
    left: float | int = 0      # prediction when x[feature] <= threshold
    right: float | int = 0     # prediction when x[feature] >  threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.feature is None:
            return np.full(len(X), self.left)
        return np.where(X[:, self.feature] > self.threshold, self.right, self.left)


def fit_stump(X, y, score="infogain", n_classes=None, sample_weight=None, features=None) -> Stump:
    """Search and score over every (feature, threshold) pair.

    ``sample_weight`` lets AdaBoost reuse this (weighted accuracy / weighted infogain).
    ``features`` restricts the search to a subset (random forests).
    """
    n, d = X.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    features = range(d) if features is None else features
    regression = score == "squared"

    if regression:
        base_pred = np.average(y, weights=w)
        best_val = np.sum(w * (y - base_pred) ** 2)        # lower is better
    else:
        n_classes = int(y.max()) + 1 if n_classes is None else n_classes
        Y = np.eye(n_classes)[y] * w[:, None]               # weighted one-hot
        tot = Y.sum(0)
        base_pred = int(np.argmax(tot))
        best_val = tot.max() if score == "accuracy" else 0.0  # higher is better
        h_all = entropy_from_counts(tot[None])[0]

    best = Stump(None, 0.0, base_pred, base_pred)
    for j in features:
        order = np.argsort(X[:, j], kind="stable")
        xs = X[order, j]
        # candidate split after position i (left = first i+1 sorted points), only where x changes
        valid = np.nonzero(xs[:-1] < xs[1:])[0]
        if len(valid) == 0:
            continue
        if regression:
            ws, ys = w[order], y[order]
            cw, cwy, cwy2 = np.cumsum(ws), np.cumsum(ws * ys), np.cumsum(ws * ys ** 2)
            W, WY, WY2 = cw[-1], cwy[-1], cwy2[-1]
            lw, lwy, lwy2 = cw[valid], cwy[valid], cwy2[valid]
            rw, rwy, rwy2 = W - lw, WY - lwy, WY2 - lwy2
            sse = (lwy2 - lwy ** 2 / lw) + (rwy2 - rwy ** 2 / rw)
            i = int(np.argmin(sse))
            if sse[i] < best_val - 1e-12:
                best_val = sse[i]
                k = valid[i]
                best = Stump(j, (xs[k] + xs[k + 1]) / 2, lwy[i] / lw[i], rwy[i] / rw[i])
        else:
            left = np.cumsum(Y[order], axis=0)[valid]        # (m, C) counts on the left
            right = tot - left
            if score == "accuracy":
                vals = left.max(1) + right.max(1)
            else:
                nl, nr = left.sum(1), right.sum(1)
                vals = h_all - (nl * entropy_from_counts(left) + nr * entropy_from_counts(right)) / (nl + nr)
            i = int(np.argmax(vals))
            if vals[i] > best_val + 1e-12:
                best_val = vals[i]
                k = valid[i]
                best = Stump(j, (xs[k] + xs[k + 1]) / 2, int(np.argmax(left[i])), int(np.argmax(right[i])))
    return best


@dataclass
class Node:
    stump: Stump
    left: "Node | None" = None
    right: "Node | None" = None


class DecisionTree:
    """Greedy recursive splitting. ``score`` is "infogain", "accuracy" or "squared"."""

    def __init__(self, max_depth=5, score="infogain", min_samples=2, n_features=None, rng=None):
        self.max_depth, self.score, self.min_samples = max_depth, score, min_samples
        self.n_features = n_features      # random forests: features tried per split
        self.rng = rng or np.random.default_rng(0)

    def fit(self, X, y, sample_weight=None):
        self.n_classes = None if self.score == "squared" else int(y.max()) + 1
        w = np.ones(len(y)) if sample_weight is None else sample_weight
        self.root = self._grow(X, y, w, self.max_depth)
        return self

    def _grow(self, X, y, w, depth):
        feats = None
        if self.n_features is not None:
            feats = self.rng.choice(X.shape[1], self.n_features, replace=False)
        stump = fit_stump(X, y, self.score, self.n_classes, w, feats)
        if depth <= 1 or stump.feature is None or len(y) < self.min_samples:
            return Node(stump)
        mask = X[:, stump.feature] > stump.threshold
        return Node(stump,
                    self._grow(X[~mask], y[~mask], w[~mask], depth - 1),
                    self._grow(X[mask], y[mask], w[mask], depth - 1))

    def predict(self, X):
        out = np.empty(len(X), dtype=float if self.score == "squared" else int)
        self._predict(self.root, X, np.arange(len(X)), out)
        return out

    def _predict(self, node, X, idx, out):
        if node.left is None:
            out[idx] = node.stump.predict(X[idx])
            return
        mask = X[idx, node.stump.feature] > node.stump.threshold
        self._predict(node.left, X, idx[~mask], out)
        self._predict(node.right, X, idx[mask], out)


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    print("Decision trees on 8x8 digits (1200 train / 597 test)")
    print(f"{'depth':>5} {'score':>9} {'train err':>9} {'test err':>9}")
    for score in ["accuracy", "infogain"]:
        for depth in [1, 2, 4, 8, 12]:
            t = DecisionTree(depth, score).fit(X[tr], y[tr])
            e_tr = np.mean(t.predict(X[tr]) != y[tr])
            e_te = np.mean(t.predict(X[te]) != y[te])
            print(f"{depth:5d} {score:>9} {e_tr:9.3f} {e_te:9.3f}")


if __name__ == "__main__":
    demo()
