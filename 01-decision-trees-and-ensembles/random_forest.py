"""Random forests (CPSC 340 L7-L8): averaging many deep, de-correlated trees.

Two ingredients:
1. Bootstrap: each tree sees n examples drawn *with replacement* (about 63% unique).
2. Random trees: each split only considers a random subset of sqrt(d) features.

Deep trees have low E_train but a large E_gap. Averaging the votes of trees whose
errors are not perfectly correlated reduces the variance, so E_gap shrinks while
E_train stays near zero.

Run:  python 01-decision-trees-and-ensembles/random_forest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_tree import DecisionTree  # noqa: E402


class RandomForest:
    def __init__(self, n_trees=50, max_depth=np.inf, n_features="sqrt", bootstrap=True, seed=0):
        self.n_trees, self.max_depth = n_trees, max_depth
        self.n_features, self.bootstrap = n_features, bootstrap
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n, d = X.shape
        k = int(np.sqrt(d)) if self.n_features == "sqrt" else self.n_features
        self.n_classes = int(y.max()) + 1
        self.trees = []
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            t = DecisionTree(self.max_depth, "infogain", n_features=k, rng=self.rng)
            self.trees.append(t.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        votes = np.zeros((len(X), self.n_classes))
        for t in self.trees:
            votes[np.arange(len(X)), t.predict(X)] += 1
        return votes.argmax(1)


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    perm = np.random.default_rng(0).permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    single = DecisionTree(max_depth=np.inf).fit(X[tr], y[tr])
    print(f"one deep tree:       train err {np.mean(single.predict(X[tr]) != y[tr]):.3f}"
          f"  test err {np.mean(single.predict(X[te]) != y[te]):.3f}")
    for n_trees in [1, 10, 50]:
        rf = RandomForest(n_trees=n_trees).fit(X[tr], y[tr])
        print(f"random forest ({n_trees:2d}):  train err {np.mean(rf.predict(X[tr]) != y[tr]):.3f}"
              f"  test err {np.mean(rf.predict(X[te]) != y[te]):.3f}")
    print(f"fraction of unique points in a bootstrap sample: "
          f"{len(np.unique(np.random.default_rng(1).integers(0, 10000, 10000))) / 10000:.3f}"
          f" (theory 1 - 1/e = {1 - np.exp(-1):.3f})")


if __name__ == "__main__":
    demo()
