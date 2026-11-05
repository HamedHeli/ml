"""Boosting (CPSC 340 L23): AdaBoost and gradient tree boosting.

Averaging (random forests) fixes models with a large E_gap. Boosting fixes models
with a large E_train: fit simple models *sequentially*, each one focusing on what
the ensemble so far gets wrong.

AdaBoost (binary labels y in {-1, +1}, base learner = decision stump)
    weights w_i = 1/n
    for m = 1..M:
        h_m   = stump fit with weighted accuracy
        err_m = sum_i w_i [h_m(x_i) != y_i] / sum_i w_i
        a_m   = 1/2 log((1 - err_m) / err_m)
        w_i  *= exp(-a_m y_i h_m(x_i))           (mistakes get exponentially heavier)
    predict sign(sum_m a_m h_m(x))
This is coordinate descent on the exponential loss sum_i exp(-y_i f(x_i)).

Gradient tree boosting (squared loss, as in the XGBoost slides)
    f_0 = 0
    for m = 1..M:  fit a regression tree T_m to the residuals r_i = y_i - f_{m-1}(x_i)
                   f_m = f_{m-1} + eta * T_m
The residual is the negative gradient of 1/2 (y - f)^2 with respect to f, so each
tree is a step of gradient descent in function space. ``eta`` (shrinkage) and the
depth limit are the regularizers; XGBoost adds an L2 penalty on leaf values.

Run:  python 01-decision-trees-and-ensembles/boosting.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_tree import DecisionTree, fit_stump  # noqa: E402


class AdaBoost:
    def __init__(self, n_rounds=50):
        self.n_rounds = n_rounds

    def fit(self, X, y):
        """y in {-1, +1}."""
        n = len(y)
        w = np.full(n, 1 / n)
        y01 = (y > 0).astype(int)
        self.stumps, self.alphas = [], []
        for _ in range(self.n_rounds):
            s = fit_stump(X, y01, "accuracy", 2, w)
            h = 2 * s.predict(X) - 1
            err = np.clip(w[h != y].sum() / w.sum(), 1e-12, 1 - 1e-12)
            if err >= 0.5:
                break
            a = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-a * y * h)
            w /= w.sum()
            self.stumps.append(s)
            self.alphas.append(a)
        return self

    def decision_function(self, X):
        return sum(a * (2 * s.predict(X) - 1) for a, s in zip(self.alphas, self.stumps))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class GradientBoostedTrees:
    def __init__(self, n_trees=100, max_depth=3, eta=0.1):
        self.n_trees, self.max_depth, self.eta = n_trees, max_depth, eta

    def fit(self, X, y):
        self.trees = []
        f = np.zeros(len(y))
        self.train_mse = []
        for _ in range(self.n_trees):
            t = DecisionTree(self.max_depth, "squared").fit(X, y - f)
            f = f + self.eta * t.predict(X)
            self.trees.append(t)
            self.train_mse.append(np.mean((y - f) ** 2))
        return self

    def predict(self, X):
        return self.eta * sum(t.predict(X) for t in self.trees)


def demo():
    rng = np.random.default_rng(0)
    # AdaBoost on a circle: no single axis-aligned stump can do it
    X = rng.uniform(-1, 1, (600, 2))
    y = np.where((X ** 2).sum(1) < 0.5, 1, -1)
    Xte = rng.uniform(-1, 1, (2000, 2))
    yte = np.where((Xte ** 2).sum(1) < 0.5, 1, -1)
    print("AdaBoost with decision stumps on a disc (not separable by one axis-aligned rule)")
    for m in [1, 10, 50, 200]:
        ab = AdaBoost(m).fit(X, y)
        print(f"  {m:3d} stumps: train err {np.mean(ab.predict(X) != y):.3f}  test err {np.mean(ab.predict(Xte) != yte):.3f}")

    # gradient boosting on a 1D regression problem
    x = rng.uniform(0, 6, (300, 1))
    yr = np.sin(x[:, 0]) * x[:, 0] + 0.3 * rng.standard_normal(300)
    xt = rng.uniform(0, 6, (2000, 1))
    yt = np.sin(xt[:, 0]) * xt[:, 0]
    print("Gradient tree boosting (depth-2 trees, eta=0.1) on y = x sin(x) + noise")
    gb = GradientBoostedTrees(200, 2, 0.1).fit(x, yr)
    for m in [1, 10, 50, 200]:
        pred = gb.eta * sum(t.predict(xt) for t in gb.trees[:m])
        print(f"  {m:3d} trees: train MSE {gb.train_mse[m - 1]:.3f}  test MSE vs. clean f {np.mean((pred - yt) ** 2):.3f}")


if __name__ == "__main__":
    demo()
