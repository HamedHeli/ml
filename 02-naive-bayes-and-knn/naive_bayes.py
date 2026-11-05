"""Naive Bayes (CPSC 340 L5-L6, CPSC 440 lecture 3) and decision theory.

Generative classifier: model p(x, y) = p(y) p(x | y), classify with Bayes rule
    p(y | x)  is proportional to  p(y) prod_j p(x_j | y)
The "naive" assumption is that the features x_j are conditionally independent
given the class. With binary features, p(x_j = 1 | y = c) = theta_{jc} is a
Bernoulli, so training is counting:

    theta_{jc} = (#{x_j = 1, y = c} + beta) / (#{y = c} + 2 beta)

``beta`` is Laplace smoothing (beta = 1). In CPSC 440 language this is the MAP
estimate under a Beta(beta + 1, beta + 1) prior; beta = 0 is the MLE, which
assigns probability 0 to anything unseen and can make every class impossible.

Predictions are computed in log space to avoid underflow (L6 bonus slide).

Decision theory: given a cost matrix C[yhat, y], predict the yhat minimizing the
expected cost sum_y C[yhat, y] p(y | x), instead of the most probable class.

Run:  python 02-naive-bayes-and-knn/naive_bayes.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class BernoulliNaiveBayes:
    def __init__(self, beta=1.0):
        self.beta = beta

    def fit(self, X, y):
        """X binary (n, d); y integer labels 0..C-1."""
        C = int(y.max()) + 1
        counts = np.array([(y == c).sum() for c in range(C)])
        ones = np.stack([X[y == c].sum(0) for c in range(C)])     # (C, d)
        self.log_prior = np.log(counts / counts.sum())
        self.theta = (ones + self.beta) / (counts[:, None] + 2 * self.beta)
        return self

    def log_joint(self, X):
        """log p(x, y = c) for every example and class, shape (n, C)."""
        with np.errstate(divide="ignore"):
            lt, l1t = np.log(self.theta), np.log1p(-self.theta)
        # 0 * log(0) terms would give nan; write the sum so they never occur
        ll = np.where(X[:, None, :] == 1, lt[None], l1t[None]).sum(-1)
        return ll + self.log_prior

    def predict_proba(self, X):
        lj = self.log_joint(X)
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))

    def predict(self, X, cost=None):
        """Most probable class, or the minimum-expected-cost class if ``cost[yhat, y]`` is given."""
        if cost is None:
            return self.log_joint(X).argmax(1)
        return (self.predict_proba(X) @ np.asarray(cost).T).argmin(1)


def demo():
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    X = (X > 7).astype(int)                       # binarize the 0..16 pixel intensities
    perm = np.random.default_rng(0).permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    for beta in [0.0, 1.0]:
        nb = BernoulliNaiveBayes(beta).fit(X[tr], y[tr])
        err = np.mean(nb.predict(X[te]) != y[te])
        impossible = np.mean(np.isinf(nb.log_joint(X[te])).all(1))
        print(f"Bernoulli naive Bayes, beta={beta}: test error {err:.3f}, "
              f"test digits that every class calls impossible: {impossible:.1%}")

    # decision theory: spam filtering where deleting real mail is 10x worse than missing spam
    yb = (y == 3).astype(int)                     # pretend "3" is spam
    nb = BernoulliNaiveBayes(1.0).fit(X[tr], yb[tr])
    cost = np.array([[0, 1],                      # predict not-spam: cost 1 if it was spam
                     [10, 0]])                    # predict spam: cost 10 if it was real mail
    for name, pred in [("most probable", nb.predict(X[te])), ("min expected cost", nb.predict(X[te], cost))]:
        c = cost[pred, yb[te]].sum()
        print(f"  {name:>17}: real mail flagged {np.sum((pred == 1) & (yb[te] == 0)):3d}, "
              f"spam missed {np.sum((pred == 0) & (yb[te] == 1)):3d}, total cost {c}")


if __name__ == "__main__":
    demo()
