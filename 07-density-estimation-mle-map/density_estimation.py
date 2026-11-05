"""Discrete density estimation: Bernoulli, Beta, categorical, Dirichlet, product of
Bernoullis (CPSC 440 lectures 1-4, CPSC 340 L24).

Bernoulli, x in {0, 1}, p(x = 1 | theta) = theta
    MLE   theta = n1 / n
    MAP   with prior Beta(alpha, beta): theta = (n1 + alpha - 1) / (n + alpha + beta - 2)
    Bayes posterior Beta(n1 + alpha, n0 + beta); posterior predictive
          p(x = 1 | data) = (n1 + alpha) / (n + alpha + beta)
    The MLE says "never" after seeing 3 zeros; MAP and Bayes do not.

Categorical, x in {1..k}, p(x = c | theta) = theta_c
    MLE   theta_c = n_c / n  (derived with a Lagrange multiplier, or by an
          unnormalized parameterization, lecture 4)
    MAP   with Dirichlet(alpha): theta_c = (n_c + alpha_c - 1) / (n + sum alpha - k)
    Sampling by inverse CDF: draw u ~ U(0, 1), return the first c with cumsum >= u.

Product of Bernoullis: x in {0, 1}^d, all coordinates independent. Learning is d
separate Bernoulli MLEs. It is a terrible model of images (no dependencies) but
the building block of naive Bayes, mixtures (10) and Markov chains (13).

Run:  python 07-density-estimation-mle-map/density_estimation.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, gammaln


class BetaBernoulli:
    def __init__(self, alpha=1.0, beta=1.0):
        self.a, self.b = alpha, beta

    def fit(self, x):
        self.n1, self.n0 = int(np.sum(x)), int(len(x) - np.sum(x))
        return self

    def mle(self):
        return self.n1 / (self.n1 + self.n0)

    def map(self):
        return (self.n1 + self.a - 1) / (self.n1 + self.n0 + self.a + self.b - 2)

    def posterior_predictive(self):
        return (self.n1 + self.a) / (self.n1 + self.n0 + self.a + self.b)

    def log_marginal_likelihood(self):
        """log p(data) = log B(n1 + a, n0 + b) - log B(a, b), used for empirical Bayes (08)."""
        return betaln(self.n1 + self.a, self.n0 + self.b) - betaln(self.a, self.b)


def categorical_mle(x, k):
    return np.bincount(x, minlength=k) / len(x)


def dirichlet_map(x, alpha):
    counts = np.bincount(x, minlength=len(alpha))
    return (counts + alpha - 1) / (len(x) + alpha.sum() - len(alpha))


def dirichlet_log_marginal(x, alpha):
    counts = np.bincount(x, minlength=len(alpha))
    return (gammaln(alpha.sum()) - gammaln(alpha.sum() + len(x))
            + np.sum(gammaln(alpha + counts) - gammaln(alpha)))


def sample_categorical(theta, size, rng):
    """Inverse-CDF sampling: O(k) setup, O(log k) per sample with binary search."""
    cdf = np.cumsum(theta)
    return np.searchsorted(cdf, rng.uniform(0, cdf[-1], size))


class ProductOfBernoullis:
    def __init__(self, alpha=1.0):
        self.alpha = alpha        # Beta(alpha, alpha) prior; alpha = 1 gives the MLE

    def fit(self, X):
        self.theta = (X.sum(0) + self.alpha - 1) / (len(X) + 2 * self.alpha - 2)
        return self

    def log_likelihood(self, X):
        t = np.clip(self.theta, 1e-12, 1 - 1e-12)
        return X @ np.log(t) + (1 - X) @ np.log1p(-t)

    def sample(self, n, rng):
        return (rng.uniform(size=(n, len(self.theta))) < self.theta).astype(int)


def demo():
    rng = np.random.default_rng(0)
    x = np.array([0, 0, 0])
    m = BetaBernoulli(2, 2).fit(x)
    print(f"Three tails in a row: MLE {m.mle():.2f}, MAP(Beta(2,2)) {m.map():.2f}, "
          f"posterior predictive {m.posterior_predictive():.2f}")
    x = rng.uniform(size=1000) < 0.3
    m = BetaBernoulli(2, 2).fit(x)
    print(f"1000 flips of a 0.3 coin: MLE {m.mle():.3f}, MAP {m.map():.3f} (the prior washes out)")

    theta = np.array([0.5, 0.3, 0.15, 0.05])
    s = sample_categorical(theta, 20, rng)
    print(f"Categorical, 20 samples: MLE {np.round(categorical_mle(s, 4), 2)}, "
          f"Dirichlet(2) MAP {np.round(dirichlet_map(s, np.full(4, 2.0)), 2)}, truth {theta}")

    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    X = (X > 7).astype(int)
    pob = ProductOfBernoullis(2).fit(X[:1200])
    print(f"Product of Bernoullis on binarized digits: mean test log-likelihood "
          f"{pob.log_likelihood(X[1200:]).mean():.1f} nats per 64-pixel image")
    samp = pob.sample(1, rng)[0].reshape(8, 8)
    print("A sample (looks like noise, because pixels are modeled independently):")
    for row in samp:
        print("   " + "".join("#" if v else "." for v in row))


if __name__ == "__main__":
    demo()
