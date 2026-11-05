"""Mixture models, expectation maximization and kernel density estimation
(CPSC 440 lecture 11).

Mixture model:  p(x) = sum_c pi_c p(x | theta_c)
Latent-variable view: z ~ Cat(pi), x | z = c ~ p(x | theta_c). Ancestral sampling
draws z, then x.

Responsibilities (E-step):  r_ic = p(z_i = c | x_i) = pi_c p(x_i | theta_c) / sum_c' (...)
computed in log space with log-sum-exp to avoid underflow.

EM (M-step): maximize the responsibility-weighted log-likelihood. For a
Gaussian mixture:
    pi_c = (1/n) sum_i r_ic
    mu_c = sum_i r_ic x_i / sum_i r_ic
    Sigma_c = sum_i r_ic (x_i - mu_c)(x_i - mu_c)^T / sum_i r_ic   (+ eps I)
For a mixture of Bernoullis: theta_cj = sum_i r_ic x_ij / sum_i r_ic.
Each EM iteration never decreases the log-likelihood (it maximizes a lower bound,
the ELBO, that touches the likelihood at the current parameters). Only a local
optimum is found; hard assignments + equal spherical covariances give k-means.

The MLE of a Gaussian mixture does not exist: put one mean on a data point and
shrink its variance to get infinite likelihood. The eps I term (a MAP-style fix)
prevents it.

Kernel density estimation (non-parametric mixture): one bump per data point,
    p(x) = (1/n) sum_i N(x | x_i, h^2 I),
with bandwidth h chosen by validation log-likelihood. Mean-shift clustering
climbs this density from every point and groups points that reach the same mode.

Run:  python 10-mixture-models-and-em/mixtures.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def gauss_logpdf(X, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    Z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * np.log(2 * np.pi)


class GaussianMixture:
    def __init__(self, k, eps=1e-3, iters=200, tol=1e-6, seed=0):
        self.k, self.eps, self.iters, self.tol = k, eps, iters, tol
        self.rng = np.random.default_rng(seed)

    def _log_joint(self, X):
        return np.stack([np.log(self.pi[c]) + gauss_logpdf(X, self.mu[c], self.Sigma[c])
                         for c in range(self.k)], 1)

    def fit(self, X):
        n, d = X.shape
        self.pi = np.full(self.k, 1 / self.k)
        self.mu = X[self.rng.choice(n, self.k, replace=False)]
        self.Sigma = np.stack([np.cov(X.T) + self.eps * np.eye(d)] * self.k)
        self.history = []
        for _ in range(self.iters):
            lj = self._log_joint(X)
            ll = logsumexp(lj, 1)
            self.history.append(ll.sum())
            R = np.exp(lj - ll[:, None])                       # E-step
            Nk = R.sum(0)                                      # M-step
            self.pi = Nk / n
            self.mu = (R.T @ X) / Nk[:, None]
            for c in range(self.k):
                D = X - self.mu[c]
                self.Sigma[c] = (R[:, c, None] * D).T @ D / Nk[c] + self.eps * np.eye(d)
            if len(self.history) > 1 and self.history[-1] - self.history[-2] < self.tol:
                break
        return self

    def log_likelihood(self, X):
        return logsumexp(self._log_joint(X), 1)

    def sample(self, n):
        z = self.rng.choice(self.k, n, p=self.pi)
        return np.stack([self.rng.multivariate_normal(self.mu[c], self.Sigma[c]) for c in z]), z


class BernoulliMixture:
    def __init__(self, k, alpha=1.0, iters=100, seed=0):
        self.k, self.alpha, self.iters = k, alpha, iters
        self.rng = np.random.default_rng(seed)

    def _log_joint(self, X):
        lt, l1t = np.log(self.theta), np.log1p(-self.theta)
        return X @ lt.T + (1 - X) @ l1t.T + np.log(self.pi)

    def fit(self, X):
        n, d = X.shape
        self.pi = np.full(self.k, 1 / self.k)
        self.theta = self.rng.uniform(0.25, 0.75, (self.k, d))
        for _ in range(self.iters):
            lj = self._log_joint(X)
            R = np.exp(lj - logsumexp(lj, 1, keepdims=True))
            Nk = R.sum(0)
            self.pi = (Nk + 1) / (n + self.k)       # Dirichlet(2) prior: no component gets pi = 0
            # Beta(alpha + 1, alpha + 1) prior keeps theta away from 0 and 1
            self.theta = (R.T @ X + self.alpha) / (Nk[:, None] + 2 * self.alpha)
        return self

    def log_likelihood(self, X):
        return logsumexp(self._log_joint(X), 1)


def kde_logpdf(Xq, Xtrain, h):
    d = Xq.shape[1]
    D2 = (Xq ** 2).sum(1)[:, None] + (Xtrain ** 2).sum(1)[None] - 2 * Xq @ Xtrain.T
    return logsumexp(-np.maximum(D2, 0) / (2 * h * h), 1) - np.log(len(Xtrain)) - d / 2 * np.log(2 * np.pi * h * h)


def mean_shift(X, h, iters=100):
    """Move every point uphill on the KDE; return mode labels."""
    M = X.copy()
    for _ in range(iters):
        D2 = (M ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2 * M @ X.T
        Wt = np.exp(-np.maximum(D2, 0) / (2 * h * h))
        M = Wt @ X / Wt.sum(1, keepdims=True)
    modes, labels = [], np.empty(len(X), int)
    for i, m in enumerate(M):
        for j, q in enumerate(modes):
            if np.linalg.norm(m - q) < h / 2:
                labels[i] = j
                break
        else:
            modes.append(m)
            labels[i] = len(modes) - 1
    return labels, np.array(modes)


def demo():
    rng = np.random.default_rng(0)
    true_mu = np.array([[0, 0], [4, 0], [2, 3.5]])
    X = np.vstack([rng.multivariate_normal(m, [[0.6, 0.3], [0.3, 0.6]] if i == 0 else np.eye(2) * 0.4, 300)
                   for i, m in enumerate(true_mu)])
    perm = rng.permutation(len(X))
    X = X[perm]
    tr, va = X[:600], X[600:]
    print("Gaussian mixtures with EM, validation log-likelihood per point:")
    for k in [1, 2, 3, 6]:
        g = GaussianMixture(k).fit(tr)
        mono = np.all(np.diff(g.history) > -1e-6)
        print(f"  k={k}: {g.log_likelihood(va).mean():.3f} (EM iterations {len(g.history)}, log-likelihood monotone: {mono})")
    g = GaussianMixture(3).fit(tr)
    print(f"  k=3 means: {np.round(g.mu[np.argsort(g.mu[:, 0] + g.mu[:, 1])], 2).tolist()}")
    print("Kernel density estimation, validation log-likelihood per point:")
    for h in [0.05, 0.3, 1.0, 3.0]:
        print(f"  h={h:4}: {kde_logpdf(va, tr, h).mean():.3f}")
    labels, modes = mean_shift(tr, 0.7)
    print(f"Mean-shift with h=0.7 finds {len(modes)} modes")

    from sklearn.datasets import load_digits
    Xd, _ = load_digits(return_X_y=True)
    Xd = (Xd > 7).astype(float)
    print("Mixture of Bernoullis on binarized digits, test log-likelihood per image:")
    for k in [1, 10, 30]:
        bm = BernoulliMixture(k).fit(Xd[:1200])
        print(f"  k={k:2d}: {bm.log_likelihood(Xd[1200:]).mean():.1f} nats")


if __name__ == "__main__":
    demo()
