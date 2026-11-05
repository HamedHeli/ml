"""Multivariate Gaussians and Gaussian discriminant analysis (CPSC 440 lectures 6-8).

N(x | mu, Sigma) = (2 pi)^{-d/2} |Sigma|^{-1/2} exp(-1/2 (x - mu)^T Sigma^{-1} (x - mu))

Closure properties used everywhere:
    affine     x ~ N(mu, Sigma)  =>  A x + b ~ N(A mu + b, A Sigma A^T)
               so sample with x = mu + L z, L L^T = Sigma (Cholesky), z ~ N(0, I)
    marginal   x_A ~ N(mu_A, Sigma_AA)                (just drop rows and columns)
    condition  x_A | x_B ~ N(mu_A + S_AB S_BB^{-1} (x_B - mu_B),
                              S_AA - S_AB S_BB^{-1} S_BA)

Learning
    MLE   mu = mean, Sigma = (1/n) sum (x - mu)(x - mu)^T   (singular if n <= d)
    MAP-like trace regularization: minimize NLL + lambda/2 Tr(Sigma^{-1}),
          which gives Sigma = S + (lambda / n) I, always invertible.

Gaussian discriminant analysis (GDA): generative classifier with x | y = c ~ N(mu_c, Sigma_c).
    QDA: each class has its own Sigma_c -> quadratic decision boundaries.
    LDA: one shared Sigma -> linear boundaries; with Sigma = I it is nearest class mean.

Run:  python 08-gaussians-and-bayesian-learning/gaussians.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def mvn_logpdf(X, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    Z = np.linalg.solve(L, (X - mu).T)                 # whitened residuals
    return -0.5 * (Z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * len(mu) * np.log(2 * np.pi)


def mvn_sample(mu, Sigma, n, rng):
    return mu + rng.standard_normal((n, len(mu))) @ np.linalg.cholesky(Sigma).T


def condition(mu, Sigma, idx_b, x_b):
    """Parameters of p(x_A | x_B = x_b), A = all indices not in idx_b."""
    idx_a = np.setdiff1d(np.arange(len(mu)), idx_b)
    Sab = Sigma[np.ix_(idx_a, idx_b)]
    Sbb = Sigma[np.ix_(idx_b, idx_b)]
    K = np.linalg.solve(Sbb, Sab.T).T
    return mu[idx_a] + K @ (x_b - mu[idx_b]), Sigma[np.ix_(idx_a, idx_a)] - K @ Sab.T


def fit_gaussian(X, lam=0.0):
    mu = X.mean(0)
    S = (X - mu).T @ (X - mu) / len(X)
    return mu, S + lam / len(X) * np.eye(X.shape[1])


class GDA:
    def __init__(self, shared_cov=False, lam=1.0):
        self.shared, self.lam = shared_cov, lam

    def fit(self, X, y):
        self.C = int(y.max()) + 1
        self.log_prior = np.log(np.bincount(y) / len(y))
        self.mu = np.stack([X[y == c].mean(0) for c in range(self.C)])
        if self.shared:
            R = X - self.mu[y]
            S = R.T @ R / len(X) + self.lam / len(X) * np.eye(X.shape[1])
            self.Sigma = [S] * self.C
        else:
            self.Sigma = [fit_gaussian(X[y == c], self.lam)[1] for c in range(self.C)]
        return self

    def log_posterior(self, X):
        lj = np.stack([mvn_logpdf(X, self.mu[c], self.Sigma[c]) for c in range(self.C)], 1) + self.log_prior
        return lj - logsumexp(lj, 1, keepdims=True)

    def predict(self, X):
        return self.log_posterior(X).argmax(1)


def demo():
    rng = np.random.default_rng(0)
    mu = np.array([1.0, -1.0, 0.5])
    A = rng.standard_normal((3, 3))
    Sigma = A @ A.T + 0.5 * np.eye(3)
    X = mvn_sample(mu, Sigma, 200000, rng)
    m_hat, S_hat = fit_gaussian(X)
    print(f"Cholesky sampling: max |Sigma_hat - Sigma| over 200k samples = {np.abs(S_hat - Sigma).max():.3f}")
    cm, cS = condition(mu, Sigma, np.array([2]), np.array([2.0]))
    near = np.abs(X[:, 2] - 2.0) < 0.02
    print(f"Conditioning on x3 = 2: formula mean {np.round(cm, 3)}, "
          f"Monte Carlo mean of samples with x3 near 2 {np.round(X[near, :2].mean(0), 3)}")

    from sklearn.datasets import load_digits
    Xd, y = load_digits(return_X_y=True)
    perm = rng.permutation(len(y))
    tr, te = perm[:1200], perm[1200:]
    for name, shared in [("QDA (one covariance per class)", False), ("LDA (shared covariance)", True)]:
        for lam in [1.0, 100.0]:
            g = GDA(shared, lam).fit(Xd[tr], y[tr])
            print(f"{name:>31}, lambda={lam:5}: test error {np.mean(g.predict(Xd[te]) != y[te]):.3f}")


if __name__ == "__main__":
    demo()
