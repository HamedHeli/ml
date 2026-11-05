"""Bayesian linear regression and empirical Bayes (CPSC 440 lectures 6-9).

Model:  y | x, w ~ N(w^T phi(x), sigma^2),  prior w ~ N(0, (1/lambda) I).

Posterior (conjugate, so exactly Gaussian):
    Sigma_post = (Phi^T Phi / sigma^2 + lambda I)^{-1}
    mu_post    = Sigma_post Phi^T y / sigma^2              (= the ridge / MAP solution)

Posterior predictive ("regression with error bars"):
    y* | x*, data ~ N(mu_post^T phi*, sigma^2 + phi*^T Sigma_post phi*)
    The second term is the model's uncertainty about w; it grows away from the data.

Empirical Bayes / type-II maximum likelihood:
    pick hyper-parameters (lambda, sigma^2, even the basis) by maximizing the
    marginal likelihood p(y | X, lambda, sigma) = N(y | 0, sigma^2 I + Phi Phi^T / lambda),
    using all the training data and no validation set. It automatically trades
    fit against complexity ("Bayesian Occam's razor").

Run:  python 08-gaussians-and-bayesian-learning/bayesian_linear_regression.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class BayesianLinearRegression:
    def __init__(self, lam=1.0, sigma2=1.0):
        self.lam, self.sigma2 = lam, sigma2

    def fit(self, Phi, y):
        d = Phi.shape[1]
        P = Phi.T @ Phi / self.sigma2 + self.lam * np.eye(d)
        self.Sigma = np.linalg.inv(P)
        self.mu = self.Sigma @ Phi.T @ y / self.sigma2
        return self

    def predict(self, Phi_t):
        mean = Phi_t @ self.mu
        var = self.sigma2 + np.einsum("ij,jk,ik->i", Phi_t, self.Sigma, Phi_t)
        return mean, var


def log_marginal_likelihood(Phi, y, lam, sigma2):
    n = len(y)
    C = sigma2 * np.eye(n) + Phi @ Phi.T / lam
    L = np.linalg.cholesky(C)
    a = np.linalg.solve(L, y)
    return -0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)


def empirical_bayes(Phi, y):
    """Maximize the marginal likelihood over (log lambda, log sigma^2)."""
    obj = lambda t: -log_marginal_likelihood(Phi, y, np.exp(t[0]), np.exp(t[1]))
    res = minimize(obj, np.zeros(2), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
    return np.exp(res.x[0]), np.exp(res.x[1]), -res.fun


def poly(x, p):
    return np.vander(x, p + 1, increasing=True)


def demo():
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * x) + 0.3 * x
    x = rng.uniform(-2, 2, 25)
    y = f(x) + 0.2 * rng.standard_normal(25)

    print("Choosing the polynomial degree by marginal likelihood (no validation set):")
    xt = np.linspace(-2, 2, 400)
    best = None
    for p in range(0, 11):
        lam, s2, lml = empirical_bayes(poly(x, p), y)
        m = BayesianLinearRegression(lam, s2).fit(poly(x, p), y)
        mse = np.mean((m.predict(poly(xt, p))[0] - f(xt)) ** 2)
        print(f"  degree {p:2d}: log p(y) = {lml:7.2f}, lambda = {lam:9.3g}, sigma = {np.sqrt(s2):.3f}, test MSE {mse:.4f}")
        if best is None or lml > best[1]:
            best = (p, lml, lam, s2)
    p, _, lam, s2 = best
    print(f"  -> empirical Bayes picks degree {p}")

    m = BayesianLinearRegression(lam, s2).fit(poly(x, p), y)
    for xs in [0.0, 1.9, 3.0]:
        mean, var = m.predict(poly(np.array([xs]), p))
        print(f"  predictive at x = {xs}: {mean[0]:6.2f} +/- {2 * np.sqrt(var[0]):.2f} (truth {f(xs):.2f})")


if __name__ == "__main__":
    demo()
