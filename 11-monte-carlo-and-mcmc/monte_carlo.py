"""Monte Carlo, rejection sampling, importance sampling and the Laplace approximation
(CPSC 440 lecture 12).

Monte Carlo:  E_p[f(X)] ~ (1/n) sum_i f(x_i), x_i ~ p. Unbiased, variance Var(f)/n,
so the error shrinks like 1/sqrt(n) regardless of dimension.

Rejection sampling: to sample from p ~ p_tilde, propose x ~ q and accept with
probability p_tilde(x) / (M q(x)), where M >= max p_tilde / q. Exact samples, but
the acceptance rate is Z/M, which collapses in high dimension.

Importance sampling: E_p[f] = E_q[f p/q]. With an unnormalized p_tilde use
self-normalized weights w_i = p_tilde(x_i)/q(x_i) / sum_j (...): consistent, biased,
and high-variance when q has lighter tails than p.

Laplace approximation: a Gaussian at the mode of the posterior with covariance =
inverse Hessian of the negative log-posterior. Cheap, but it is local.

Running example (the lecture's motivating problem): Bayesian logistic regression,
whose posterior is not conjugate, so the predictive
    p(y = 1 | x, data) = E_{w ~ posterior}[sigmoid(w^T x)]
has to be approximated.

Run:  python 11-monte-carlo-and-mcmc/monte_carlo.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp


def make_problem(seed=0, n=30):
    rng = np.random.default_rng(seed)
    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, 1))])
    w_true = np.array([0.5, 2.0])
    y = np.where(rng.uniform(size=n) < expit(X @ w_true), 1, -1)
    return X, y


def log_post(W, X, y, lam=1.0):
    """Unnormalized log posterior for a batch of weight vectors W (m, d)."""
    return -np.logaddexp(0, -(W @ X.T) * y).sum(1) - lam / 2 * (W ** 2).sum(1)


def laplace(X, y, lam=1.0, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(y * (X @ w))
        g = -X.T @ (y * (1 - p)) + lam * w
        H = X.T @ ((p * (1 - p))[:, None] * X) + lam * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w, np.linalg.inv(H)


def grid_truth(X, y, x_star, lam=1.0, m=400, lim=6):
    """Brute-force posterior expectation on a 2D grid (only feasible in 2D)."""
    a = np.linspace(-lim, lim, m)
    W = np.stack(np.meshgrid(a, a), -1).reshape(-1, 2)
    lp = log_post(W, X, y, lam)
    p = np.exp(lp - logsumexp(lp))
    return p @ expit(W @ x_star)


def rejection_sample(X, y, n, rng, lam=1.0, max_tries=10_000_000):
    """Proposal q = prior N(0, I/lam), target p_tilde = prior * likelihood.

    p_tilde / q = likelihood, so M = max_w likelihood(w), the likelihood at the
    (unregularized) MLE, and we accept with probability lik(w) / lik(w_MLE).
    """
    w_mle, _ = laplace(X, y, lam=1e-8)
    loglik = lambda W: -np.logaddexp(0, -(W @ X.T) * y).sum(1)
    log_M = loglik(w_mle[None])[0]
    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        W = rng.standard_normal((10000, X.shape[1])) / np.sqrt(lam)
        acc = np.log(rng.uniform(size=len(W))) < loglik(W) - log_M
        out.extend(W[acc])
        tries += len(W)
    return np.array(out[:n]), len(out) / tries


def importance(X, y, x_star, n, rng, q_mu, q_cov, lam=1.0):
    L = np.linalg.cholesky(q_cov)
    W = q_mu + rng.standard_normal((n, len(q_mu))) @ L.T
    Z = np.linalg.solve(L, (W - q_mu).T)
    log_q = -0.5 * (Z ** 2).sum(0) - np.log(np.diag(L)).sum()
    lw = log_post(W, X, y, lam) - log_q
    w = np.exp(lw - logsumexp(lw))
    ess = 1 / np.sum(w ** 2)
    return w @ expit(W @ x_star), ess


def demo():
    rng = np.random.default_rng(1)
    print("Monte Carlo estimate of E[X^2] for X ~ N(0,1) (truth 1):")
    for n in [10, 1000, 100000]:
        print(f"  n={n:6d}: {np.mean(rng.standard_normal(n) ** 2):.4f}")

    X, y = make_problem()
    x_star = np.array([1.0, 1.0])
    truth = grid_truth(X, y, x_star)
    w_map, cov = laplace(X, y)
    print(f"Bayesian logistic regression, p(y=1 | x*=1, data) by brute-force grid: {truth:.4f}")
    print(f"  plug-in MAP              {expit(w_map @ x_star):.4f}   (ignores uncertainty in w)")
    Wl = w_map + rng.standard_normal((20000, 2)) @ np.linalg.cholesky(cov).T
    print(f"  Laplace + Monte Carlo    {expit(Wl @ x_star).mean():.4f}")
    Wr, rate = rejection_sample(X, y, 2000, rng)
    print(f"  rejection sampling       {expit(Wr @ x_star).mean():.4f}   (acceptance rate {rate:.4f})")
    est, ess = importance(X, y, x_star, 20000, rng, w_map, cov * 1.5)
    print(f"  importance (Laplace q)   {est:.4f}   (effective sample size {ess:.0f} of 20000)")
    est, ess = importance(X, y, x_star, 20000, rng, np.zeros(2), np.eye(2))
    print(f"  importance (prior q)     {est:.4f}   (effective sample size {ess:.0f} of 20000)")


if __name__ == "__main__":
    demo()
