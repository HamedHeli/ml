"""Exponential families (CPSC 440 lecture 10).

    p(x | w) = h(x) exp(w^T T(x) - A(w)),   A(w) = log sum_x h(x) exp(w^T T(x))

w are the canonical (natural) parameters, T(x) the sufficient statistics, A the
log-partition function. Two facts make everything work:

    grad A(w)    = E[T(X)]        (the mean of the sufficient statistics)
    Hessian A(w) = Cov[T(X)]      (positive semi-definite, so A is convex)

So the negative log-likelihood  n A(w) - w^T sum_i T(x_i)  is convex, and its
gradient vanishes when  E_w[T(X)] = (1/n) sum_i T(x_i): **MLE = moment matching**.

Generalized linear models: make the canonical parameter depend on features,
w = theta^T x. Bernoulli gives logistic regression, Gaussian gives least squares,
Poisson (counts) gives Poisson regression, implemented below.

Run:  python 07-density-estimation-mle-map/exponential_family.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


# Each family: T(x), A(w), grad A(w). Moment matching solves grad A(w) = mean T.
FAMILIES = {
    "bernoulli": dict(T=lambda x: x, A=lambda w: np.logaddexp(0, w), dA=lambda w: 1 / (1 + np.exp(-w)),
                      inv=lambda mu: np.log(mu / (1 - mu))),
    "poisson": dict(T=lambda x: x, A=lambda w: np.exp(w), dA=lambda w: np.exp(w),
                    inv=lambda mu: np.log(mu)),
}


def mle_by_gradient(family, x, iters=200):
    """Newton's method on the convex NLL A(w) - w mean(T(x)); compares with the closed-form inverse."""
    f = FAMILIES[family]
    t_bar = np.mean(f["T"](x))
    w = 0.0
    for _ in range(iters):
        eps = 1e-5
        hess = (f["dA"](w + eps) - f["dA"](w - eps)) / (2 * eps)
        w -= (f["dA"](w) - t_bar) / hess
    return w, f["inv"](t_bar)


def poisson_regression(X, y, lam=1e-3, iters=50):
    """GLM with log link: y ~ Poisson(exp(theta^T x)). Newton's method."""
    theta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(X @ theta)
        g = X.T @ (mu - y) + lam * theta
        H = X.T @ (mu[:, None] * X) + lam * np.eye(X.shape[1])
        theta -= np.linalg.solve(H, g)
    return theta


def poisson_nll(theta, X, y):
    eta = X @ theta
    return np.sum(np.exp(eta) - y * eta + gammaln(y + 1))


def demo():
    rng = np.random.default_rng(0)
    for fam, x in [("bernoulli", rng.uniform(size=500) < 0.2), ("poisson", rng.poisson(3.5, 500))]:
        w, w_closed = mle_by_gradient(fam, x.astype(float))
        print(f"{fam:>9}: canonical MLE by Newton {w:.4f}, by moment matching {w_closed:.4f}, "
              f"E[T] at the MLE {FAMILIES[fam]['dA'](w):.4f} = sample mean {x.mean():.4f}")
    n = 1000
    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, 2))])
    theta_true = np.array([0.5, 0.8, -0.4])
    y = rng.poisson(np.exp(X @ theta_true))
    theta = poisson_regression(X, y)
    print(f"Poisson regression: true theta {theta_true}, estimate {np.round(theta, 3)}")
    ls = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"  least squares on the counts predicts negative rates for {np.mean(X @ ls < 0):.1%} of inputs; "
          f"the GLM never does")


if __name__ == "__main__":
    demo()
