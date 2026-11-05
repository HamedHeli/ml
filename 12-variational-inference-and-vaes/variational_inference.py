"""Variational inference with the reparameterization trick (CPSC 440 lectures 12-13).

Pick a family q_phi (here a full-covariance Gaussian) and find the member closest
to the posterior p(w | data) in *reverse* KL:

    KL(q || p) = E_q[log q(w) - log p(w | data)]
               = log p(data) - ELBO(phi),
    ELBO(phi)  = E_q[log p_tilde(w)] + H(q),     p_tilde = prior * likelihood.

So maximizing the ELBO minimizes KL(q || p), using only the unnormalized posterior.

Reparameterization trick: write w = mu + L eps with eps ~ N(0, I), so
    grad_phi E_q[f(w)] = E_eps[grad_phi f(mu + L eps)],
a low-variance Monte Carlo gradient that flows through the sample. (The
alternative, REINFORCE / score-function gradients, works for discrete q but is
much noisier.)

Reverse KL is "mode seeking": q prefers to sit inside one mode and to
under-estimate variance. Mean-field (diagonal) q cannot represent correlations.

Run:  python 12-variational-inference-and-vaes/variational_inference.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "11-monte-carlo-and-mcmc"))
from monte_carlo import grid_truth, make_problem  # noqa: E402


def log_p_tilde(w, X, y, lam=1.0):
    return -jnp.sum(jnp.logaddexp(0, -(X @ w) * y)) - lam / 2 * w @ w


def unpack(phi, d, mean_field):
    mu = phi[:d]
    if mean_field:
        L = jnp.diag(jnp.exp(phi[d:2 * d]))
    else:
        raw = phi[d:].reshape(d, d)
        L = jnp.tril(raw, -1) + jnp.diag(jnp.exp(jnp.diag(raw)))
    return mu, L


def neg_elbo(phi, key, X, y, d, mean_field, n_samples=64):
    mu, L = unpack(phi, d, mean_field)
    eps = jax.random.normal(key, (n_samples, d))
    W = mu + eps @ L.T                                      # reparameterized samples
    e_log_p = jnp.mean(jax.vmap(lambda w: log_p_tilde(w, X, y))(W))
    entropy = jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))         # + const
    return -(e_log_p + entropy)


def fit_vi(X, y, mean_field=False, steps=3000, seed=0):
    d = X.shape[1]
    phi = jnp.zeros(2 * d if mean_field else d + d * d)
    opt = optax.adam(1e-2)
    state = opt.init(phi)
    grad = jax.jit(jax.value_and_grad(neg_elbo), static_argnums=(4, 5))
    key = jax.random.PRNGKey(seed)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        _, g = grad(phi, sub, X, y, d, mean_field)
        upd, state = opt.update(g, state)
        phi = optax.apply_updates(phi, upd)
    return unpack(phi, d, mean_field)


def demo():
    Xn, yn = make_problem()
    X, y = jnp.asarray(Xn), jnp.asarray(yn, dtype=jnp.float32)
    x_star = np.array([1.0, 1.0])
    truth = grid_truth(Xn, yn, x_star)
    rng = np.random.default_rng(0)
    print(f"Bayesian logistic regression posterior predictive at x* = 1 (grid truth {truth:.4f})")
    for mf in [False, True]:
        mu, L = fit_vi(X, y, mean_field=mf)
        mu, L = np.asarray(mu), np.asarray(L)
        W = mu + rng.standard_normal((20000, 2)) @ L.T
        cov = L @ L.T
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        name = "mean-field Gaussian q" if mf else "full-covariance Gaussian q"
        print(f"  {name:>26}: estimate {np.mean(1 / (1 + np.exp(-W @ x_star))):.4f}, "
              f"q mean {np.round(mu, 3)}, q std {np.round(np.sqrt(np.diag(cov)), 3)}, correlation {corr:+.2f}")
    # reference posterior moments from the grid
    a = np.linspace(-6, 6, 400)
    G = np.stack(np.meshgrid(a, a), -1).reshape(-1, 2)
    lp = np.asarray(jax.vmap(lambda w: log_p_tilde(w, X, y))(jnp.asarray(G)))
    p = np.exp(lp - lp.max())
    p /= p.sum()
    m = p @ G
    C = (G - m).T @ ((G - m) * p[:, None])
    print(f"  {'true posterior (grid)':>26}: mean {np.round(m, 3)}, std {np.round(np.sqrt(np.diag(C)), 3)}, "
          f"correlation {C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):+.2f}")


if __name__ == "__main__":
    demo()
