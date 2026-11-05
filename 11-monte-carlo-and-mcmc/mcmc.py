"""Markov chain Monte Carlo (CPSC 440 lecture 18).

Build a Markov chain whose stationary distribution is the target p, run it, and
use the (correlated) states as Monte Carlo samples after a burn-in.

Metropolis-Hastings, for any proposal q(x' | x):
    propose x' ~ q(. | x)
    accept with probability min(1, p_tilde(x') q(x | x') / (p_tilde(x) q(x' | x)))
    otherwise stay at x (and count x again).
Only the unnormalized p_tilde is needed. Detailed balance makes p stationary.
Random-walk Metropolis is the special case of a symmetric Gaussian proposal.

Gibbs sampling: update one variable at a time from its full conditional
p(x_j | x_{-j}). It is MH with acceptance probability 1. It mixes slowly when
variables are strongly correlated.

Run:  python 11-monte-carlo-and-mcmc/mcmc.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monte_carlo import grid_truth, log_post, make_problem  # noqa: E402


def metropolis_hastings(log_p, x0, n, step, rng):
    """Random-walk Metropolis. ``log_p`` takes a batch (1, d) and returns (1,)."""
    x = np.array(x0, float)
    lp = log_p(x[None])[0]
    out = np.empty((n, len(x)))
    acc = 0
    for t in range(n):
        prop = x + step * rng.standard_normal(len(x))
        lp_prop = log_p(prop[None])[0]
        if np.log(rng.uniform()) < lp_prop - lp:
            x, lp = prop, lp_prop
            acc += 1
        out[t] = x
    return out, acc / n


def gibbs_gaussian(mu, Sigma, n, rng, x0=None):
    """Gibbs sampling for a 2-D Gaussian, using the closed-form conditionals."""
    x = np.zeros(2) if x0 is None else np.array(x0, float)
    out = np.empty((n, 2))
    for t in range(n):
        for j in range(2):
            k = 1 - j
            m = mu[j] + Sigma[j, k] / Sigma[k, k] * (x[k] - mu[k])
            v = Sigma[j, j] - Sigma[j, k] ** 2 / Sigma[k, k]
            x[j] = m + np.sqrt(v) * rng.standard_normal()
        out[t] = x
    return out


def autocorr_time(x, max_lag=500):
    """Integrated autocorrelation time: how many MCMC steps one independent sample is worth."""
    x = x - x.mean()
    v = x @ x / len(x)
    tau = 1.0
    for lag in range(1, max_lag):
        rho = (x[:-lag] @ x[lag:]) / (len(x) * v)
        if rho < 0.05:
            break
        tau += 2 * rho
    return tau


def demo():
    rng = np.random.default_rng(0)
    X, y = make_problem()
    x_star = np.array([1.0, 1.0])
    truth = grid_truth(X, y, x_star)
    print(f"Bayesian logistic regression posterior predictive (grid truth {truth:.4f}) by random-walk Metropolis:")
    for step in [0.05, 0.5, 5.0]:
        S, acc = metropolis_hastings(lambda W: log_post(W, X, y), [0, 0], 40000, step, rng)
        S = S[5000:]                                   # burn-in
        tau = autocorr_time(S[:, 1])
        print(f"  step {step:4}: acceptance {acc:.2f}, autocorrelation time {tau:6.1f}, "
              f"estimate {expit(S @ x_star).mean():.4f}")

    print("Gibbs sampling a 2-D Gaussian: correlation slows mixing")
    for rho in [0.0, 0.9, 0.99]:
        Sig = np.array([[1, rho], [rho, 1]])
        S = gibbs_gaussian(np.zeros(2), Sig, 20000, rng)[1000:]
        print(f"  rho={rho:4}: sample correlation {np.corrcoef(S.T)[0, 1]:.3f}, "
              f"autocorrelation time {autocorr_time(S[:, 0]):6.1f}")


if __name__ == "__main__":
    demo()
