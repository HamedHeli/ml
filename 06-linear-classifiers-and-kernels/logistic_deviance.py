"""Logistic regression's loss, cross-entropy, deviance and KL divergence, checked numerically.

Companion to logistic-loss-deviance-kl.md. Every identity in that note is tested here:

    binary cross-entropy (y in {0,1})  ==  logistic loss log(1 + exp(-y w^T x)) (y in {-1,+1})
    average NLL                        ==  cross-entropy H(P_emp, Q)
    cross-entropy H(P, Q)              ==  H(P) + KL(P || Q)     (soft labels: H(P) > 0)
    deviance D = 2 (l_sat - l_fit)     ==  2 * sum_i KL(saturated_i || fitted_i)
        ungrouped 0/1 data:  l_sat = 0, so D = 2 * NLL
        grouped binomial:    l_sat != 0, same beta-hat, different D
        Poisson:             D = 2 sum KL(Pois(y_i) || Pois(mu_i)), and l_sat != -sum H(Pois(y_i))
    D(small) - D(big) for nested models  ~  chi^2_(extra params)   (Wilks), checked by simulation
    a Gaussian log-likelihood can be positive (densities can exceed 1)

Run:  python 06-linear-classifiers-and-kernels/logistic_deviance.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, xlogy
from scipy.stats import chi2, poisson


# --- fitting: Newton's method (= IRLS) for the two GLMs ----------------------------------
def fit_logistic(X, y, m=None, iters=50):
    """MLE of P(y=1 | x) = sigmoid(x^T beta). y are counts of successes out of m trials (m=1: 0/1 labels)."""
    m = np.ones(len(y)) if m is None else m
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ beta)
        g = X.T @ (y - m * p)                       # gradient of the log-likelihood
        H = X.T @ (X * (m * p * (1 - p))[:, None])  # Fisher information
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-12:
            break
    return beta


def fit_poisson(X, y, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(X @ beta)
        step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-12:
            break
    return beta


# --- losses --------------------------------------------------------------------------------
def bce(y01, p):
    """Binary cross-entropy summed over examples: -sum [y log p + (1-y) log(1-p)], with 0 log 0 = 0."""
    return -np.sum(xlogy(y01, p) + xlogy(1 - y01, 1 - p))


def logistic_loss_pm1(beta, X, ypm):
    """The repo's form (06/linear_classifiers.py): sum log(1 + exp(-y x^T beta)), y in {-1,+1}."""
    return np.sum(np.logaddexp(0, -ypm * (X @ beta)))


def entropy_bern(q):
    return -(xlogy(q, q) + xlogy(1 - q, 1 - q))


def kl_bern(q, p):
    """KL(Bern(q) || Bern(p)), with 0 log 0 = 0."""
    return xlogy(q, q) - xlogy(q, p) + xlogy(1 - q, 1 - q) - xlogy(1 - q, 1 - p)


def kl_poisson(a, b):
    """KL(Pois(a) || Pois(b)) = a log(a/b) - a + b."""
    return xlogy(a, a) - xlogy(a, b) - a + b


def binom_loglik(y, m, p):
    """Binomial log-likelihood, keeping the log C(m, y) term."""
    logC = gammaln(m + 1) - gammaln(y + 1) - gammaln(m - y + 1)
    return np.sum(logC + xlogy(y, p) + xlogy(m - y, 1 - p))


# --- data ----------------------------------------------------------------------------------
def make_binary(n=400, rng=None):
    """One categorical feature with 5 levels (so the data can be grouped) plus an intercept."""
    rng = np.random.default_rng(0) if rng is None else rng
    level = rng.integers(0, 5, n)
    X = np.column_stack([np.ones(n), level - 2.0])
    y = rng.binomial(1, expit(X @ np.array([-0.3, 0.8])))
    return X, y, level


def checks():
    rng = np.random.default_rng(0)
    X, y, level = make_binary(rng=rng)
    n = len(y)
    beta = fit_logistic(X, y)
    p = expit(X @ beta)
    out = {}

    # 1. {0,1} cross-entropy == {-1,+1} logistic loss
    out["bce"] = bce(y, p)
    out["logistic_pm1"] = logistic_loss_pm1(beta, X, 2 * y - 1)

    # 2. average NLL == cross-entropy of the per-example one-hot labels with the model; H(one-hot) = 0
    out["avg_nll"] = out["bce"] / n
    out["H_onehot"] = entropy_bern(y.astype(float)).sum()

    # 3. soft labels: H(P,Q) = H(P) + KL(P||Q) and the floor is H(P) > 0
    q_soft = np.clip(y, 0.1, 0.9)                                     # label smoothing
    out["ce_soft"] = bce(q_soft, p)
    out["H_soft_plus_KL"] = entropy_bern(q_soft).sum() + kl_bern(q_soft, p).sum()
    out["ce_soft_at_perfect_model"] = bce(q_soft, q_soft)              # = H(P), not 0
    out["H_soft"] = entropy_bern(q_soft).sum()

    # 4. ungrouped deviance: l_sat = 0 so D = 2 * NLL = 2 * sum KL(onehot || Bern(p))
    out["dev_ungrouped"] = 2 * out["bce"]
    out["2_sum_KL_ungrouped"] = 2 * kl_bern(y.astype(float), p).sum()

    # 5. grouped binomial: same beta-hat, but l_sat != 0 and D is 2 sum m_g KL(Bern(ybar_g) || Bern(p_g))
    levels = np.unique(level)
    Xg = np.column_stack([np.ones(len(levels)), levels - 2.0])
    mg = np.array([(level == k).sum() for k in levels], float)
    yg = np.array([y[level == k].sum() for k in levels], float)
    beta_g = fit_logistic(Xg, yg, mg)
    pg = expit(Xg @ beta_g)
    ybar = yg / mg
    l_sat_g, l_fit_g = binom_loglik(yg, mg, ybar), binom_loglik(yg, mg, pg)
    out["beta_ungrouped"], out["beta_grouped"] = beta, beta_g
    out["l_sat_grouped"] = l_sat_g
    out["dev_grouped"] = 2 * (l_sat_g - l_fit_g)
    out["2_sum_mKL_grouped"] = 2 * np.sum(mg * kl_bern(ybar, pg))
    out["dev_grouped_df"] = len(levels) - X.shape[1]

    # 6. Poisson: D = 2 sum KL(Pois(y) || Pois(mu)); l_sat is NOT minus the entropy of Pois(y)
    Xp = np.column_stack([np.ones(300), rng.normal(size=300)])
    yp = rng.poisson(np.exp(Xp @ np.array([1.0, 0.5]))).astype(float)
    mu = np.exp(Xp @ fit_poisson(Xp, yp))
    l_sat_p, l_fit_p = poisson.logpmf(yp, yp).sum(), poisson.logpmf(yp, mu).sum()
    out["dev_poisson"] = 2 * (l_sat_p - l_fit_p)
    out["2_sum_KL_poisson"] = 2 * kl_poisson(yp, mu).sum()
    out["l_sat_poisson"] = l_sat_p
    out["minus_sum_H_poisson"] = -sum(poisson.entropy(v) for v in yp)  # entropy(0) = 0

    # 7. Wilks: D(intercept only) - D(with x) ~ chi^2_1 when the true slope is 0
    diffs = []
    for _ in range(2000):
        x = rng.normal(size=150)
        yy = rng.binomial(1, 0.4, 150)
        X1 = np.column_stack([np.ones(150), x])
        b1 = fit_logistic(X1, yy)
        p0 = np.full(150, yy.mean())                               # intercept-only MLE
        diffs.append(2 * (bce(yy, p0) - bce(yy, expit(X1 @ b1))))
    diffs = np.array(diffs)
    out["wilks_rejection_rate_at_5pct"] = np.mean(diffs > chi2.ppf(0.95, 1))

    # 8. ungrouped residual deviance is NOT chi^2_(n-p): compare with its claimed mean n - p
    out["dev_ungrouped_vs_n_minus_p"] = (out["dev_ungrouped"], n - X.shape[1])

    # 9. continuous densities: log-likelihood can be positive
    z = rng.normal(0, 0.01, 100)
    out["gaussian_loglik_sigma_0.01"] = np.sum(-0.5 * np.log(2 * np.pi * 0.01 ** 2) - z ** 2 / (2 * 0.01 ** 2))
    return out


def demo():
    o = checks()
    print("1. {0,1} cross-entropy vs. {-1,+1} logistic loss:  %.6f  %.6f" % (o["bce"], o["logistic_pm1"]))
    print("2. average NLL = %.4f nats; entropy of the one-hot labels = %.1f" % (o["avg_nll"], o["H_onehot"]))
    print("3. soft labels (0.1/0.9): CE = %.4f, H(P) + KL = %.4f; CE at the perfect model = %.4f = H(P) = %.4f"
          % (o["ce_soft"], o["H_soft_plus_KL"], o["ce_soft_at_perfect_model"], o["H_soft"]))
    print("4. ungrouped deviance = 2 NLL = %.4f; 2 sum KL = %.4f" % (o["dev_ungrouped"], o["2_sum_KL_ungrouped"]))
    print("5. grouped by level: beta-hat %s vs. ungrouped %s" % (np.round(o["beta_grouped"], 6),
                                                                 np.round(o["beta_ungrouped"], 6)))
    print("   l_sat = %.3f (not 0); deviance = %.4f on %d df; 2 sum m KL = %.4f"
          % (o["l_sat_grouped"], o["dev_grouped"], o["dev_grouped_df"], o["2_sum_mKL_grouped"]))
    print("6. Poisson deviance = %.4f; 2 sum KL(Pois||Pois) = %.4f; l_sat = %.2f but -sum H(Pois(y)) = %.2f"
          % (o["dev_poisson"], o["2_sum_KL_poisson"], o["l_sat_poisson"], o["minus_sum_H_poisson"]))
    print("7. Wilks LR test, true slope 0, 2000 sims: rejected at 5%% in %.3f of them"
          % o["wilks_rejection_rate_at_5pct"])
    print("8. ungrouped residual deviance %.1f vs. n - p = %d (no chi^2 goodness-of-fit for 0/1 data)"
          % o["dev_ungrouped_vs_n_minus_p"])
    print("9. Gaussian log-likelihood with sigma = 0.01, n = 100: %+.1f (positive)" % o["gaussian_loglik_sigma_0.01"])


if __name__ == "__main__":
    demo()
