"""Fast correctness checks for every implementation in the repo.

Each check compares against a closed form, brute force, finite differences or
scikit-learn. Run from the repo root:  python -m pytest -q
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(folder, name):
    path = os.path.join(ROOT, folder, name + ".py")
    sys.path.insert(0, os.path.join(ROOT, folder))
    spec = importlib.util.spec_from_file_location(f"{folder.replace('-', '_')}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def digits():
    from sklearn.datasets import load_digits
    return load_digits(return_X_y=True)


# 00 ---------------------------------------------------------------------------------
def test_kfold_partitions():
    m = load("00-learning-fundamentals", "cross_validation")
    folds = list(m.kfold_indices(23, 5, np.random.default_rng(0)))
    valid = np.sort(np.concatenate([v for _, v in folds]))
    assert np.array_equal(valid, np.arange(23))
    for tr, va in folds:
        assert not set(tr) & set(va)


# 01 ---------------------------------------------------------------------------------
def test_tree_fits_training_data(digits):
    m = load("01-decision-trees-and-ensembles", "decision_tree")
    X, y = digits
    t = m.DecisionTree(max_depth=np.inf).fit(X[:300], y[:300])
    assert np.mean(t.predict(X[:300]) == y[:300]) == 1.0


def test_infogain_beats_accuracy_when_accuracy_is_stuck():
    m = load("01-decision-trees-and-ensembles", "decision_tree")
    # majority is class 0 on both sides of every split, so accuracy cannot split; infogain can
    X = np.array([[0], [0], [0], [0], [1], [1], [1], [1]], float)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    assert m.fit_stump(X, y, "accuracy").feature is None
    assert m.fit_stump(X, y, "infogain").feature == 0


def test_regression_stump_matches_bruteforce():
    m = load("01-decision-trees-and-ensembles", "decision_tree")
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((40, 3)), rng.standard_normal(40)
    s = m.fit_stump(X, y, "squared")
    best = np.inf
    for j in range(3):
        for t in np.unique(X[:, j])[:-1]:
            L = X[:, j] <= t
            best = min(best, ((y[L] - y[L].mean()) ** 2).sum() + ((y[~L] - y[~L].mean()) ** 2).sum())
    assert np.isclose(((s.predict(X) - y) ** 2).sum(), best)


def test_adaboost_and_gradient_boosting_reduce_training_error():
    m = load("01-decision-trees-and-ensembles", "boosting")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 2))
    y = np.where((X ** 2).sum(1) < 0.5, 1, -1)
    assert np.mean(m.AdaBoost(100).fit(X, y).predict(X) != y) < 0.05
    gb = m.GradientBoostedTrees(50, 2, 0.3).fit(X, X[:, 0] ** 2)
    assert np.all(np.diff(gb.train_mse) <= 1e-12)


# 02 ---------------------------------------------------------------------------------
def test_naive_bayes_matches_sklearn(digits):
    from sklearn.naive_bayes import BernoulliNB
    m = load("02-naive-bayes-and-knn", "naive_bayes")
    X, y = digits
    X = (X > 7).astype(int)
    ours = m.BernoulliNaiveBayes(1.0).fit(X[:500], y[:500]).predict_proba(X[500:600])
    ref = BernoulliNB(alpha=1.0).fit(X[:500], y[:500]).predict_proba(X[500:600])
    assert np.allclose(ours, ref, atol=1e-8)


def test_knn_matches_sklearn(digits):
    from sklearn.neighbors import KNeighborsClassifier
    m = load("02-naive-bayes-and-knn", "knn")
    X, y = digits
    X = X.astype(float) + 1e-6 * np.random.default_rng(0).standard_normal(X.shape)   # break distance ties
    ours = m.KNN(1).fit(X[:800], y[:800]).predict(X[800:])
    ref = KNeighborsClassifier(1).fit(X[:800], y[:800]).predict(X[800:])
    assert np.mean(ours == ref) == 1.0


# 03 ---------------------------------------------------------------------------------
def test_clustering_against_sklearn():
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.datasets import make_blobs
    m = load("03-clustering-and-outliers", "clustering")
    X, _ = make_blobs(150, centers=3, random_state=0)
    for link in ["single", "complete", "average"]:
        ref = AgglomerativeClustering(3, linkage=link).fit_predict(X)
        assert m.adjusted_rand(m.agglomerative(X, 3, link), ref) == pytest.approx(1.0)
    ref = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    ours = m.dbscan(X, 0.8, 5)
    assert np.array_equal(ours == -1, ref == -1)
    W, labels, cost = m.kmeans(X, 3)
    assert np.isclose(cost, ((X - W[labels]) ** 2).sum())


def test_local_outlier_ratio_flags_local_outlier():
    m = load("03-clustering-and-outliers", "outliers")
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (100, 2)), rng.normal(3, 1, (100, 2)), [[0.6, 0]]])
    assert np.argsort(-m.local_outlierness(X))[:5].tolist().count(200) == 1


# 04-05 ------------------------------------------------------------------------------
def test_least_squares_gd_and_huber():
    m = load("04-linear-regression", "linear_regression")
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 4))
    y = Z @ np.array([1, -2, 0.5, 3]) + 0.1 * rng.standard_normal(50)
    w = m.least_squares(Z, y)
    assert np.allclose(w, np.linalg.lstsq(Z, y, rcond=None)[0])
    L = np.linalg.eigvalsh(Z.T @ Z).max()
    w_gd, _ = m.gradient_descent(lambda v: Z.T @ (Z @ v - y), np.zeros(4), 1 / L, 20000, 1e-10)
    assert np.allclose(w_gd, w, atol=1e-6)
    # Huber with a huge threshold is least squares
    assert np.allclose(m.robust_regression(Z, y, eps=1e6, max_iter=20000), w, atol=1e-5)


def test_lasso_matches_sklearn_and_ridge_closed_form():
    from sklearn.linear_model import Lasso
    m = load("05-feature-selection-and-regularization", "regularization")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 10))
    y = X[:, 0] * 2 - X[:, 3] + 0.1 * rng.standard_normal(80)
    lam = 5.0
    ref = Lasso(alpha=lam / 80, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y).coef_
    assert np.allclose(m.lasso(X, y, lam, max_iter=100000, tol=1e-12), ref, atol=1e-6)
    w = m.ridge(X, y, 3.0)
    assert np.allclose(X.T @ (X @ w - y) + 3.0 * w, 0, atol=1e-8)
    assert set(m.forward_selection(X, y, "bic")) == {0, 3}


# 06 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("fn", ["logistic_loss_grad", "hinge_loss_grad"])
def test_linear_classifier_gradients(fn):
    m = load("06-linear-classifiers-and-kernels", "linear_classifiers")
    rng = np.random.default_rng(0)
    X, y, w = rng.standard_normal((20, 5)), np.sign(rng.standard_normal(20)), rng.standard_normal(5)
    f = getattr(m, fn)
    _, g = f(w, X, y, 0.7)
    num = np.array([(f(w + 1e-6 * e, X, y, 0.7)[0] - f(w - 1e-6 * e, X, y, 0.7)[0]) / 2e-6 for e in np.eye(5)])
    assert np.allclose(g, num, atol=1e-4)


def test_softmax_gradient():
    m = load("06-linear-classifiers-and-kernels", "linear_classifiers")
    rng = np.random.default_rng(1)
    X, y, W = rng.standard_normal((15, 4)), rng.integers(0, 3, 15), rng.standard_normal((4, 3))
    _, G = m.softmax_loss_grad(W, X, y, 0.1)
    E = np.zeros_like(W)
    E[2, 1] = 1e-6
    num = (m.softmax_loss_grad(W + E, X, y, 0.1)[0] - m.softmax_loss_grad(W - E, X, y, 0.1)[0]) / 2e-6
    assert np.isclose(G[2, 1], num, atol=1e-5)


def test_kernel_trick_equals_explicit_features():
    m = load("06-linear-classifiers-and-kernels", "kernels")
    rng = np.random.default_rng(0)
    X, Xt = rng.standard_normal((10, 3)), rng.standard_normal((4, 3))
    assert np.allclose(m.explicit_poly2(X) @ m.explicit_poly2(Xt).T, m.poly_kernel(X, Xt, 2))


# 07 ---------------------------------------------------------------------------------
def test_beta_bernoulli_and_dirichlet():
    from scipy.special import betaln
    m = load("07-density-estimation-mle-map", "density_estimation")
    bb = m.BetaBernoulli(2, 3).fit(np.array([1, 0, 0, 1, 1]))
    assert bb.mle() == 0.6 and np.isclose(bb.map(), 4 / 8) and np.isclose(bb.posterior_predictive(), 5 / 10)
    # marginal likelihood = sum over the sequence, check against brute force integration
    th = np.linspace(1e-6, 1 - 1e-6, 200001)
    integrand = th ** 3 * (1 - th) ** 2 * th * (1 - th) ** 2 / np.exp(betaln(2, 3))
    assert np.isclose(np.exp(bb.log_marginal_likelihood()), np.trapezoid(integrand, th), rtol=1e-4)
    x = np.array([0, 0, 1, 2, 2, 2])
    assert np.isclose(m.dirichlet_log_marginal(x, np.ones(3)) - m.dirichlet_log_marginal(x[:5], np.ones(3)),
                      np.log((2 + 1) / (5 + 3)))      # sequential posterior predictive of the last "2"
    s = m.sample_categorical(np.array([0.2, 0.5, 0.3]), 200000, np.random.default_rng(0))
    assert np.allclose(np.bincount(s) / len(s), [0.2, 0.5, 0.3], atol=0.01)


def test_exponential_family_moment_matching():
    m = load("07-density-estimation-mle-map", "exponential_family")
    x = np.random.default_rng(0).poisson(2.0, 300).astype(float)
    w, w_closed = m.mle_by_gradient("poisson", x)
    assert np.isclose(w, w_closed) and np.isclose(np.exp(w), x.mean())


# 08 ---------------------------------------------------------------------------------
def test_gaussian_conditioning_and_logpdf():
    from scipy.stats import multivariate_normal
    m = load("08-gaussians-and-bayesian-learning", "gaussians")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    S, mu = A @ A.T + np.eye(3), rng.standard_normal(3)
    X = rng.standard_normal((5, 3))
    assert np.allclose(m.mvn_logpdf(X, mu, S), multivariate_normal(mu, S).logpdf(X))
    cm, cS = m.condition(mu, S, np.array([2]), np.array([0.7]))
    P = np.linalg.inv(S)                                  # conditional precision is the precision sub-block
    assert np.allclose(np.linalg.inv(cS), P[:2, :2])
    assert np.allclose(cm, mu[:2] - np.linalg.solve(P[:2, :2], P[:2, 2:] @ (np.array([0.7]) - mu[2:])))


def test_bayesian_linear_regression_mean_is_ridge():
    m = load("08-gaussians-and-bayesian-learning", "bayesian_linear_regression")
    rng = np.random.default_rng(0)
    Phi, y = rng.standard_normal((30, 4)), rng.standard_normal(30)
    blr = m.BayesianLinearRegression(lam=2.0, sigma2=0.5).fit(Phi, y)
    ridge = np.linalg.solve(Phi.T @ Phi + 2.0 * 0.5 * np.eye(4), Phi.T @ y)
    assert np.allclose(blr.mu, ridge)
    from scipy.stats import multivariate_normal
    C = 0.5 * np.eye(30) + Phi @ Phi.T / 2.0
    assert np.isclose(m.log_marginal_likelihood(Phi, y, 2.0, 0.5), multivariate_normal(np.zeros(30), C).logpdf(y))


# 09 ---------------------------------------------------------------------------------
def test_pca_and_mds(digits):
    from sklearn.decomposition import PCA as SKPCA
    m = load("09-latent-factor-models", "latent_factors")
    X, _ = digits
    p = m.PCA(5).fit(X)
    assert m.subspace_angle(p.W, SKPCA(5).fit(X).components_) < 1e-4
    P = np.random.default_rng(0).standard_normal((20, 2))
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    E = m.classical_mds(D, 2)
    assert np.allclose(np.sqrt(((E[:, None] - E[None]) ** 2).sum(-1)), D, atol=1e-8)


# 10 ---------------------------------------------------------------------------------
def test_em_is_monotone_and_kde_normalizes():
    m = load("10-mixture-models-and-em", "mixtures")
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (100, 1)), rng.normal(2, 1, (100, 1))])
    g = m.GaussianMixture(2).fit(X)
    assert np.all(np.diff(g.history) > -1e-8)
    grid = np.linspace(-10, 10, 4001)[:, None]
    assert np.isclose(np.trapezoid(np.exp(m.kde_logpdf(grid, X, 0.3)), grid[:, 0]), 1.0, atol=1e-4)


# 11 ---------------------------------------------------------------------------------
def test_mcmc_and_importance_sampling_agree_with_grid():
    mc = load("11-monte-carlo-and-mcmc", "monte_carlo")
    mm = load("11-monte-carlo-and-mcmc", "mcmc")
    X, y = mc.make_problem()
    x_star = np.array([1.0, 1.0])
    truth = mc.grid_truth(X, y, x_star)
    rng = np.random.default_rng(0)
    w_map, cov = mc.laplace(X, y)
    est, _ = mc.importance(X, y, x_star, 20000, rng, w_map, 1.5 * cov)
    assert abs(est - truth) < 0.01
    S, _ = mm.metropolis_hastings(lambda W: mc.log_post(W, X, y), [0, 0], 20000, 0.5, rng)
    assert abs(np.mean(1 / (1 + np.exp(-S[2000:] @ x_star))) - truth) < 0.02
    G = mm.gibbs_gaussian(np.zeros(2), np.array([[1, 0.8], [0.8, 1]]), 20000, rng)
    assert abs(np.corrcoef(G[500:].T)[0, 1] - 0.8) < 0.03


# 12 ---------------------------------------------------------------------------------
def test_variational_inference_matches_posterior_mean():
    import jax.numpy as jnp
    vi = load("12-variational-inference-and-vaes", "variational_inference")
    mc = load("11-monte-carlo-and-mcmc", "monte_carlo")
    X, y = mc.make_problem()
    mu, L = vi.fit_vi(jnp.asarray(X), jnp.asarray(y, dtype=jnp.float32), steps=1500)
    a = np.linspace(-6, 6, 400)
    G = np.stack(np.meshgrid(a, a), -1).reshape(-1, 2)
    lp = mc.log_post(G, X, y)
    p = np.exp(lp - lp.max())
    post_mean = (p / p.sum()) @ G
    assert np.allclose(np.asarray(mu), post_mean, atol=0.05)


def test_vae_kl_and_iwae_bound():
    import jax
    import jax.numpy as jnp
    v = load("12-variational-inference-and-vaes", "vae")
    params = v.make_params(jax.random.PRNGKey(0), 64, 3)
    x = jnp.asarray((np.random.default_rng(0).uniform(size=(20, 64)) > 0.5).astype(np.float32))
    rec, kl = v.vae_terms(params, x, jax.random.PRNGKey(1))
    assert np.all(np.asarray(kl) >= 0)
    elbos = []
    for s in range(50):
        r, k = v.vae_terms(params, x, jax.random.PRNGKey(s))
        elbos.append(float(jnp.mean(r - k)))
    elbo_mc = np.mean(elbos)
    # the importance-weighted bound is at least as tight as the ELBO (up to Monte Carlo noise)
    assert float(v.iwae_bound(params, x, jax.random.PRNGKey(2), K=500)) >= elbo_mc - 0.5


# 13 ---------------------------------------------------------------------------------
def test_forward_backward_and_viterbi_vs_bruteforce():
    m = load("13-markov-chains-and-hmms", "hmm")
    rng = np.random.default_rng(0)
    pi, A, B = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3), 3), rng.dirichlet(np.ones(4), 3)
    x = rng.integers(0, 4, 6)
    lp = {z: np.log(pi[z[0]]) + sum(np.log(A[a, b]) for a, b in zip(z[:-1], z[1:])) + sum(np.log(B[s, o]) for s, o in zip(z, x))
          for z in itertools.product(range(3), repeat=6)}
    lse = np.logaddexp.reduce(list(lp.values()))
    g, _, ll = m.forward_backward(x, pi, A, B)
    assert np.isclose(ll, lse)
    marg = np.zeros((6, 3))
    for z, v in lp.items():
        marg[np.arange(6), list(z)] += np.exp(v - lse)
    assert np.allclose(g, marg)
    assert tuple(m.viterbi(x, pi, A, B)[0]) == max(lp, key=lp.get)


def test_stationary_distribution():
    m = load("13-markov-chains-and-hmms", "markov_chain")
    A = np.random.default_rng(0).dirichlet(np.ones(4), 4)
    p = m.stationary(A)
    assert np.allclose(p @ A, p) and np.isclose(p.sum(), 1)


# 14 ---------------------------------------------------------------------------------
def test_d_separation():
    m = load("14-graphical-models", "dag")
    g = m.ALARM
    assert g.d_separated("burglary", "earthquake")
    assert not g.d_separated("burglary", "earthquake", ["mary_calls"])     # descendant of a collider
    assert g.d_separated("john_calls", "mary_calls", ["alarm"])
    assert not g.d_separated("john_calls", "burglary")


def test_graphical_lasso_is_optimal():
    m = load("14-graphical-models", "ugm")
    rng = np.random.default_rng(0)
    S = np.cov(rng.standard_normal((200, 5)).T)
    Z = m.graphical_lasso(S, 0.1, iters=3000)
    # KKT: Theta^{-1} - S = lam * sign(Theta) on nonzero off-diagonals, |.| <= lam on zeros
    G = np.linalg.inv(Z) - S
    off = ~np.eye(5, dtype=bool)
    nz = off & (np.abs(Z) > 1e-6)
    assert np.allclose(G[nz], 0.1 * np.sign(Z[nz]), atol=1e-3)
    assert np.all(np.abs(G[off & ~nz]) <= 0.1 + 1e-3)


# 15-16 ------------------------------------------------------------------------------
def test_backprop_and_transposed_convolution(digits):
    nn = load("15-neural-networks", "mlp")
    X, y = digits
    assert nn.grad_check(nn.MLP([64, 16, 16, 10], activation="sigmoid"), X[:30] / 16, y[:30]) < 1e-5
    assert nn.grad_check(nn.MLP([64, 16, 16, 16, 10], residual=True), X[:30] / 16, y[:30]) < 1e-4
    cv = load("16-convolutional-networks", "convolution")
    rng = np.random.default_rng(0)
    F, a, z = rng.standard_normal((3, 3, 2, 3)), rng.standard_normal((7, 7, 2)), rng.standard_normal((3, 3, 3))
    assert np.isclose(np.sum(cv.conv2d(a, F, 2) * z), np.sum(a * cv.conv2d_transpose(z, F, (7, 7), 2)))
    from scipy.signal import correlate2d
    img = rng.standard_normal((6, 6, 1))
    k = rng.standard_normal((3, 3))
    assert np.allclose(cv.conv2d(img, k[:, :, None, None])[..., 0], correlate2d(img[..., 0], k, "valid"))


# 17 ---------------------------------------------------------------------------------
def test_attention_is_causal():
    import jax
    tf = load("17-sequence-models-and-transformers", "transformer")
    p = tf.init(jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
    out1, A = tf.mha(p["blocks"][0], x, 2)
    x2 = x.at[:, 6:].set(0.0)                       # change only the future
    out2, _ = tf.mha(p["blocks"][0], x2, 2)
    assert np.allclose(out1[:, :6], out2[:, :6], atol=1e-5)
    assert np.allclose(np.asarray(A).sum(-1), 1.0, atol=1e-5)


def test_bpe_and_beam_search():
    m = load("17-sequence-models-and-transformers", "tokens_and_decoding")
    merges = m.learn_bpe("aaab aaab aaab", 2)
    assert merges[0] == ("a", "a")
    assert "".join(m.bpe_encode("aaab", merges)) == "aaab</w>"
    rng = np.random.default_rng(0)
    P0, A = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3), 3)
    ln = lambda s: np.log(P0) if not s else np.log(A[s[-1]])
    best = max(itertools.product(range(3), repeat=4),
               key=lambda s: np.log(P0[s[0]]) + sum(np.log(A[a, b]) for a, b in zip(s[:-1], s[1:])))
    assert m.beam_search(ln, 4, 3 ** 4, 3)[0] == best


# 18 ---------------------------------------------------------------------------------
def test_diffusion_posterior_coefficients():
    d = load("18-diffusion-and-image-generation", "diffusion")
    rng = np.random.default_rng(0)
    t = 40
    x0 = 1.3
    # sample x_{t-1} ~ q(. | x0), x_t ~ q(. | x_{t-1}); regress x_{t-1} on (x_t, x0 fixed)
    xprev = np.sqrt(d.ABAR[t - 1]) * x0 + np.sqrt(1 - d.ABAR[t - 1]) * rng.standard_normal(400000)
    xt = np.sqrt(d.ALPHAS[t]) * xprev + np.sqrt(d.BETAS[t]) * rng.standard_normal(400000)
    slope = np.cov(xprev, xt)[0, 1] / np.var(xt)
    assert np.isclose(slope, d.GAMMA[t], atol=0.01)
    resid = xprev - slope * xt
    assert np.isclose(resid.mean(), d.DELTA[t] * x0, atol=0.01)
    assert np.isclose(resid.var(), d.SIG2[t], rtol=0.02)


def test_edm_preconditioning_and_solvers():
    e = load("18-diffusion-and-image-generation", "edm")
    import jax.numpy as jnp
    sig = jnp.array([0.01, 1.0, 80.0])
    c_skip, c_out, c_in, _ = e.precond(sig)
    # the effective training target (y - c_skip x)/c_out has unit variance when y ~ N(0, sd^2), x = y + sigma n
    sd = e.SIGMA_DATA
    var_target = ((1 - c_skip) ** 2 * sd ** 2 + c_skip ** 2 * sig ** 2) / c_out ** 2
    assert np.allclose(var_target, 1, atol=1e-5)
    assert np.allclose(c_in ** 2 * (sd ** 2 + sig ** 2), 1)
    s = e.sigma_steps(18)
    assert np.isclose(s[0], 80) and np.isclose(s[-2], 0.002) and s[-1] == 0 and np.all(np.diff(s) < 0)
    # Gaussian data: the exact denoiser is linear and the ODE maps N(0, sigma_max^2 + sd^2) to N(0, sd^2)
    den = lambda x, sg: x * sd ** 2 / (sd ** 2 + sg ** 2)
    x0 = np.array([[80.0, -40.0]])
    exact = x0 * sd / np.sqrt(80.0 ** 2 + sd ** 2)
    err_e = abs(e.ode_sample(den, x0, 35, "euler")[0] - exact).max()
    err_h = abs(e.ode_sample(den, x0, 18, "heun")[0] - exact).max()
    assert err_h < 0.7 * err_e and err_e < 0.2   # at 35 NFE Heun beats Euler (0.045 vs 0.073)


def test_prompt_to_image_pipeline():
    p = load("18-diffusion-and-image-generation", "prompt_to_image")
    assert p.text_condition("give me the picture of a dog") == "dog" and p.text_condition("") is None
    imgs, labels, _ = p.make_dataset(n_per_class=40)
    ae = p.LinearAutoencoder(imgs)
    Z = ae.encode(imgs)
    assert np.isclose(Z.std(), 1.0) and ae.decode(Z).shape == imgs.shape
    den = p.make_denoiser(Z, labels)
    for seed in range(3):
        z, hist = p.generate(den, "a cat", seed, n_steps=20)
        assert labels[np.argmin(((Z - z[0]) ** 2).sum(1))] == "cat" and len(hist) == 20


# 19 ---------------------------------------------------------------------------------
def test_info_nce_at_init_is_near_log_n():
    import jax
    import jax.numpy as jnp
    c = load("19-representation-learning-and-llms", "contrastive")
    params = c.init(jax.random.PRNGKey(0))
    x = jnp.asarray(np.random.default_rng(0).uniform(size=(128, 64)).astype(np.float32))
    loss = float(c.info_nce(params, x, jax.random.PRNGKey(1)))
    assert 0.5 * np.log(128) < loss < 1.5 * np.log(128)


def test_worked_example_numbers():
    w = load("17-sequence-models-and-transformers", "worked_example")
    out = w.ATTN_TREE @ w.VALUES
    assert np.allclose(out, [1.9, 2.3, 2.7, 3.5])
    res = w.X_TREE + out
    assert np.isclose(res.var(), 2.9)
    assert np.allclose(w.layer_norm(res), [-1.292, -0.470, 0.352, 1.409], atol=1e-3)
    rng = np.random.default_rng(1)
    X, W1, W2 = rng.normal(size=(7, 4)), rng.normal(size=(16, 4)), rng.normal(size=(4, 16))
    together = w.mlp(X, W1, np.zeros(16), W2, np.zeros(4))
    assert together.shape == (7, 4)
    assert np.allclose(together, np.stack([w.mlp(x, W1, np.zeros(16), W2, np.zeros(4)) for x in X]))
    _, A = w.attention(X, X, X)
    assert np.allclose(np.triu(A, 1), 0) and np.allclose(A.sum(1), 1)


# 20 ---------------------------------------------------------------------------------
def test_leakage_demos():
    lk = load("20-evaluation", "leakage")
    leaky, honest = lk.selection_leakage(seed=1)
    assert leaky > 0.75 and abs(honest - 0.5) < 0.15
    r = lk.target_leakage(seed=1)
    assert r["offline_with_leak"] > r["honest"] > r["deployed_with_leak"] - 0.02
    r = lk.temporal_leakage(n_series=20, seed=1)
    assert r["random_split"] * 3 < r["time_split"]


def test_auc_and_psi():
    from sklearn.metrics import roc_auc_score
    sk = load("20-evaluation", "skew")
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 500), rng.random(500)
    assert np.isclose(sk.auc(y, s), roc_auc_score(y, s))
    x = rng.normal(size=20000)
    assert sk.psi(x, rng.normal(size=20000)) < 0.01 < 0.25 < sk.psi(x, rng.normal(1, 1, 20000))


def test_online_evaluation():
    on = load("20-evaluation", "online")
    n = on.sample_size_per_arm(0.10, 0.01)
    assert 14000 < n < 15500
    assert abs(on.power_by_simulation(reps=2000, seed=1) - 0.8) < 0.04
    r = on.peeking(reps=2000, seed=1)
    assert r["fixed_horizon"] < 0.07 and r["daily_peeking"] > 0.15
    r = on.ope(seed=1)
    assert abs(r["ips"] - r["true_new"]) < 0.01 < abs(r["naive"] - r["true_new"])
