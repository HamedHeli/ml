# 08 Gaussians, Gaussian discriminant analysis and Bayesian linear regression

**CPSC 440 lectures 6, 7, 8, 9** · [back to the graph](../README.md)

**What is learned:** a (multivariate) Gaussian density, used three ways: as a class-conditional model for
classification (GDA), as a noise model for regression, and as a *posterior over the weights* that gives
predictions with error bars. Hyper-parameters are learned from the marginal likelihood (empirical Bayes).

## Lectures

| Slides | Covers |
| --- | --- |
| `6-gaussians.pdf` | continuous density estimation, the univariate Gaussian, CDF sampling, generative vs. discriminative regression, Bayesian learning, posterior predictive, Bayesian linear regression intro |
| `7-mvn.pdf` | product of Gaussians, multivariate Gaussians, covariance structures, affine transformations, marginals and conditionals, MLE for mean and covariance, MAP and trace regularization, gradient of log-det (bonus) |
| `8-learning-gauss.pdf` | conjugate priors for the covariance, Gaussian discriminant analysis, QDA vs. LDA, LDA and nearest neighbour, generative regression, Bayesian linear regression, Gaussian processes (digression) |
| `9-empbayes.pdf` | setting hyper-parameters, marginal likelihood, empirical Bayes for λ, σ, degree and basis, Bayesian Occam's razor, automatic statistician, Bayesian feature selection and Bayes factors (bonus) |

## The models

### The multivariate Gaussian

`N(x | μ, Σ) ∝ |Σ|^{-1/2} exp(−½(x − μ)ᵀΣ⁻¹(x − μ))`. Everything follows from three closure properties:

- **Affine maps**: `Ax + b ~ N(Aμ + b, AΣAᵀ)`. Sampling is `x = μ + Lz` with `LLᵀ = Σ` (Cholesky), `z ~ N(0, I)`.
- **Marginals**: drop the rows and columns.
- **Conditionals**: `x_A | x_B ~ N(μ_A + Σ_AB Σ_BB⁻¹(x_B − μ_B), Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA)`. Zeros in the *precision*
  `Σ⁻¹` are conditional independences (the bridge to [14](../14-graphical-models/)).

MLE: the sample mean and `S = (1/n)Σ(x − μ)(x − μ)ᵀ`, singular if `n ≤ d`. Penalizing `Tr(Σ⁻¹)` (trace
regularization, a MAP-style fix) gives `S + (λ/n)I`, always invertible.

### Gaussian discriminant analysis (GDA)

A generative classifier with `x | y = c ~ N(μ_c, Σ_c)`. With one covariance per class the decision boundaries are
quadratic (**QDA**); with a shared covariance they are linear (**LDA**); with `Σ = I` LDA reduces to "nearest class
mean." GDA is the Gaussian analogue of naive Bayes, but it models correlations between features.

### Bayesian linear regression

Model `y | x, w ~ N(wᵀφ(x), σ²)`, prior `w ~ N(0, I/λ)`. The prior is conjugate, so the posterior is Gaussian:

`Σ_post = (ΦᵀΦ/σ² + λI)⁻¹`, `μ_post = Σ_post Φᵀy/σ²` (the ridge solution with penalty `λσ²`).

The **posterior predictive** `N(μ_postᵀφ*, σ² + φ*ᵀΣ_post φ*)` has two variances: the irreducible noise `σ²` and the
model's uncertainty about `w`, which grows away from the training data. Bayesian learning *averages* over `w` instead
of picking one; MAP picks the mode.

### Empirical Bayes (type-II maximum likelihood)

Choose hyper-parameters by maximizing the **marginal likelihood** `p(y | X, λ, σ) = N(y | 0, σ²I + ΦΦᵀ/λ)`: integrate
out `w`, then optimize what is left. It uses all the data and no validation set, and it automatically penalizes
complexity: a flexible model spreads its prior mass over many datasets, so it assigns less to the one observed
(**Bayesian Occam's razor**). The same score can compare bases or polynomial degrees, and with a per-feature `λ_j`
it does sparse feature selection (automatic relevance determination).

## Code in this folder

| File | What it implements |
| --- | --- |
| [`gaussians.py`](gaussians.py) | log-density via Cholesky, sampling, conditioning, MLE with trace regularization, QDA and LDA |
| [`bayesian_linear_regression.py`](bayesian_linear_regression.py) | posterior and posterior predictive, log marginal likelihood via Cholesky, empirical Bayes over (λ, σ²) and over the polynomial degree |

```bash
python 08-gaussians-and-bayesian-learning/gaussians.py
python 08-gaussians-and-bayesian-learning/bayesian_linear_regression.py
```

| Result (saved run) | |
| --- | --- |
| conditioning formula vs. Monte Carlo (200k samples, keep those with `x₃ ≈ 2`) | means (0.828, −1.261) vs. (0.869, −1.257) |
| QDA on digits, λ = 1 / 100 | test error 0.025 / **0.008** |
| LDA on digits, λ = 1 / 100 | test error 0.057 / 0.057 |
| empirical Bayes over polynomial degree 0–10 on 25 points of `sin 2x + 0.3x` | picks degree 5 (log marginal likelihood −13.2); test MSE 0.011, near the best of all degrees |
| predictive at x = 0 / 1.9 / 3.0 (training data in [−2, 2]) | 0.14 ± 0.39 / −0.04 ± 0.70 / 12.7 ± 13.1 |

QDA with regularized covariances reaches 0.8% test error, the lowest digit error in this repo (on its own random
split, so compare loosely): a full covariance per class captures how strokes co-vary, and with 64 features and ~120
examples per class the regularization is essential. The predictive at `x = 3` is far off, but the model says so: its error bar is 30 times wider than inside
the data.

## Knowledge gap

**What the slides leave out**

- **Numerical practice**: always work with Cholesky factors and `log det = 2 Σ log L_ii`; never invert `Σ`.
- **Conjugate priors for the covariance** (normal–inverse-Wishart) are named but the full posterior and its Student-t
  predictive are not derived.
- **Gaussian processes** are only a digression: kernel ridge regression from [06](../06-linear-classifiers-and-kernels/)
  plus the Bayesian linear regression here, in function space, with kernel hyper-parameters set by exactly this
  empirical Bayes. They are the natural next step.
- **When empirical Bayes fails**: it can overfit with many hyper-parameters, and it double-uses the data; fully
  Bayesian hierarchical models put a prior on λ instead.
- **Bayesian logistic regression** is mentioned as non-conjugate; the methods to handle it are in
  [11](../11-monte-carlo-and-mcmc/) and [12](../12-variational-inference-and-vaes/).
- Bayes factors and the ASA statement on p-values are bonus material; the relation between marginal likelihood and
  BIC (a Laplace approximation of it) is not shown.

**Prerequisites to shore up**

- Completing the square in matrix form; the Schur complement; block matrix inversion.
- Matrix calculus with `log|Σ|` and `Tr(SΣ⁻¹)`.

**Open questions / further study**

- Calibration: is a 95% Bayesian predictive interval right 95% of the time when the model is misspecified?
- Reading: Bishop, *PRML* ch. 2–3; Rasmussen & Williams, *Gaussian Processes for Machine Learning* ch. 2 and 5.

## Connections

- ← [07 Density estimation](../07-density-estimation-mle-map/): the continuous analogue of Beta–Bernoulli conjugacy.
- ← [04 Linear regression](../04-linear-regression/) and [05 Regularization](../05-feature-selection-and-regularization/):
  the posterior mean is ridge; the marginal likelihood replaces cross-validation for λ.
- → [10 Mixtures](../10-mixture-models-and-em/): a Gaussian mixture is GDA with the labels hidden.
- → [11](../11-monte-carlo-and-mcmc/), [12](../12-variational-inference-and-vaes/): the Laplace approximation and
  Gaussian variational families are this folder applied to non-conjugate posteriors.
- → [14 Graphical models](../14-graphical-models/): Gaussian UGMs and the graphical lasso read structure from `Σ⁻¹`.
