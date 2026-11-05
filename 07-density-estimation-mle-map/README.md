# 07 Density estimation, MLE, MAP and exponential families

**CPSC 440 lectures 1, 2, 3, 4, 10 · CPSC 340 L24** · [back to the graph](../README.md)

**What is learned:** a probability distribution `p(x | θ)` over the data itself. Once you have it you can compute
probabilities, find the most likely values, sample new data, and plug it into a classifier.

## Lectures

| Slides | Covers |
| --- | --- |
| `1-binary-density-est.pdf` | binary density estimation, the Bernoulli, inference tasks (probabilities, mode/decoding, likelihood, sampling) |
| `2-bernoulli-mle-map.pdf` | the likelihood, MLE, problems with MLE, MAP, priors, the Beta distribution, Beta–Bernoulli, hyper-parameters |
| `3-generative.pdf` (first half) | multivariate binary data, product of Bernoullis on MNIST |
| `4-cat-discrim.pdf` (first half) | categorical density estimation, inverse-CDF sampling, categorical MLE (and the derivation that fails), Dirichlet MAP, conjugate priors |
| `10-expfam.pdf` | exponential families, sufficient statistics, MLE as moment matching, convexity, conjugate priors, maximum entropy (bonus) |
| `L24 - MLE & MAP.pdf` (CPSC 340) | MLE and MAP for regression and classification; losses as negative log-likelihoods, regularizers as negative log-priors |

## The models

### Bernoulli and the Beta prior

`p(x = 1 | θ) = θ`. Given `n₁` ones in `n` samples:

- **MLE** `θ̂ = n₁ / n`. After three tails it says heads is impossible, which is the classic MLE failure on small data.
- **MAP** with a `Beta(α, β)` prior: `θ̂ = (n₁ + α − 1) / (n + α + β − 2)`; `α = β = 2` is "add one of each".
- **Bayesian**: the posterior is `Beta(n₁ + α, n₀ + β)` (the prior is **conjugate**), and the posterior predictive
  is `p(x = 1 | data) = (n₁ + α) / (n + α + β)`. As `n` grows all three agree.

### Categorical and the Dirichlet prior

`p(x = c | θ) = θ_c`. The MLE `θ_c = n_c / n` needs the constraint `Σθ_c = 1` (Lagrange multiplier, or an unnormalized
parameterization); the conjugate prior is the **Dirichlet**, giving MAP `(n_c + α_c − 1)/(n + Σα − k)`. Sampling
uses the inverse CDF: draw `u ~ U(0,1)`, return the first `c` with cumulative probability `≥ u` (`O(log k)` with
binary search; alias tables are `O(1)`).

### Product of Bernoullis

For `x ∈ {0,1}^d`, assume all coordinates are **independent**: `p(x) = Π_j θ_j^{x_j}(1 − θ_j)^{1 − x_j}`. Learning
is `d` separate Bernoulli fits. It is a terrible image model (a sample is speckle) but it is the component that
naive Bayes, mixtures of Bernoullis and Markov chains are built from.

### Exponential families

`p(x | w) = h(x) exp(wᵀT(x) − A(w))`, with natural parameters `w`, sufficient statistics `T(x)` and log-partition
`A(w)`. Bernoulli, categorical, Gaussian, Poisson, Beta, Dirichlet, gamma all fit; the uniform with unknown support,
Student-t and mixtures do not.

- `∇A(w) = E[T(X)]` and `∇²A(w) = Cov[T(X)] ⪰ 0`, so the NLL is **convex**, and setting its gradient to zero gives
  **MLE = moment matching**: choose `w` so the model's expected sufficient statistics equal the data average.
- Every exponential family has a conjugate prior.
- Making the natural parameter a linear function of features, `w = θᵀx`, gives a **generalized linear model**:
  Bernoulli → logistic regression, Gaussian → least squares, Poisson → Poisson regression for counts.
- Bonus: the maximum-entropy distribution subject to moment constraints is the exponential family with those
  sufficient statistics.

### MLE and MAP for supervised learning (CPSC 340 L24)

Minimizing a loss is maximizing a likelihood: squared error ↔ Gaussian noise, absolute error ↔ Laplace noise,
logistic loss ↔ Bernoulli. Adding a regularizer is MAP: `λ‖w‖²` ↔ Gaussian prior, `λ‖w‖₁` ↔ Laplace prior.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`density_estimation.py`](density_estimation.py) | Beta–Bernoulli (MLE, MAP, posterior predictive, marginal likelihood), categorical MLE, Dirichlet MAP and marginal likelihood, inverse-CDF sampling, product of Bernoullis |
| [`exponential_family.py`](exponential_family.py) | MLE by Newton's method vs. moment matching for Bernoulli and Poisson; Poisson regression (a GLM) by Newton's method |

```bash
python 07-density-estimation-mle-map/density_estimation.py
python 07-density-estimation-mle-map/exponential_family.py
```

| Result (saved run) | |
| --- | --- |
| three tails in a row: MLE / MAP (Beta(2,2)) / posterior predictive | 0.00 / 0.20 / 0.29 |
| 1000 flips of a 0.3 coin | MLE = MAP = 0.277 (the prior washes out) |
| product of Bernoullis on binarized digits | −25.1 nats per 64-pixel test image; samples look like noise |
| Newton MLE vs. moment matching (Bernoulli, Poisson) | identical; `E[T]` at the MLE equals the sample mean exactly |
| Poisson regression, true θ = (0.5, 0.8, −0.4) | estimate (0.466, 0.841, −0.370); least squares predicts negative rates for 13% of inputs |

The −25.1 nats is the baseline that every later density model of the same digits improves on: a mixture of 30
Bernoullis reaches −19.4 ([10](../10-mixture-models-and-em/)) and a VAE about −16.8 ([12](../12-variational-inference-and-vaes/)).

## Knowledge gap

**What the slides leave out**

- **Why MLE is a good idea**: consistency, asymptotic normality with covariance = inverse Fisher information, and
  efficiency (the Cramér–Rao bound). MLE minimizes `KL(p_true ‖ p_θ)` (a bonus slide in lecture 13) but its sampling
  behaviour is not covered.
- **Uncertainty about θ** beyond conjugate posteriors: confidence intervals, the bootstrap.
- **MAP is not invariant to reparameterization**: the mode of a density changes under a change of variables, while
  the MLE and the full posterior do not. This is a real argument for full Bayesian inference over MAP.
- **Duality** between the natural parameters `w` and the mean parameters `μ = ∇A(w)`, and the convex conjugate `A*`
  (bonus slide only), which underlies variational inference ([12](../12-variational-inference-and-vaes/)).
- **GLM practice**: link functions, over-dispersion (negative binomial when Poisson variance is too small), offsets
  and exposure, deviance.
- Density estimation evaluation: held-out log-likelihood and why bits-per-dimension is used for images.

**Prerequisites to shore up**

- Logs of products, Lagrange multipliers, the gamma and beta functions.
- Convexity of log-sum-exp.

**Open questions / further study**

- How do you set a prior you actually believe? Empirical Bayes ([08](../08-gaussians-and-bayesian-learning/)) learns it
  from data; hierarchical models put priors on priors.
- Reading: Murphy, *PML1* ch. 2–4; Wainwright & Jordan 2008, "Graphical models, exponential families and variational
  inference," ch. 3.

## Connections

- → [02 Naive Bayes](../02-naive-bayes-and-knn/): class-conditional products of Bernoullis; Laplace smoothing is a Beta
  prior.
- → [08 Gaussians](../08-gaussians-and-bayesian-learning/): the continuous version, with Gaussian–Gaussian conjugacy.
- → [10 Mixtures](../10-mixture-models-and-em/): mixtures of exponential families, fit by EM with moment-matching
  M-steps.
- → [13 Markov chains](../13-markov-chains-and-hmms/): a chain is a product of categorical conditionals, learned by
  counting.
- ↔ [06 Linear classifiers](../06-linear-classifiers-and-kernels/): logistic and softmax regression are GLMs.
