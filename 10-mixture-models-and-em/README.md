# 10 Mixture models, expectation maximization and kernel density estimation

**CPSC 440 lecture 11** · [back to the graph](../README.md)

**What is learned:** a density built from simple pieces: `p(x) = Σ_c π_c p(x | θ_c)`, with a hidden "which piece"
variable for every example. The non-parametric version puts one piece on every data point (KDE).

## Lectures

| Slides | Covers |
| --- | --- |
| `11-mixtures.pdf` | mixtures of Gaussians, the latent-variable view, ancestral sampling, responsibilities, imputation vs. EM, k-means as a special case, the MLE that does not exist, mixtures of Bernoullis on digits, EM for exponential families, EM properties, missing data, the ELBO, Dirichlet processes, KDE, violin plots, mean-shift, log-sum-exp (bonus), mixture of experts (bonus) |

## The models

### Mixture of Gaussians and mixture of Bernoullis

**Latent-variable view**: `z ~ Cat(π)`, then `x | z = c ~ p(x | θ_c)`. Sampling is ancestral (draw `z`, then `x`).
Inference is computing the **responsibility** `r_ic = p(z_i = c | x_i) ∝ π_c p(x_i | θ_c)`, done in log space with
log-sum-exp because high-dimensional likelihoods underflow.

- A mixture of Gaussians can approximate any continuous density; a mixture of Bernoullis can represent any
  distribution over binary vectors, and is "unsupervised naive Bayes": it captures dependencies between pixels
  through the cluster variable.

### Expectation maximization

Maximizing `Σ_i log Σ_c π_c p(x_i | θ_c)` directly is awkward (log of a sum). EM alternates:

- **E-step**: compute responsibilities `r_ic` under the current parameters.
- **M-step**: maximize the responsibility-weighted complete-data log-likelihood. For Gaussians:
  `π_c = mean_i r_ic`, `μ_c = Σ r_ic x_i / Σ r_ic`, `Σ_c = Σ r_ic (x_i − μ_c)(x_i − μ_c)ᵀ / Σ r_ic`; for any exponential
  family the M-step is weighted moment matching.

Why it works: for any distribution `q(z)`, `log p(x) ≥ E_q[log p(x, z)] + H(q)` (the **ELBO**), with equality at
`q = p(z | x)`. The E-step makes the bound tight, the M-step pushes it up, so the log-likelihood never decreases. It
finds a local optimum; random restarts help. **Hard** assignments (impute the most likely `z`) give k-means when the
Gaussians are spherical with equal variance.

**The MLE does not exist**: put one component's mean on a data point and shrink its variance, and the likelihood goes
to infinity. A small `εI` added to every covariance (a MAP-style prior) prevents the collapse.

### Kernel density estimation (KDE) and mean-shift

`p(x) = (1/n) Σ_i N(x | x_i, h²I)`: a mixture with one component per data point and no fitting at all. The bandwidth
`h` is the only knob (choose it by held-out likelihood). A smooth histogram, the basis of violin plots; poor in high
dimensions. **Mean-shift** moves every point uphill on the KDE until it reaches a mode; points sharing a mode form a
cluster, so the number of clusters is set by `h` instead of `k`.

## Code in this folder

[`mixtures.py`](mixtures.py): EM for Gaussian mixtures (full covariances, log-sum-exp responsibilities, ε
regularization, monotonicity tracking), EM for mixtures of Bernoullis with Beta / Dirichlet smoothing, KDE log-density,
mean-shift.

```bash
python 10-mixture-models-and-em/mixtures.py
```

| Result (saved run) | |
| --- | --- |
| 3 Gaussian clusters, validation log-likelihood per point, k = 1 / 2 / 3 / 6 | −3.94 / −3.39 / **−3.10** / −3.13 |
| EM log-likelihood monotone in every run | yes |
| learned means for k = 3 (truth (0,0), (4,0), (2,3.5)) | (0.03, 0.07), (4.00, −0.02), (2.04, 3.43) |
| KDE, h = 0.05 / 0.3 / 1 / 3 | −6.70 / **−3.13** / −3.56 / −4.59 |
| mean-shift, h = 0.7 | finds the 3 modes |
| mixture of Bernoullis on binarized digits, k = 1 / 10 / 30 | −25.1 / −20.3 / **−19.4** nats per image |

A tuned KDE (−3.13) nearly matches the correctly specified 3-component mixture (−3.10) in 2-D. The mixture of
Bernoullis gains almost 6 nats per image over independent pixels (k = 1 is the product of Bernoullis).

## Knowledge gap

**What the slides leave out**

- **Convergence of EM**: it is linear, and slow when components overlap (the missing information is large);
  ECM, accelerated EM, or direct gradient methods can be faster. EM as coordinate ascent on the ELBO, and as a
  majorize–minimize (MM) algorithm.
- **Choosing k**: held-out likelihood (used here), BIC, or a Bayesian non-parametric prior. Dirichlet processes are
  mentioned; the Chinese restaurant process and collapsed Gibbs sampling are not developed.
- **Identifiability and label switching**: the likelihood is invariant to permuting components, which matters for
  Bayesian inference and for interpreting clusters.
- **Variational EM / variational Bayes for mixtures**, the step from EM (point estimates of θ) to posteriors over θ.
- **KDE theory**: the bias–variance trade-off in `h`, optimal rates `n^{-4/(4+d)}` (the curse of dimensionality made
  quantitative), and Silverman's rule of thumb.
- Mixture of experts (a bonus slide) is the ancestor of the sparse MoE layers in modern large language models.

**Prerequisites to shore up**

- Jensen's inequality; KL divergence and why it is non-negative.
- Weighted MLE for Gaussians and Bernoullis.

**Open questions / further study**

- EM with a neural network in the M-step (or the E-step) is the bridge to the VAE: amortize the responsibilities with
  an encoder ([12](../12-variational-inference-and-vaes/)).
- Reading: Bishop, *PRML* ch. 9; Dempster, Laird & Rubin 1977; Neal & Hinton 1998, "A view of the EM algorithm that
  justifies incremental, sparse, and other variants."

## Connections

- ← [03 Clustering](../03-clustering-and-outliers/): k-means is hard-assignment EM; mean-shift clusters by KDE modes.
- ← [07](../07-density-estimation-mle-map/), [08](../08-gaussians-and-bayesian-learning/): the components and their
  closed-form M-steps.
- → [12 Variational inference and VAEs](../12-variational-inference-and-vaes/): the ELBO derived here is the VAE
  objective; a VAE is a continuous, infinite mixture with a neural decoder.
- → [13 HMMs](../13-markov-chains-and-hmms/): Baum–Welch is EM where the hidden variables form a Markov chain.
- → [18 Diffusion](../18-diffusion-and-image-generation/): KDE is a diffusion model's "perfect memorizer" failure mode
  (a mixture centred on training points), which is why sample-quality metrics can be fooled.
