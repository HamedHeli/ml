# 12 Variational inference, autoencoders and VAEs

**CPSC 440 lectures 13, 14 · CPSC 340 L34** · [back to the graph](../README.md)

**What is learned:** an approximate posterior `q` fit by optimization instead of sampling; and, in a VAE, a deep
generative model `p_θ(x | z)` together with an *encoder* `q_φ(z | x)` that does that inference in one forward pass.

## Lectures

| Slides | Covers |
| --- | --- |
| `13-vae.pdf`, `13-vae+conditional.pdf` | KL divergence, reverse-KL variational inference, best Gaussian approximation, the reparameterization trick, mean field, deep latent-variable models, amortized inference, the ELBO, VAEs on MNIST, conditional VAEs for in-painting; bonus: MLE minimizes KL, REINFORCE. The two files are near-duplicates; the second adds the conditional-generation slides. |
| `14-deconv-reps.pdf` (second half) | VAE decoders, autoencoders, representation learning, semi-supervised learning, posterior collapse, VQ-VAE, β-VAE and Wasserstein autoencoders (bonus) |
| `L34 - Autoencoders.pdf` (CPSC 340) | autoencoders, PCA vs. deep autoencoders, decoders as generative models, denoising autoencoders, multi-label classification |

## The models

### Variational inference

Choose a tractable family `q_φ` and minimize the **reverse KL** `KL(q ‖ p(· | data))`. Since
`log p(data) = ELBO(φ) + KL(q ‖ p)`, this is the same as maximizing

`ELBO(φ) = E_q[log p̃(w)] + H(q)`,

which needs only the unnormalized posterior. Reverse KL is **mode-seeking**: `q` avoids putting mass where `p` has
none, so it tends to sit in one mode and underestimate variance. **Mean-field** (fully factorized) `q` cannot represent
correlations at all.

**Reparameterization trick**: write `w = μ + Lε`, `ε ~ N(0, I)`, so `∇_φ E_q[f(w)] = E_ε[∇_φ f(μ + Lε)]`: a low-variance
gradient through the sample, usable with SGD. (REINFORCE / score-function gradients work for discrete `q` but are much
noisier.)

### Autoencoders

An encoder `f: x → z` and decoder `g: z → x` trained to reconstruct the input through a bottleneck. Linear `f, g` with
squared error is PCA ([09](../09-latent-factor-models/)). **Denoising** autoencoders reconstruct `x` from a corrupted
copy, which prevents the identity solution and is the seed of diffusion models
([18](../18-diffusion-and-image-generation/)).

### Variational autoencoders

A deep latent-variable model: `z ~ N(0, I)`, `x | z ~ p_θ(x | z)` with a neural decoder. `p_θ(x) = ∫p(z)p_θ(x | z)dz` is
intractable, so learn an **amortized** inference network `q_φ(z | x) = N(μ_φ(x), diag σ²_φ(x))` and maximize, per
example,

`ELBO(x) = E_q[log p_θ(x | z)] − KL(q_φ(z | x) ‖ N(0, I))` (≤ `log p_θ(x)`).

The first term is a reconstruction loss through a *noisy* code; the second keeps codes near the prior so that decoding a
prior sample gives a sensible `x`. The Gaussian KL has a closed form.

- **Conditional VAE**: condition prior, encoder and decoder on side information `c` (a label, or the visible half of an
  image). Sampling different `z` with the same `c` gives different plausible completions.
- **Posterior collapse**: a powerful decoder (or a heavy KL weight) can ignore `z`, making `q(z | x) = p(z)` and the code
  useless. **β-VAE** (`β > 1` on the KL) trades reconstruction for more "disentangled" codes, and pushes toward
  collapse. **VQ-VAE** uses a discrete codebook instead.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`variational_inference.py`](variational_inference.py) | full-covariance and mean-field Gaussian VI for Bayesian logistic regression, reparameterized ELBO gradients with JAX + Adam |
| [`vae.py`](vae.py) | autoencoder, VAE (Bernoulli decoder, closed-form KL), β-VAE, the importance-weighted (IWAE) bound on `log p(x)`, conditional VAE that in-paints the bottom half of a digit |

```bash
python 12-variational-inference-and-vaes/variational_inference.py
python 12-variational-inference-and-vaes/vae.py        # about 30 s on a CPU
```

| Result (saved run) | |
| --- | --- |
| VI posterior predictive (grid truth 0.8925): full-covariance / mean-field | 0.8915 / 0.8939 |
| posterior std of `w` (truth 0.414, 0.608) and correlation (truth +0.19) | full: (0.419, 0.598), +0.19; mean-field: (0.413, 0.587), +0.00 |
| autoencoder (k = 4) reconstruction NLL, train / test | 4.2 / **26.8** nats: a deterministic code memorizes |
| VAE (k = 4) test ELBO = reconstruction − KL | −18.8 = −12.6 − 6.2 nats |
| VAE importance-weighted bound (K = 200) on `log p(x)` | **−16.8** nats per image |
| β-VAE, β = 4 | KL 0.18 nats: the posterior collapses and reconstruction falls to −24.6 |

The VAE's −16.8 nats beats the product of Bernoullis (−25.1, [07](../07-density-estimation-mle-map/)) and the 30-component
mixture of Bernoullis (−19.4, [10](../10-mixture-models-and-em/)) on the same binarized digits. The 2-nat gap between
ELBO and IWAE bound is the price of the Gaussian `q`. The conditional VAE's in-painting output (printed by the script)
gives four completions that follow the stroke of the visible half and one that disagrees.

## Knowledge gap

**What the slides leave out**

- **The amortization gap vs. the approximation gap**: the ELBO is loose both because `q` is Gaussian and because one
  encoder serves all `x`. Semi-amortized VI and richer posteriors (normalizing flows, Rezende & Mohamed 2015) close it;
  IWAE (Burda et al. 2016, used here for evaluation) trains on a tighter bound.
- **Stochastic VI for conjugate-exponential models** (Hoffman et al. 2013) and coordinate-ascent mean-field updates,
  the classic pre-deep-learning VI.
- **Forward vs. reverse KL** and their alternatives (α-divergences, expectation propagation), which change mode-seeking
  into mass-covering behaviour.
- **Why VAE samples are blurry**: a factorized Gaussian or Bernoulli decoder averages over modes; the standard fixes
  are hierarchical VAEs and powerful autoregressive or diffusion decoders.
- **Disentanglement claims** for β-VAE: Locatello et al. 2019 show unsupervised disentanglement is impossible without
  inductive biases.
- **Discrete latents**: Gumbel-softmax / concrete relaxations, and how VQ-VAE's straight-through estimator works.

**Prerequisites to shore up**

- KL divergence, entropy of a Gaussian, Jensen's inequality.
- Automatic differentiation through sampling (what a reparameterized sample *is* in the computation graph).

**Open questions / further study**

- Is a good ELBO a good representation? Often not: posterior collapse gives a fine likelihood and a useless code. See
  the representation-learning discussion in [19](../19-representation-learning-and-llms/).
- Reading: Kingma & Welling 2014 and their 2019 monograph "An introduction to variational autoencoders"; Blei,
  Kucukelbir & McAuliffe 2017, "Variational inference: a review for statisticians."

## Connections

- ← [10 Mixtures and EM](../10-mixture-models-and-em/): the ELBO is the same bound; EM computes the exact posterior in the
  E-step, a VAE learns to approximate it.
- ← [11 Monte Carlo](../11-monte-carlo-and-mcmc/): the same Bayesian logistic regression, solved by optimization instead of
  sampling; IWAE is importance sampling with the encoder as proposal.
- ← [09 PCA](../09-latent-factor-models/): a linear-Gaussian VAE is probabilistic PCA.
- → [16 CNNs](../16-convolutional-networks/): image VAE decoders use transposed convolutions.
- → [18 Diffusion](../18-diffusion-and-image-generation/): a diffusion model is a hierarchical VAE with a fixed encoder.
