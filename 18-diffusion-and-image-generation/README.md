# 18 Diffusion models and image generation

**CPSC 440 lecture 15 · CPSC 340 L36** · [back to the graph](../README.md)

**What is learned:** a denoiser `x̂_θ(x_t, t)` that, applied over and over starting from pure noise, turns noise into a
sample from the data distribution; optionally steered towards a condition (a class or a text prompt).

## Lectures

| Slides | Covers |
| --- | --- |
| `15-img-gen.pdf` | denoising autoencoders, diffusion models as hierarchical VAEs, the simplified objective, U-nets, other diffusion processes (cold diffusion, flow matching), score-matching and SDE views, conditioning with classifier guidance and classifier-free guidance, Stable Diffusion (latent diffusion); evaluating generative models: log-likelihood pitfalls, FID, precision/recall, classification accuracy score |
| `L36 - Generative Modelling.pdf` (CPSC 340) | what models learn, bias, energy costs, generative sampling, style transfer, GANs and diffusion for text-to-image, deepfakes, GPT-4, video models, course wrap-up |

## The model

### Forward process (fixed)

Add a little Gaussian noise `T` times: `q(x_t | x_{t−1}) = N(√(1 − β_t) x_{t−1}, β_t I)`. In closed form,
`q(x_t | x_0) = N(√ᾱ_t x_0, (1 − ᾱ_t) I)` with `ᾱ_t = Π_{s≤t}(1 − β_s)`. Choose the schedule so `ᾱ_T ≈ 0`: then `x_T`
is standard normal no matter where it started.

### Reverse process (learned)

A diffusion model is a **hierarchical VAE** whose encoder is the fixed forward process. Its ELBO reduces to matching
the true denoising posterior, which is Gaussian given `x_0`:
`q(x_{t−1} | x_t, x_0) = N(γ_t x_t + δ_t x_0, σ_t² I)`. So set
`p_θ(x_{t−1} | x_t) = N(γ_t x_t + δ_t x̂_θ(x_t, t), σ_t² I)` and train the denoiser with

`min E_{x_0, t, x_t ~ q(x_t | x_0)} ‖x̂_θ(x_t, t) − x_0‖²`

(the lecture drops the per-step weights, as most implementations do). Predicting `x_0`, the noise `ε`, or the score
`∇ log p_t(x_t)` are equivalent parameterizations, which is why the same model is also "denoising score matching."
Sampling runs the reverse chain from `x_T ~ N(0, I)`. Images use a U-net denoiser; Stable Diffusion runs the whole
process in the latent space of a pretrained autoencoder.

### Conditioning and guidance

- **Classifier guidance**: add `w∇_x log p(y | x_t)` from a separately trained, noise-aware classifier.
- **Classifier-free guidance**: train one network on both `(x, y)` and `(x, ∅)` (drop the label at random) and sample
  with `(1 + w) x̂(x_t, t, y) − w x̂(x_t, t, ∅)`. Ideally this samples `∝ p(x) p(y | x)^{1+w}`: larger `w` sharpens toward
  the condition and trades away diversity.

### Evaluating generative models

Held-out log-likelihood is the principled score but it can be fooled in both directions: a model can have great
likelihood and bad samples (mix in 0.1% of a good model: log-likelihood drops by only 7 nats), or great samples and
terrible likelihood (a KDE on the training set). **FID** compares the mean and covariance of Inception features of
real and generated images (a Fréchet distance between Gaussians): low variance but biased, and blind to memorization.
Precision/recall metrics separate "samples look real" from "samples cover the data."

## Code in this folder

[`diffusion.py`](diffusion.py): a DDPM-style model in JAX on 2-D two-moons data. Linear β schedule with `T = 100`,
closed-form posterior coefficients `γ_t, δ_t, σ_t²`, an MLP denoiser with sinusoidal time features predicting `x_0`,
ancestral sampling, classifier-free guidance, and a 2-D Fréchet distance.

```bash
python 18-diffusion-and-image-generation/diffusion.py        # about 20 s on a CPU
```

| Result (saved run) | Fréchet distance to data | median distance to nearest real point |
| --- | --- | --- |
| pure Gaussian noise | 0.093 | 0.170 |
| diffusion samples | **0.005** | **0.013** |
| a second draw of real data (reference) | 0.002 | 0.012 |

Classifier-free guidance asking for "moon 0", `w` = 0 / 1 / 3: fraction of samples on moon 0 is 1.000 / 1.000 / 0.982,
and the spread along the moon shrinks (std 0.815 → 0.737 → 0.704): guidance concentrates samples and reduces diversity.

Notice the first row: plain noise has a Fréchet distance of only 0.093 because the data were standardized, so the
noise already matches their mean and covariance. A two-moment metric cannot see the moons at all; the nearest-neighbour
distance can. FID has the same blind spot, in Inception-feature space.

## Knowledge gap

**What the slides leave out**

- **The full ELBO derivation** (the slides point to Calvin Luo's notes for equations 34–58), and why dropping the
  weights still works: the unweighted loss emphasizes perceptually important noise levels.
- **Noise schedules and parameterizations**: cosine schedules, ε- vs. x₀- vs. v-prediction, and their effect on sample
  quality at high and low noise.
- **Fast sampling**: DDIM (deterministic, few steps), probability-flow ODEs and ODE solvers, distillation and
  consistency models. The `T = ∞` SDE limit is only mentioned.
- **Flow matching / rectified flows** (named as "≈ cold diffusion") are now the standard for large image and video
  models.
- **GANs** appear in L36 only as history: the adversarial objective, mode collapse, and why diffusion displaced them.
- **Memorization and copyright**: diffusion models can reproduce training images (Carlini et al. 2023), the generative
  side of the memorization issue in [19](../19-representation-learning-and-llms/).
- **Evaluation for text-to-image**: CLIP score, human preference models, and the problems with FID on modern models.

**Prerequisites to shore up**

- Products and conditionals of Gaussians ([08](../08-gaussians-and-bayesian-learning/)); the VAE ELBO
  ([12](../12-variational-inference-and-vaes/)).
- Langevin dynamics and scores (the gradient of a log-density) for the score-based view.

**Open questions / further study**

- Why do diffusion models generalize instead of memorizing their (finite) training set, when the optimal denoiser for
  the empirical distribution would only reproduce training points?
- Reading: Ho, Jain & Abbeel 2020 (DDPM); Song et al. 2021 (score-based SDEs); Ho & Salimans 2022 (classifier-free
  guidance); Luo 2022, "Understanding diffusion models: a unified perspective"; Lipman et al. 2023 (flow matching).

## Connections

- ← [12 VAEs](../12-variational-inference-and-vaes/): a diffusion model is a hierarchical VAE with a fixed encoder; a
  denoising autoencoder is its one-step ancestor.
- ← [16 CNNs](../16-convolutional-networks/): U-net denoisers are fully-convolutional encoder–decoders.
- ← [11 MCMC](../11-monte-carlo-and-mcmc/): sampling is a Markov chain; score-based samplers are Langevin dynamics.
- ↔ [10 KDE](../10-mixture-models-and-em/): a KDE on the training set is the "perfect but useless" generative model that
  sample metrics can reward.
- → [19 Representation learning and LLMs](../19-representation-learning-and-llms/): text conditioning uses CLIP / language
  model embeddings; bias, energy use and deepfakes are discussed in L36 and lecture 24.
