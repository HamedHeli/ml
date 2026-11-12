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

## The two papers to know

### DDPM: Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020

The paper that made diffusion competitive with GANs. The model is the one above: a fixed forward chain with `T = 1000`
steps and a linear schedule `β_1 = 10⁻⁴ … β_T = 0.02`, and a learned Gaussian reverse chain. Its contributions:

- **ε-prediction.** Write `x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε` and let the network predict the noise `ε` instead of the
  posterior mean. The reverse step becomes
  `x_{t−1} = (1/√α_t) (x_t − β_t / √(1 − ᾱ_t) · ε_θ(x_t, t)) + σ_t z`, with `σ_t² = β_t` (or the posterior variance;
  both worked about equally).
- **The simplified loss** `L_simple = E_{t, x_0, ε} ‖ε − ε_θ(√ᾱ_t x_0 + √(1 − ᾱ_t) ε, t)‖²`: the ELBO with its
  per-step weights dropped. It gives *worse* likelihood and *better* samples, because the dropped weights would put
  most effort on tiny noise levels that hardly affect what images look like.
- **The link to score matching.** `ε_θ(x_t, t) = −√(1 − ᾱ_t) ∇ log p_t(x_t)`, so training is denoising score matching
  over all noise levels, and sampling resembles annealed Langevin dynamics (Song & Ermon 2019).
- **Results.** CIFAR-10 FID 3.17 (Inception score 9.46), then state of the art for unconditional models, and
  256 × 256 LSUN samples comparable to ProgressiveGAN. The costs: 1000 network evaluations per sample, and a
  log-likelihood (≤ 3.75 bits/dim for the `L_simple` model) worse than autoregressive models. The authors read this as diffusion being an
  excellent *lossy* compressor: most bits go into imperceptible detail.

[`diffusion.py`](diffusion.py) implements this model (with `x_0`-prediction, an equivalent parameterization,
and `T = 100`).

### EDM: Karras, Aittala, Aila & Laine, "Elucidating the Design Space of Diffusion-Based Generative Models," NeurIPS 2022

By 2022, DDPM, score-based SDEs (VP and VE), DDIM and improved DDPM each mixed their own choice of noise schedule,
scaling, network parameterization, loss weighting and sampler, so it was unclear which choice mattered. EDM writes
them all as one ODE and shows the choices can be made **independently**.

**One ODE for every model.** With the noise added directly (`x = y + σn`, no shrinking of `y`) and time equal to the
noise level (`σ(t) = t`), the probability-flow ODE is

`dx/dσ = (x − D(x; σ)) / σ`,  with score `∇_x log p(x; σ) = (D(x; σ) − x) / σ²`,

where `D` is the denoiser, the same object as `x̂_θ` above. With `σ(t) = t` the exact trajectories are close to
straight lines at high noise, so few large steps suffice.

**Sampling (the "deterministic sampler" choices)**

- **Heun's 2nd-order method**: take an Euler step, evaluate the slope at the end point, and redo the step with the
  average slope. It costs `2N − 1` evaluations for `N` steps but its error shrinks as `h²` instead of `h`.
- **Step spacing**: `σ_i = (σ_max^{1/ρ} + (i/(N−1)) (σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ` with `ρ = 7`, `σ_max = 80`,
  `σ_min = 0.002`, then `σ_N = 0`. A large `ρ` puts most steps at low noise, where the fine detail is decided.
- A **stochastic sampler** ("churn": add a little noise, then step back) corrects errors made in earlier steps. It
  helps some models but not all.

**Training (the "network and loss" choices)**

- **Preconditioning**: wrap the raw network `F` as `D(x; σ) = c_skip(σ) x + c_out(σ) F(c_in(σ) x; c_noise(σ))`, with
  `c_in = 1/√(σ² + σ_d²)`, `c_skip = σ_d² / (σ² + σ_d²)`, `c_out = σ σ_d / √(σ² + σ_d²)` and `c_noise = ¼ ln σ`
  (`σ_d` is the data's standard deviation). The network's input and target then have unit variance at every
  noise level. At low noise the skip connection does the work (`D ≈ x`); at high noise the network predicts the clean
  data itself. ε-prediction and x₀-prediction are the two extremes of this blend.
- **Loss weighting** `λ(σ) = 1/c_out(σ)²`, so the effective loss is a unit-weighted MSE on the network's output.
- **Noise distribution for training**: `ln σ ~ N(−1.2, 1.2²)`, which concentrates training on the middle noise levels
  where the loss can actually be reduced.

**Results.** CIFAR-10 FID 1.79 class-conditional and 1.97 unconditional with **35** network evaluations (DDPM used
1000), and ImageNet-64 FID 1.36 by retraining an existing network with the new choices. EDM's preconditioning, noise
distribution and "Karras" step spacing became a standard baseline: they are the default samplers in common Stable
Diffusion tooling, and the starting point of EDM2 and of consistency models.

[`edm.py`](edm.py) implements EDM's preconditioning, loss weighting, noise distribution, `ρ` step spacing and the
Euler and Heun samplers on the same data as `diffusion.py`.

## Notebook: a diffusion model on 8×8 digits

[**diffusion_digits.ipynb**](diffusion_digits.ipynb) trains a class-conditional DDPM with a small **U-Net** (about
270k parameters: conv blocks, residual blocks with time and class embeddings, self-attention at 4×4, skip
connections). It takes about 5 minutes on a CPU. The notebook documents the network's structure with a diagram, the
layer-by-layer shapes and a parameter count per block. It is built around the question *"if every image ends as pure
noise, how does the model know which image you want?"*:

- **The forward process erases the image.** By the last step the correlation with the original is 0, so the model
  cannot recover a particular image. It learns the distribution and draws a **new** sample. Unconditional samples
  cover all ten digits and are as far from their nearest training image as training images are from each other.
- **The noise is a random seed and sets the style.** With a deterministic sampler (DDIM), the same noise gives the
  same image every time, and interpolating between two noises morphs one digit into another.
- **The condition sets the content.** The same 8 noises with labels 0–9 give the requested digit in 100% of 80
  samples. Classifier-free guidance raises label accuracy from 97% to 100%.
- **To get a variation of a specific image, start from that image noised part-way (SDEdit).** After noising a 7 to
  `t = 40` and denoising, 100% of results are still 7s; at `t = 120`, 60%; at full noise, 10% (chance).

## Code in this folder

| File | What it implements |
| --- | --- |
| [`diffusion.py`](diffusion.py) | DDPM with `x_0`-prediction, ancestral sampling, classifier-free guidance, 2-D Fréchet distance |
| [`edm.py`](edm.py) | EDM: preconditioning, `λ(σ)` weighting, log-normal `σ` training, `ρ` step spacing, Euler and Heun ODE samplers, the exact denoiser of a finite training set |

**DDPM** ([`diffusion.py`](diffusion.py)): a DDPM-style model in JAX on 2-D two-moons data. Linear β schedule with `T = 100`,
closed-form posterior coefficients `γ_t, δ_t, σ_t²`, an MLP denoiser with sinusoidal time features predicting `x_0`,
ancestral sampling, classifier-free guidance, and a 2-D Fréchet distance.

```bash
python 18-diffusion-and-image-generation/diffusion.py        # about 20 s on a CPU
python 18-diffusion-and-image-generation/edm.py              # about 35 s
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

**EDM** ([`edm.py`](edm.py)), same data, same network size and training budget as DDPM.

With the learned denoiser, Heun at `N = 18` (35 evaluations) reaches Fréchet distance **0.0037** and median
nearest-neighbour distance 0.0127. DDPM above needs 100 evaluations for 0.005 and 0.013.

| Experiment | Result |
| --- | --- |
| solver error only (exact denoiser of 500 training points; distance to a 400-step solution), Euler vs. Heun | 9 NFE: 0.335 vs. 1.076 · 15: 0.193 vs. 0.221 · 21: 0.123 vs. **0.111** · 35: 0.071 vs. **0.039** |
| learned denoiser, Fréchet distance, Euler vs. Heun | 9 NFE: **0.062** vs. 2.14 · 15: **0.017** vs. 0.062 · 21: **0.008** vs. 0.014 · 35: 0.0032 vs. 0.0037 |
| step spacing `ρ`, Heun `N = 18` | `ρ = 1` (uniform): 5.10 · `ρ = 3`: 0.0050 · `ρ = 7`: **0.0037** |
| no preconditioning and no `λ(σ)` (plain `x_0` regression), Heun `N = 18` | 0.52, i.e. 140× worse |

What these show:

- **Heun only wins once the steps are small enough.** With the exact denoiser, where only the solver makes errors,
  Heun's error falls faster and it is ahead from about 21 evaluations; at 35 it has half of Euler's error. At 9
  evaluations, a step from `σ ≈ 0.27` to `0.002` divides by a tiny `σ` at a point the Euler step guessed badly, and the
  second-order correction makes things worse. With the learned denoiser, network error dominates at 35 evaluations and
  the two tie. The paper's 35-evaluation sweet spot is where Heun's advantage appears on images.
- **`ρ` matters more than the solver.** Uniform spacing (`ρ = 1`) spends 17 of 18 steps above `σ = 4`, where nothing
  happens, and fails.
- **Preconditioning is the largest single gain.** Without it the network must output values that range over 80
  units at high noise and 0.002 units of correction at low noise.
- **The exact denoiser memorizes.** Its ODE lands exactly on training points (median distance 0), so an "optimal"
  diffusion model of a finite dataset can only copy. That the learned model does not (see the open question below)
  is due to the network's limited capacity and inductive bias.

## Knowledge gap

**What the slides leave out**

- **The full ELBO derivation** (the slides point to Calvin Luo's notes for equations 34–58), and why dropping the
  weights still works: the unweighted loss emphasizes perceptually important noise levels.
- **Noise schedules and parameterizations**: cosine schedules, ε- vs. x₀- vs. v-prediction, and their effect on sample
  quality at high and low noise. EDM (above) unifies these as choices of `c_skip`, `c_out` and `λ(σ)`.
- **Fast sampling**: DDIM (deterministic, few steps), probability-flow ODEs and ODE solvers (EDM's Heun sampler
  above; DPM-Solver), distillation and consistency models. The `T = ∞` SDE limit is only mentioned.
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
- Numerical ODE solvers: Euler vs. Heun (explicit trapezoidal), local vs. global error and order, step-size
  selection; these are what EDM's sampler choices are about.

**Open questions / further study**

- Why do diffusion models generalize instead of memorizing their (finite) training set, when the optimal denoiser for
  the empirical distribution would only reproduce training points?
- Reading: Ho, Jain & Abbeel 2020, "Denoising diffusion probabilistic models," NeurIPS (DDPM); Karras, Aittala,
  Aila & Laine 2022, "Elucidating the design space of diffusion-based generative models," NeurIPS (EDM);
  Song et al. 2021 (score-based SDEs); Ho & Salimans 2022 (classifier-free
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
