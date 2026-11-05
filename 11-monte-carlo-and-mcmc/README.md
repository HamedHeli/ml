# 11 Monte Carlo, importance sampling and MCMC

**CPSC 440 lectures 12, 18** · [back to the graph](../README.md)

**What is learned:** not a model but a way to *use* one: approximate expectations under distributions you can only
evaluate up to a constant, most importantly Bayesian posteriors that are not conjugate.

## Lectures

| Slides | Covers |
| --- | --- |
| `12-mc-var.pdf` | Bayesian inference tasks, Monte Carlo estimation and its variance, snakes and ladders, conditional probabilities by MC, Bayesian logistic regression as motivation, rejection sampling, importance sampling, self-normalized IS, the variational idea, the Laplace approximation |
| `18-mcmc.pdf` | Markov chains for Monte Carlo, Metropolis, Metropolis–Hastings, burn-in and thinning, Gibbs sampling, Gibbs as MH, advanced methods (bonus pointers) |

## The methods

### Plain Monte Carlo

`E_p[f(X)] ≈ (1/n) Σ f(x_i)` with `x_i ~ p`. Unbiased, variance `Var(f)/n`, so the error falls like `1/√n` in any
dimension. The catch is that you need samples from `p`.

### Rejection sampling

Sample `x ~ q`, accept with probability `p̃(x) / (M q(x))` where `M ≥ max p̃/q`. Accepted samples are exact draws from
`p`. The acceptance rate is `Z_p / M`, which collapses when `q` is a poor match, especially in high dimension.

### Importance sampling

`E_p[f] = E_q[f p/q]`: draw from `q`, reweight. With an unnormalized `p̃`, normalize the weights
(**self-normalized IS**): consistent but biased. The effective sample size `1/Σ w̃_i²` warns when a few weights
dominate; `q` must have heavier tails than `p` or the variance can be infinite.

### Laplace approximation

A Gaussian at the posterior mode with covariance equal to the inverse Hessian of the negative log-posterior. One
Newton solve; accurate when the posterior is close to Gaussian (lots of data), wrong for skewed or multimodal
posteriors.

### Markov chain Monte Carlo

Construct a Markov chain whose **stationary distribution is `p`**, run it, and average `f` over the states after a
**burn-in**. Samples are correlated, so `n` steps are worth fewer independent samples.

- **Metropolis–Hastings**: propose `x' ~ q(· | x)`, accept with probability
  `min(1, p̃(x')q(x | x') / (p̃(x)q(x' | x)))`, otherwise repeat `x`. Needs only `p̃`. Detailed balance makes `p`
  stationary. Random-walk Metropolis uses a symmetric Gaussian step, whose size trades acceptance rate against
  distance moved.
- **Gibbs sampling**: resample one variable from its full conditional `p(x_j | x_{−j})`. It is MH with acceptance 1,
  needs no tuning, and is natural for graphical models ([14](../14-graphical-models/)), but mixes slowly when
  variables are strongly correlated.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`monte_carlo.py`](monte_carlo.py) | MC estimates, a Bayesian logistic regression posterior, brute-force grid truth, Laplace approximation by Newton's method, rejection sampling (prior proposal, `M` = likelihood at the MLE), importance sampling with effective sample size |
| [`mcmc.py`](mcmc.py) | random-walk Metropolis on the same posterior, Gibbs sampling for a correlated 2-D Gaussian, integrated autocorrelation time |

```bash
python 11-monte-carlo-and-mcmc/monte_carlo.py
python 11-monte-carlo-and-mcmc/mcmc.py
```

The running problem: Bayesian logistic regression (30 points, 2 weights), `p(y = 1 | x* = 1, data)`, whose exact
value from a 400×400 grid is **0.8925**.

| method | estimate | cost / diagnostics |
| --- | --- | --- |
| plug-in MAP | 0.9016 | ignores uncertainty in `w`, overconfident |
| Laplace + MC | 0.8799 | one Newton solve |
| rejection sampling | 0.8899 | acceptance rate 0.8% |
| importance sampling, Laplace proposal | **0.8926** | effective sample size 17682 of 20000 |
| importance sampling, prior proposal | 0.8916 | effective sample size 703 of 20000 |
| Metropolis, step 0.05 / 0.5 / 5 | 0.8946 / 0.8957 / 0.8957 | acceptance 0.95 / 0.54 / 0.02, autocorrelation time 388 / **14** / 83 |

Gibbs on a 2-D Gaussian with correlation 0 / 0.9 / 0.99: autocorrelation time 1.0 / 8.4 / 98.

## Knowledge gap

**What the slides leave out**

- **Convergence diagnostics**: how do you know burn-in is over and the chain has mixed? Trace plots, the
  Gelman–Rubin `R̂` with multiple chains, effective sample size (computed here), and the fact that no diagnostic can
  prove convergence.
- **Why MH works**: detailed balance is sufficient, not necessary; ergodicity (irreducible, aperiodic) is needed for
  convergence; the optimal random-walk acceptance rate is about 0.234 in high dimension (Roberts et al. 1997).
- **Gradient-based MCMC**: Hamiltonian Monte Carlo and NUTS (what Stan and PyMC use), Langevin dynamics, and
  stochastic-gradient Langevin for big data. The bonus slide lists them; this is where modern practice lives, and
  Langevin dynamics is also how score-based diffusion models sample ([18](../18-diffusion-and-image-generation/)).
- **Variance reduction**: control variates, antithetic sampling, Rao–Blackwellization, quasi-Monte Carlo.
- **Sequential Monte Carlo / particle filters** appear only as a bonus slide in [13](../13-markov-chains-and-hmms/).
- **Estimating normalizing constants** (marginal likelihoods) by annealed importance sampling, needed to compare
  models.

**Prerequisites to shore up**

- Markov chain stationary distributions ([13](../13-markov-chains-and-hmms/)); the law of large numbers and central
  limit theorem.

**Open questions / further study**

- MCMC is exact in the limit but slow; variational inference ([12](../12-variational-inference-and-vaes/)) is fast but
  biased. When is each the right tool, and can they be combined (e.g. MCMC-refined variational posteriors)?
- Reading: MacKay, *Information Theory, Inference and Learning Algorithms* ch. 29–30; Neal 2011, "MCMC using
  Hamiltonian dynamics"; Betancourt 2017, "A conceptual introduction to Hamiltonian Monte Carlo."

## Connections

- ← [06 Linear classifiers](../06-linear-classifiers-and-kernels/): Bayesian logistic regression is the running example.
- ← [08 Gaussians](../08-gaussians-and-bayesian-learning/): the Laplace approximation and Gaussian proposals.
- ← [13 Markov chains](../13-markov-chains-and-hmms/): MCMC needs stationary distributions; read the chains folder for
  the theory.
- → [12 Variational inference](../12-variational-inference-and-vaes/): replace sampling with optimization.
- → [14 Graphical models](../14-graphical-models/): Gibbs sampling for Ising models and UGM learning.
