# 13 Markov chains, message passing and hidden Markov models

**CPSC 440 lectures 16, 17, 20 (HMM section)** · [back to the graph](../README.md)

**What is learned:** a distribution over *sequences* in which each element depends only on the previous one; with
hidden states, a model of an unobserved process that generates the observations.

## Lectures

| Slides | Covers |
| --- | --- |
| `16-markov-chains.pdf` | rain data, the chain rule, the Markov assumption, homogeneous vs. inhomogeneous chains, MLE by counting, ancestral sampling, Monte Carlo and exact (Chapman–Kolmogorov) marginals, stationary distributions, PageRank, label propagation (bonus) |
| `17-msg-passing.pdf` | decoding vs. maximizing marginals, Viterbi decoding by dynamic programming, conditioning with forward messages, forward–backward and potentials (bonus), particle filters (bonus) |
| `20-ugms.pdf` (HMM part) | hidden Markov models, applications (speech, DNA, tracking with Kalman filters), inference and learning in HMMs |

## The models

### Markov chains

Chain rule plus the **Markov assumption** (the future is independent of the past given the present):
`p(x_1..x_T) = p(x_1) Π_t p(x_t | x_{t−1})`. A **homogeneous** chain shares one transition matrix `A` across time, so
it handles sequences of any length and pools data. Learning is counting transitions.

- **Ancestral sampling**: sample `x_1`, then each `x_t` from `A[x_{t−1}]`.
- **Exact marginals** (Chapman–Kolmogorov): `p(x_t) = p(x_{t−1})A`, `O(Tk²)` instead of summing over `kᵀ` paths.
- **Stationary distribution** `π* = π*A`: where the chain spends its time in the long run (unique if the chain is
  irreducible and aperiodic). **PageRank** is the stationary distribution of a surfer who follows a random link with
  probability `d` and teleports otherwise.

### Message passing

- **Decoding** `argmax_x p(x)` is not the same as taking each marginal's argmax. **Viterbi** computes it by dynamic
  programming: `M_t(j) = max_i M_{t−1}(i) A_ij`, then backtrack, `O(Tk²)`.
- **Conditioning on the future** needs messages from both directions: **forward–backward** computes every
  `p(x_t | evidence)` at once.

### Hidden Markov models

Hidden states `z_t` follow a Markov chain; each emits an observation `x_t ~ B[z_t]`. Inference uses the same
messages on the hidden chain:
`α_t(j) = B_j(x_t) Σ_i α_{t−1}(i)A_ij`, `β_t(i) = Σ_j A_ij B_j(x_{t+1}) β_{t+1}(j)`, `p(z_t | x) ∝ α_t β_t`,
normalized at every step to avoid underflow. Learning is EM (**Baum–Welch**): the E-step is forward–backward
(expected state and transition counts), the M-step re-normalizes those counts. Continuous Gaussian states with linear
dynamics give the **Kalman filter**.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`markov_chain.py`](markov_chain.py) | homogeneous chain with Dirichlet smoothing, ancestral sampling, Chapman–Kolmogorov marginals, stationary distribution by power iteration, PageRank |
| [`hmm.py`](hmm.py) | scaled forward–backward (marginals, pair marginals, log-likelihood), Viterbi in log space, Baum–Welch, an "occasionally dishonest casino" |

```bash
python 13-markov-chains-and-hmms/markov_chain.py
python 13-markov-chains-and-hmms/hmm.py
```

| Result (saved run) | |
| --- | --- |
| character chain sample | `t don lofoche. sulsta whess, arg. t pong. ros st tecererembre sthe…`: locally plausible, globally meaningless |
| career chain marginals at t = 10, exact vs. 5000 ancestral samples | industry 0.839 vs. 0.842, academia 0.113 vs. 0.112 |
| Viterbi, forward log-likelihood, and all marginals vs. brute force over 2⁸ paths | identical (marginal error 2e-15) |
| casino, 2000 rolls: state accuracy of Viterbi / max-marginals / "always fair" | 0.802 / 0.815 / 0.648 |
| Baum–Welch on 10 × 300 rolls: learned `A` | [[0.962, 0.038], [0.091, 0.909]] vs. true [[0.95, 0.05], [0.1, 0.9]] |
| learned `p(6 | state)` | 0.163, 0.493 vs. true 0.167, 0.5 |

Max-marginals beat Viterbi on per-roll accuracy because they optimize exactly that; Viterbi optimizes the probability
of the whole path. Which one you want depends on the loss.

## Knowledge gap

**What the slides leave out**

- **Convergence rates** of chains to stationarity: the spectral gap (`1 − |λ₂|`) controls mixing time, which is also
  what makes MCMC ([11](../11-monte-carlo-and-mcmc/)) fast or slow.
- **Higher-order and variable-order chains** (n-gram language models with smoothing such as Kneser–Ney), the practical
  version of character chains, and the baseline that neural LMs replaced.
- **Kalman filters and smoothers** are named but not derived; they are forward–backward with Gaussian messages.
  Extended / unscented Kalman filters and particle filters handle non-linear dynamics.
- **Semi-Markov / duration models**, input–output HMMs, and how HMMs were combined with neural networks in speech
  recognition (hybrid HMM-DNN, CTC).
- **Identifiability of HMMs and local optima of Baum–Welch**; spectral learning methods give consistent estimates.

**Prerequisites to shore up**

- Dynamic programming (the CPSC 320 prerequisite); eigenvectors of stochastic matrices.

**Open questions / further study**

- State-space models (S4, Mamba) are linear dynamical systems with learned parameters that rival Transformers on long
  sequences; they are a modern revival of this folder ([17](../17-sequence-models-and-transformers/)).
- Reading: Rabiner 1989, "A tutorial on hidden Markov models"; Murphy, *PML2* ch. 29; Jurafsky & Martin, *Speech and
  Language Processing* ch. 3 (n-grams) and appendix A (HMMs).

## Connections

- ← [07 Density estimation](../07-density-estimation-mle-map/): every conditional is a categorical learned by counting.
- ← [10 Mixtures and EM](../10-mixture-models-and-em/): an HMM is a mixture whose cluster labels form a Markov chain;
  Baum–Welch is EM.
- → [11 MCMC](../11-monte-carlo-and-mcmc/): MCMC builds a chain whose stationary distribution is the target.
- → [14 Graphical models](../14-graphical-models/): chains are the simplest DAGs and UGMs; forward–backward is belief
  propagation on a chain.
- → [17 RNNs](../17-sequence-models-and-transformers/): an RNN is a discriminative, deterministic-state HMM.
