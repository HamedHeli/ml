# 14 Graphical models: DAGs, undirected models, CRFs and beyond

**CPSC 440 lectures 19, 20** · [back to the graph](../README.md)

**What is learned:** a joint distribution over many variables that factorizes according to a graph, so that it can be
stored, sampled, and reasoned about (which variables are independent of which, given what).

## Lectures

| Slides | Covers |
| --- | --- |
| `19-dags.pdf` | higher-order Markov models, DAG models, d-separation (chain, fork, collider), the alarm example, equivalent graphs, why DAGs are not automatically causal, plate notation, tabular and sigmoid-belief-network conditionals, inference in forests (belief propagation), structure learning (score-based and constraint-based) |
| `20-ugms.pdf` | Gaussians as graphical models, undirected models (MRFs), conditional independence by separation, graphical lasso, tractability and treewidth, log-linear models, pseudo-likelihood and MCMC-based learning (persistent contrastive divergence), CRFs, CRFs + neural nets, HMMs; bonus: ICM, graph cuts, factor graphs, moralization, topic models (LSA, LDA), RBMs, deep belief networks |

## The models

### Directed acyclic graphs (Bayesian networks)

`p(x) = Π_j p(x_j | x_pa(j))`: a product of local conditionals, one per node given its parents. Markov chains, naive
Bayes, mixtures and HMMs are all DAGs. Conditionals can be tables (exponential in the number of parents), logistic
regressions (a sigmoid belief network), or neural networks. Sampling is ancestral in topological order.

**D-separation** reads conditional independences off the graph. A path is blocked at node `v` if `v` is observed in a
chain (`→ v →`) or fork (`← v →`), or if `v` is a collider (`→ v ←`) and neither `v` nor any descendant is observed.
`A ⊥ B | E` if every path is blocked. The collider rule is **explaining away**: two independent causes become dependent
once you observe their common effect. A DAG that fits the data is not automatically causal: different DAGs can encode
the same independences (Markov equivalence).

### Undirected graphical models (Markov random fields)

`p(x) = (1/Z) Π_c φ_c(x_c)` with non-negative potentials on cliques. Independence is plain graph separation. The
normalizer `Z` sums over every joint state, so exact inference is exponential in the graph's **treewidth**: easy on
chains and trees (belief propagation), hard on grids.

- **Ising model** / pairwise binary MRF: `log p(x) = Σ h_i x_i + Σ_(i,j) w_ij x_i x_j − log Z`, the standard image prior.
- **Learning** a log-linear UGM needs expectations under the model (because of `Z`): approximate them with MCMC
  (persistent contrastive divergence), or maximize the **pseudo-likelihood** `Σ_i log p(x_i | x_neighbours)`, which
  avoids `Z` entirely and is consistent.
- **Approximate decoding**: ICM (greedy coordinate-wise maximization), graph cuts for submodular binary problems.
- **Gaussian UGMs**: the graph is the sparsity pattern of the precision matrix. The **graphical lasso** maximizes
  `log det Θ − tr(SΘ) − λ‖Θ‖₁` to learn a sparse graph.

### Conditional random fields

A UGM over outputs `y` conditioned on inputs `x`: `p(y | x) ∝ exp(Σ features(y_c, x))`. It models dependencies among
outputs (neighbouring pixel labels, consecutive part-of-speech tags) while staying discriminative, and can be stacked
on a neural network that produces the unary potentials.

### Bonus models (lecture 20)

**Topic models**: latent Dirichlet allocation represents each document as a mixture of topics, each topic a
distribution over words, fit with Gibbs sampling or VI. **Restricted Boltzmann machines**: a bipartite UGM between
visible and hidden units, trained with contrastive divergence; stacked into deep belief networks, the 2006 spark of
deep learning.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`dag.py`](dag.py) | DAG with topological sort, d-separation by "Bayes ball" reachability, tabular conditional-probability tables learned by counting, ancestral sampling, the alarm network |
| [`ugm.py`](ugm.py) | Ising model Gibbs sampling, ICM, image denoising with a grid prior, pseudo-likelihood learning of `(h, w)`, graphical lasso by ADMM |

```bash
python 14-graphical-models/dag.py
python 14-graphical-models/ugm.py
```

| Result (saved run) | |
| --- | --- |
| d-separation in the alarm network | burglary ⊥ earthquake: yes; given alarm: no; given only a call: no; calls ⊥ each other given alarm: yes |
| explaining away, 200k samples: `p(burglary | alarm)` → `p(burglary | alarm, earthquake)` | 0.443 → 0.150 |
| correlation of burglary and earthquake, marginally / given alarm | 0.000 / −0.743 |
| Ising denoising of a disc with 20% flipped pixels: error noisy / ICM / Gibbs max-marginals | 0.194 / 0.035 / **0.024** |
| pseudo-likelihood on Gibbs samples of a 6×6 Ising model | `h` 0.100 (true 0.1), `w` 0.306 (true 0.3) |
| graphical lasso on an 8-node chain | 7/7 true edges, 1 false edge; the plain inverse covariance has all 28 entries nonzero |

## Knowledge gap

**What the slides leave out**

- **Exact inference in general graphs**: variable elimination, the junction tree algorithm, and why treewidth is the
  right complexity measure (only sketched in bonus slides).
- **Loopy belief propagation and variational inference for MRFs** (mean field, Bethe free energy), the standard
  approximate-inference methods for grids and codes.
- **Causal inference**: the slides warn that DAGs are not causal; the do-calculus, interventions, the backdoor
  criterion and structural causal models (Pearl) are what makes a DAG causal and when you can estimate an effect
  from observational data.
- **Structure learning theory**: score-based search is NP-hard; PC / GES algorithms, faithfulness assumptions, and
  why only the Markov equivalence class is identifiable from observational data.
- **CRF training details**: the gradient is "observed features − expected features," needing forward–backward on
  chains; structured SVMs as the max-margin alternative.
- **Energy-based models** today: modern EBMs and score matching are the descendants of UGMs and RBMs, and connect to
  diffusion models ([18](../18-diffusion-and-image-generation/)).

**Prerequisites to shore up**

- Graph algorithms (topological sort, BFS, cliques); the precision matrix of a Gaussian ([08](../08-gaussians-and-bayesian-learning/)).

**Open questions / further study**

- Do we still need explicit graphical structure when Transformers can learn dependencies from data? The slides pose this
  ("Do we need UGMs in neural networks?"); structured outputs and small-data scientific settings are where the answer
  is still yes.
- Reading: Koller & Friedman, *Probabilistic Graphical Models*; Pearl, *Causality* (or *The Book of Why* for intuition);
  Wainwright & Jordan 2008.

## Connections

- ← [13 Markov chains and HMMs](../13-markov-chains-and-hmms/): chains are the base case; forward–backward is belief
  propagation on a chain.
- ← [11 MCMC](../11-monte-carlo-and-mcmc/): Gibbs sampling is the natural sampler for MRFs and is used inside learning.
- ← [08 Gaussians](../08-gaussians-and-bayesian-learning/): zeros of the precision matrix are missing edges.
- → [15 Neural networks](../15-neural-networks/) and [16](../16-convolutional-networks/): CRFs on top of networks for
  structured outputs; RBMs and DBNs in deep-learning history.
- → [05 Regularization](../05-feature-selection-and-regularization/): the graphical lasso is lasso on the precision matrix;
  causal DAGs explain why regression coefficients are not effects.
