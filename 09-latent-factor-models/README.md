# 09 Latent-factor models: PCA, recommender systems, MDS

**CPSC 340 L25, L26, L27, L28** · [back to the graph](../README.md)

**What is learned:** a low-dimensional code `z_i` for every example and a small set of "parts" `W` such that
`x_i ≈ Wᵀz_i`; or, for embeddings, low-dimensional points whose distances match the data's.

## Lectures

| Slides | Covers |
| --- | --- |
| `L25 - PCA.pdf` | end of the supervised part, latent-factor models, PCA as matrix factorization, what PCA is doing |
| `L26 - PCA (2).pdf` | PCA geometry, the loss and prediction, eigenfaces, non-uniqueness and how SVD fixes it |
| `L27 - PCA (3).pdf` | alternatives to SVD (alternating minimization, gradient descent), variations: robust PCA, NMF, sparse PCA; recommender systems intro; bonus: probabilistic PCA, factor analysis, ICA, topic models |
| `L28 - Recommender Systems.pdf` | collaborative filtering, matrix factorization with missing entries, biases, content-based and hybrid methods, MDS, manifolds and t-SNE, word2vec |

## The models

### Principal component analysis

Approximate centred data `X` (n × d) by `ZW` with `Z` (n × k), `W` (k × d), minimizing `‖ZW − X‖²_F`. The solution
is the top-`k` right singular vectors of `X` as the rows of `W` (orthonormal, sorted by variance explained), and
`Z = XWᵀ`. Uses: compression, visualization (`k = 2`), denoising, de-correlating features, "eigenfaces."

- **Non-uniqueness**: any invertible `R` gives the same fit with `(ZR, R⁻¹W)`, so we normalize (orthonormal rows),
  orthogonalize and order by variance to pin it down.
- **Alternating minimization** (solve for `Z` with `W` fixed, then `W` with `Z` fixed) reaches the same subspace
  and generalizes to other losses, regularizers and missing data, which SVD does not.
- **Variations**: robust PCA (absolute instead of squared error, ignores sparse gross errors); non-negative matrix
  factorization (`Z, W ≥ 0`, gives additive "parts"); sparse PCA (L1 on `W`).

### Recommender systems: matrix factorization with missing entries

The rating matrix `Y` (users × items) is mostly missing. Fit

`Σ_(i,j) observed (b + b_i + b_j + z_iᵀw_j − y_ij)² + λ(‖Z‖² + ‖W‖²)`,

only over observed entries: a global mean, user and item biases, and a low-rank "taste × attribute" term. This is
**collaborative filtering** (uses only ratings). **Content-based** filtering regresses ratings on item/user
features and handles new items (the cold-start problem); hybrids combine both. Fit by SGD or alternating least
squares.

### Multi-dimensional scaling and t-SNE

MDS directly optimizes low-dimensional points `z_i` so that `‖z_i − z_j‖ ≈ D_ij`. With Euclidean distances the
classical solution is PCA of the double-centred squared-distance matrix. **t-SNE** matches *neighbourhood
probabilities* (Gaussian in the data, heavy-tailed Student-t in the embedding) rather than distances, which preserves
local structure and unrolls manifolds, at the cost of meaningless global distances.

### word2vec

Learn a vector per word so that words appearing in similar contexts get similar vectors: a factorization of
(shifted) word co-occurrence statistics, trained by predicting context words. The ancestor of the token embeddings in
[17](../17-sequence-models-and-transformers/).

## Code in this folder

[`latent_factors.py`](latent_factors.py): PCA by SVD, PCA by alternating least squares, robust PCA by gradient
descent on a smoothed L1 loss, subspace-angle comparison, matrix factorization with biases by ALS, classical MDS.

```bash
python 09-latent-factor-models/latent_factors.py
```

| Result (saved run) | |
| --- | --- |
| PCA on 8×8 digits, k = 2 / 10 / 30 | 28.5% / 73.8% / 95.9% of variance |
| alternating minimization vs. SVD, k = 5 | same subspace (largest principal angle 1e-6°) |
| rank-2 data with 5% gross corruptions: angle to the true subspace | PCA 36.7°, robust PCA **0.4°** |
| synthetic ratings, 10% observed, RMSE on the unobserved 90% | global mean 1.086, biases only 0.885, biases + rank-3 factors **0.499** |
| classical MDS of 10-D points on a plane | distances reproduced to 6e-15 in 2-D |

## Knowledge gap

**What the slides leave out**

- **Why SVD solves PCA**: the Eckart–Young theorem, and the equivalence of "maximize projected variance" and
  "minimize reconstruction error." Randomized SVD for large matrices.
- **Probabilistic PCA and factor analysis** are bonus slides only: PCA as the MLE of a linear-Gaussian latent-variable
  model, which gives a likelihood (for choosing `k`), handles missing data with EM, and is the linear special case of
  the VAE in [12](../12-variational-inference-and-vaes/).
- **Choosing k**: scree plots and "variance explained" are heuristics; probabilistic PCA allows held-out likelihood
  and Bayesian model selection.
- **Matrix completion theory**: when can a low-rank matrix be recovered from few entries (incoherence, nuclear-norm
  relaxation, Candès & Recht 2009)? Missing-not-at-random ratings (people rate what they like) bias everything.
- **Recommender evaluation**: RMSE is not what matters in practice; ranking metrics (recall@k, NDCG), implicit
  feedback (clicks, not ratings), and two-tower neural retrieval models are not covered.
- **t-SNE pitfalls**: perplexity dependence, cluster sizes and inter-cluster distances that mean nothing; UMAP as the
  common alternative.
- ICA (non-Gaussian latent factors, blind source separation) is bonus only.

**Prerequisites to shore up**

- The SVD and eigendecomposition; orthonormal bases and projections; the Frobenius norm.

**Open questions / further study**

- Non-linear latent-factor models: autoencoders and VAEs ([12](../12-variational-inference-and-vaes/)) replace `ZW`
  with a neural decoder; when are they worth it over PCA?
- Reading: *ESL* ch. 14.5; Koren, Bell & Volinsky 2009, "Matrix factorization techniques for recommender systems";
  van der Maaten & Hinton 2008 (t-SNE); Wattenberg et al. 2016, "How to use t-SNE effectively."

## Connections

- ← [03 Clustering](../03-clustering-and-outliers/): k-means is factorization with one-hot `Z`.
- ← [05 Regularization](../05-feature-selection-and-regularization/): L2 on `Z` and `W`, L1 for sparse PCA.
- → [10 Mixtures and EM](../10-mixture-models-and-em/): EM fits probabilistic PCA and factor analysis.
- → [12 Autoencoders and VAEs](../12-variational-inference-and-vaes/): PCA is a linear autoencoder.
- → [17 Sequence models](../17-sequence-models-and-transformers/): word2vec embeddings become the input layer of RNNs
  and Transformers.
