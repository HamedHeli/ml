# 02 Naive Bayes and k-nearest neighbours

**CPSC 340 L5, L6, L7 · CPSC 440 lecture 3** · [back to the graph](../README.md)

**What is learned:** naive Bayes learns a *generative* model `p(x, y)` from counts; kNN learns nothing and keeps the
training set (the standard non-parametric model).

## Lectures

| Slides | Covers |
| --- | --- |
| `L5 - Probalistic Classifier.pdf` (second half) | spam filtering, Bayes rule, the naive Bayes assumption |
| `L6 - Non Parametric Model.pdf` | Laplace smoothing, decision theory, kNN, parametric vs. non-parametric, consistency, curse of dimensionality, ROC / precision-recall (bonus) |
| `L7 - Ensemble.pdf` (first half) | norms as distances, kNN for OCR, data augmentation for invariance |
| `3-generative.pdf` (CPSC 440) | product of Bernoullis, generative classifiers, naive Bayes as conditional density estimation, expected cost |

## The models

### Naive Bayes

A **generative classifier**: model `p(x, y) = p(y) p(x | y)` and classify with Bayes rule,
`p(y | x) ∝ p(y) Π_j p(x_j | y)`. The naive assumption is that features are **conditionally independent given the
class**. For binary features each `p(x_j = 1 | y = c) = θ_jc` is a Bernoulli, so training is counting.

- **Laplace smoothing** `θ_jc = (count + β) / (n_c + 2β)` stops a single unseen word from giving probability 0 to a
  class. In CPSC 440 terms it is the MAP estimate under a Beta prior (see [07](../07-density-estimation-mle-map/)).
- Work in **log space**: a product of 64+ probabilities underflows.
- Binary naive Bayes has a **linear** decision boundary in `x`: the same family as logistic regression, but fit by
  counting instead of by optimizing `p(y | x)` (CPSC 440 lecture 4).

### Decision theory

When mistakes have different costs `C[ŷ, y]`, predict the class with the lowest **expected cost**
`Σ_y C[ŷ, y] p(y | x)`, not the most probable class. This needs calibrated probabilities, which is a reason to use a
probabilistic classifier.

### k-nearest neighbours

To classify `x`, find the `k` closest training points and vote. No training, `O(nd)` per prediction.

- `k` controls complexity: `k = 1` has zero training error; large `k` smooths.
- **Non-parametric**: the model's size grows with `n`. kNN is **consistent**: with `k → ∞` and `k/n → 0`, its error
  converges to the best possible (Bayes) error.
- **Curse of dimensionality**: in high dimension all points are nearly equally far apart, so "nearest" carries less
  information and the data needed to "fill the space" grows exponentially.
- Distances are only as good as the features: standardize them, and choose the norm (L1, L2, L∞) to suit the data.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`naive_bayes.py`](naive_bayes.py) | Bernoulli naive Bayes with Laplace smoothing in log space; minimum-expected-cost prediction |
| [`knn.py`](knn.py) | vectorized squared distances (`‖x‖² + ‖z‖² − 2xᵀz`), batched kNN, curse-of-dimensionality experiment |

```bash
python 02-naive-bayes-and-knn/naive_bayes.py
python 02-naive-bayes-and-knn/knn.py
```

| Result (8×8 digits, 1200 train / 597 test) | |
| --- | --- |
| Bernoulli naive Bayes on binarized pixels, β = 0 (MLE) / β = 1 | test error 0.109 / 0.111; with β = 0, 0.3% of test digits get probability 0 under every class |
| "is it a 3?" as spam, deleting real mail costs 10× missing spam | most-probable rule: total cost 299; minimum-expected-cost rule: 133 |
| kNN, k = 1 / 3 / 10 / 50 | test error 0.012 / 0.012 / 0.023 / 0.059 |
| nearest / farthest distance among 1000 uniform points, d = 2 / 10 / 100 / 1000 | 0.006 / 0.29 / 0.70 / 0.90 |

kNN beats every tree and the random forest from [01](../01-decision-trees-and-ensembles/) here, because pixel
distances are meaningful for these small, centred digits. That stops being true once images are shifted (see
[16](../16-convolutional-networks/)).

## Knowledge gap

**What the slides leave out**

- **When naive Bayes works despite a false assumption.** Its *classifications* can be good even when its
  *probabilities* are badly overconfident (Domingos & Pazzani 1997); its probabilities should not be used for
  decision theory without calibration.
- **Generative vs. discriminative asymptotics** (Ng & Jordan 2001): naive Bayes reaches its (higher) asymptotic error
  with far fewer examples than logistic regression. CPSC 440 lecture 4 states the trade-off; the result explains it.
- Multinomial naive Bayes for word *counts* and TF-IDF features, the standard text variants.
- **Fast nearest-neighbour search**: k-d trees, ball trees, locality-sensitive hashing and approximate methods
  (HNSW, FAISS) that make kNN usable at scale, and are the retrieval step of modern RAG systems.
- **Metric learning**: learning the distance itself (Mahalanobis, siamese / contrastive embeddings; see
  [19](../19-representation-learning-and-llms/)).
- Evaluation under class imbalance is a bonus slide only: precision/recall, ROC-AUC vs. PR-AUC, and why accuracy
  misleads.

**Prerequisites to shore up**

- Bayes rule and conditional independence (written out, not just named).
- Vectorized linear algebra; the norm identities used for the distance trick.

**Open questions / further study**

- Can we get kNN's flexibility with a learned representation instead of raw pixels? That is the theme of
  [09](../09-latent-factor-models/), [12](../12-variational-inference-and-vaes/) and [19](../19-representation-learning-and-llms/).
- Reading: Murphy, *PML1* ch. 9 and 16; *ESL* ch. 13.

## Connections

- → [07 Density estimation](../07-density-estimation-mle-map/): naive Bayes is conditional density estimation with a
  product of Bernoullis; Laplace smoothing is a Beta prior.
- → [06 Linear classifiers](../06-linear-classifiers-and-kernels/): logistic regression is the discriminative twin of
  binary naive Bayes.
- → [08 Gaussians](../08-gaussians-and-bayesian-learning/): Gaussian naive Bayes and GDA replace Bernoullis with
  Gaussians; LDA with identity covariance is a nearest-class-mean classifier.
- → [10 Mixtures](../10-mixture-models-and-em/): a mixture of Bernoullis is "unsupervised naive Bayes."
