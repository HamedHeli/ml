# 06 Linear classifiers, kernels and stochastic gradient descent

**CPSC 340 L18, L19, L20, L21, L22 · CPSC 440 lectures 4–5** · [back to the graph](../README.md)

**What is learned:** a hyperplane `wᵀx = 0` (or one per class) that separates classes, trained by minimizing a
convex surrogate of the 0-1 loss; with kernels, a hyperplane in a huge or infinite feature space.

## Lectures

| Slides | Covers |
| --- | --- |
| `L18 - Linear Classifier.pdf` | classification by regression, the 0-1 loss, perceptron, hinge loss, logistic loss |
| `L19 - More Linear Classifier.pdf` | maximum-margin SVMs, support vectors, converting scores to probabilities (sigmoid), multi-class: one-vs-all, softmax |
| `L20 - Feature Engineering.pdf` | discretization, standardization, bag of words, n-grams, global vs. local (personalized) features, ranking (bonus) |
| `L21 - Kernel Trick.pdf` | polynomial bases, the "other" normal equations, Gram matrix, polynomial and Gaussian-RBF kernels, string kernels, kernel logistic regression |
| `L22 - Stochastic Gradient Descent.pdf` | SGD, step-size schedules, mini-batches, averaging |
| `4-cat-discrim.pdf`, `5-discrim.pdf` (CPSC 440) | discriminative vs. generative, logistic regression as maximum conditional likelihood, MAP = regularization, binary naive Bayes is linear, neural features |

## The models

All predict `sign(wᵀx)` (or `argmax_c w_cᵀx`). They differ in the loss that trains `w`:

| loss on margin `m = y wᵀx` | name | notes |
| --- | --- | --- |
| `[m ≤ 0]` | 0-1 | what we care about; non-convex and flat, so gradient-free |
| `max(0, −m)` | perceptron | SGD on it is the perceptron algorithm; converges iff the data are separable |
| `max(0, 1 − m)` | hinge | with `(λ/2)‖w‖²` it is the **SVM**: maximizes the margin `1/‖w‖`; only "support vectors" with `m ≤ 1` matter |
| `log(1 + exp(−m))` | logistic | smooth and convex; `p(y | x) = σ(y wᵀx)`, so it gives probabilities |
| `−w_yᵀx + log Σ_c exp(w_cᵀx)` | softmax | multi-class logistic regression; trains all classes jointly (unlike one-vs-all) |

**Logistic regression, probabilistically** (CPSC 440): it is the MLE of the conditional model
`p(y | x, w) = σ(y wᵀx)`; adding `(λ/2)‖w‖²` is MAP with a `N(0, I/λ)` prior. It models only `p(y | x)`, the
**discriminative** counterpart of naive Bayes, which models `p(x, y)`.

### The kernel trick

L2-regularized least squares with features `Z = φ(X)` can be solved two ways:
`w = (ZᵀZ + λI)⁻¹Zᵀy` (cost in the number of features `k`) or `v = (ZZᵀ + λI)⁻¹y`, `w = Zᵀv` (cost in `n`).
Predictions then need only inner products: `ŷ = K̃ v` with `K = ZZᵀ`. If `k(x, x') = φ(x)ᵀφ(x')` can be computed
directly, the features never need to exist:

- polynomial `(1 + xᵀx')^p`: all monomials up to degree `p`, which would be `O(d^p)` features;
- Gaussian RBF `exp(−‖x − x'‖² / 2σ²)`: an infinite-dimensional basis;
- kernels on strings, graphs and images, for data that are not vectors at all.

A function is a valid kernel iff every Gram matrix it produces is positive semi-definite (Mercer). Any
L2-regularized linear model whose optimum has the form `w = Zᵀu` (the representer theorem) can be kernelized, e.g.
kernel logistic regression and kernel SVMs.

### Stochastic gradient descent

Use the gradient of one random example (or a mini-batch) per step: `O(d)` per iteration instead of `O(nd)`. It is an
unbiased estimate of the full gradient, but its variance does not vanish at the optimum, so with a constant step SGD
bounces around in a ball; it converges only with decreasing steps (e.g. `α_t ∝ 1/√t`) or with iterate averaging.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`linear_classifiers.py`](linear_classifiers.py) | perceptron; logistic, hinge and softmax losses with gradients; gradient descent with Armijo backtracking; SGD with constant / `1/√t` steps and iterate averaging |
| [`kernels.py`](kernels.py) | polynomial and RBF kernels, an explicit degree-2 feature map that checks the trick, kernel ridge regression, kernel logistic regression over the dual weights |

```bash
python 06-linear-classifiers-and-kernels/linear_classifiers.py
python 06-linear-classifiers-and-kernels/kernels.py
```

| Result (saved run) | |
| --- | --- |
| "is it an 8?" on digits: perceptron / SVM / logistic, test error | 0.052 / **0.034** / 0.035 |
| softmax regression, all 10 digits | test error 0.022 |
| SGD, 10 epochs, L2-logistic objective (optimum 153.4): constant step last / averaged | 486.6 / 391.2 |
| SGD, `1/√t` step, last / averaged | 210.3 / **167.8** |
| degree-2 kernel vs. explicit features | predictions agree to 1e-11 |
| concentric circles, linear / poly-2 / poly-4 / RBF kernel logistic regression, test error | 0.495 / 0.005 / 0.005 / 0.005 |

## Knowledge gap

**What the slides leave out**

- **The SVM dual** and why only support vectors have nonzero dual weights; the KKT conditions; SMO, the classic
  solver. The slides reach the margin view but not the Lagrangian derivation that makes kernels natural for SVMs.
- **Reproducing-kernel Hilbert spaces** and the representer theorem proved, not just used; what function class an
  RBF kernel actually defines and how `σ` controls its smoothness.
- **Scaling kernels**: the `n × n` Gram matrix is `O(n²)` memory and `O(n³)` to solve. Nyström approximations and
  random Fourier features (Rahimi & Recht 2007) turn kernels back into explicit low-dimensional features.
- **SGD theory**: the `O(1/√T)` rate for convex problems, `O(1/T)` with strong convexity and decreasing steps,
  variance-reduced methods (SAG, SVRG) that get linear rates on finite sums, and the adaptive methods (AdaGrad,
  Adam) that the neural-network folders use.
- **Calibration**: SVM scores are not probabilities; Platt scaling fits a sigmoid on top.
- **Multi-class reductions** beyond one-vs-all and softmax (all-pairs, error-correcting output codes) appear only in
  bonus slides; structured prediction (predicting sequences or graphs) is deferred to CRFs in
  [14](../14-graphical-models/).

**Prerequisites to shore up**

- Lagrange multipliers and duality for constrained convex problems.
- Positive semi-definite matrices (for valid kernels).

**Open questions / further study**

- Infinitely wide neural networks trained by gradient descent behave like kernel machines (the neural tangent
  kernel). When does feature learning beat a fixed kernel?
- Reading: Schölkopf & Smola, *Learning with Kernels*; Bottou, Curtis & Nocedal 2018, "Optimization methods for
  large-scale machine learning."

## Connections

- ← [04 Linear regression](../04-linear-regression/) and [05 Regularization](../05-feature-selection-and-regularization/):
  same linear model and penalties, new losses.
- ↔ [02 Naive Bayes](../02-naive-bayes-and-knn/): generative vs. discriminative versions of the same linear boundary.
- → [07 Exponential families](../07-density-estimation-mle-map/): logistic and softmax regression are generalized linear
  models.
- → [11 Monte Carlo](../11-monte-carlo-and-mcmc/): Bayesian logistic regression, the running example for approximate
  inference.
- → [15 Neural networks](../15-neural-networks/): a softmax regression on *learned* features; trained with SGD.
