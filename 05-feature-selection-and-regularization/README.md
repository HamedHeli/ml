# 05 Feature selection and regularization

**CPSC 340 L14 (complexity penalties), L15, L16, L17, L18 (first part)** · [back to the graph](../README.md)

**What is learned:** a weight vector that fits the data *and* stays small or sparse, so that the model uses only the
features it needs and generalizes.

## Lectures

| Slides | Covers |
| --- | --- |
| `L14 - Robust Regression.pdf` (second part) | finding the "true" model, complexity penalties |
| `L15 - Feature Selection.pdf` | feature relevance, search and score, forward selection, AIC / BIC, why feature selection is ambiguous (collinearity, confounding) |
| `L16 - Regularization.pdf` | L2 regularization (ridge), standardizing features, causality (bonus) |
| `L17 - Non-parametric.pdf` (second part) | regularizing the intercept, L1 regularization |
| `L18 - Linear Classifier.pdf` (first part) | L1 vs. L2 sparsity geometry, combined L0/L1/L2 view |

## The models

### Search and score feature selection

Define `score(S) = training error with features S + complexity penalty` and search over subsets `S`. Exhaustive
search is `2^d` fits, so use **forward selection**: start empty, add the single feature that lowers the score most,
stop when none helps. Penalties: **AIC** adds `|S|`, **BIC** adds `½ log(n) |S|` (to a negative log-likelihood).
BIC penalizes more and selects fewer features; it is consistent (finds the true model as `n → ∞`), AIC is not but
predicts well.

Feature selection is ambiguous: with collinear features any one of a correlated group can stand in for the others;
a feature can be "relevant" only jointly with another; and correlation is not causation.

### L2 regularization (ridge)

`f(w) = ½‖Xw − y‖² + (λ/2)‖w‖²`, solved by `w = (XᵀX + λI)⁻¹Xᵀy`. The matrix is always invertible, so the solution
is unique even when `d > n`. It shrinks weights toward 0, which lowers the variance (`E_gap`) at the cost of some bias.
It never sets weights exactly to zero. Do not regularize the intercept, and **standardize features** first (using
training statistics only), since the penalty treats every weight the same.

### L1 regularization (lasso)

`f(w) = ½‖Xw − y‖² + λ‖w‖₁`. The kink of `|w_j|` at 0 means that for large enough `λ` the optimum puts many weights
**exactly** at 0: regularization and feature selection in one convex problem. It is not differentiable, so here it is
solved by **proximal gradient (ISTA)**: a gradient step on the squared error, then soft-thresholding
`w_j ← sign(w_j) max(|w_j| − αλ, 0)`. With correlated features lasso tends to pick one arbitrarily; the elastic net
(L1 + L2) keeps groups together.

## Code in this folder

[`regularization.py`](regularization.py): ridge (closed form), lasso (ISTA), forward selection with AIC / BIC,
standardization.

```bash
python 05-feature-selection-and-regularization/regularization.py
```

Sparse truth: 5 relevant features out of 50, `n = 100`, noise σ = 0.5.

| method | test MSE | nonzero weights | which features |
| --- | --- | --- | --- |
| least squares | 0.250 | 50 | all |
| ridge, λ = 1 / 10 / 100 | 0.278 / 0.903 / 6.61 | 50 | all |
| lasso, λ = 1 | 0.118 | 37 | |
| lasso, λ = 10 | **0.121** | 8 | {0–4} + 3 spurious |
| lasso, λ = 50 | 1.80 | 5 | exactly {0–4}, but over-shrunk |
| forward selection, AIC | | 16 | {0–4} + 11 spurious |
| forward selection, BIC | | 7 | {0–4} + 2 spurious |

With a sparse truth and strong signals, ridge only hurts (it shrinks the five real weights and keeps all 45 noise
features); lasso helps. The λ = 50 lasso finds exactly the right support but its shrunken weights predict badly,
which is why people refit least squares on the lasso's support ("relaxed lasso").

## Knowledge gap

**What the slides leave out**

- **Choosing λ**: the slides use validation; the regularization *path* (LARS / coordinate descent over a grid of λ
  with warm starts) is what libraries actually compute.
- **Optimization for L1**: subgradients, proximal operators in general, coordinate descent (what scikit-learn and
  glmnet use), and the dual view.
- **Ridge in terms of the SVD**: `w = V diag(s/(s² + λ)) Uᵀy`, shrinking directions of low variance most; effective
  degrees of freedom `Σ s²/(s² + λ)`.
- **Recovery guarantees for lasso**: support recovery needs the irrepresentable / restricted eigenvalue conditions,
  which fail exactly when features are highly correlated (the ambiguity the slides illustrate).
- **Post-selection inference**: p-values computed after selecting features are invalid without correction
  (selective inference, Berk et al. 2013).
- Group lasso, elastic net, non-convex penalties (SCAD, MCP) that reduce lasso's shrinkage bias.
- Causality is a bonus slide: regression coefficients are not causal effects without assumptions about confounders
  (backdoor criterion, see the DAGs in [14](../14-graphical-models/)).

**Prerequisites to shore up**

- Norms and their unit balls (why the L1 ball has corners on the axes).
- Subgradients of `|w|`.

**Open questions / further study**

- Implicit regularization: early stopping and SGD act like L2 penalties in linear models; what do they do in deep
  networks?
- Reading: Hastie, Tibshirani & Wainwright, *Statistical Learning with Sparsity*, ch. 2–5; *ESL* ch. 3.

## Connections

- ← [04 Linear regression](../04-linear-regression/): the base loss; RBF bases need L2 regularization to work.
- → [06 Linear classifiers](../06-linear-classifiers-and-kernels/): the same penalties on classification losses; L2 +
  hinge is the SVM.
- → [07 MLE and MAP](../07-density-estimation-mle-map/) and [08](../08-gaussians-and-bayesian-learning/): L2 is MAP with
  a Gaussian prior, L1 is MAP with a Laplace prior, and the BIC is a large-sample approximation to the marginal
  likelihood that empirical Bayes maximizes.
