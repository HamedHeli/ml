# 00 Learning fundamentals

**CPSC 340 L1, L2, L4, L5** · [back to the graph](../README.md)

**What is learned:** nothing yet. This folder is about *how we know a learned model is any good*: the
train/validation/test protocol, the fundamental trade-off and cross-validation. Every other folder relies on it.

## Lectures

| Slides | Covers |
| --- | --- |
| `L1 - Syllabus.pdf` | course logistics, data mining vs. machine learning, applications and failures |
| `L2 - EDA.pdf` | feature representations, summary statistics, visualization, supervised learning notation |
| `L4 - Fund of Learning.pdf` | training vs. test error, IID assumption, the fundamental trade-off, double descent |
| `L5 - Probalistic Classifier.pdf` (first half) | validation error, optimization bias, cross-validation, no free lunch |

## Key ideas

- **The IID assumption.** Training and test examples are independent draws from the same distribution. Every
  guarantee below depends on it, and most real failures (distribution shift, leakage, time series) break it.
- **The fundamental trade-off.** `E_test = E_train + E_gap`. Simple models have a large `E_train` and small `E_gap`;
  complex models the reverse. You cannot see `E_test`, so you estimate it with data the model never touched.
- **Train / validation / test.** Fit parameters on training data, choose hyper-parameters (depth, `k`, `λ`) on
  validation data, report once on test data. Using the test set to make choices turns it into a validation set.
- **Optimization bias.** The best of many validation scores is optimistic: with enough models, one looks good by
  chance. The bias grows with the number of models compared and shrinks with the size of the validation set.
- **k-fold cross-validation.** Split the data into `k` folds; train on `k-1`, validate on the remaining one, rotate
  and average. Uses all the data for both purposes at a cost of `k` fits.
- **No free lunch.** Averaged over all possible problems, no learner beats any other; a model is only good relative
  to the assumptions it makes about the problems you actually have.
- **Double descent.** For very flexible models (wide nets, interpolating fits) test error can fall again past the
  interpolation threshold, which the classic U-shaped picture does not predict.

## Code in this folder

[`cross_validation.py`](cross_validation.py): k-fold CV written by hand, used to choose a polynomial degree,
plus a 20-line demonstration of optimization bias.

```bash
python 00-learning-fundamentals/cross_validation.py
```

| Result (saved run) | |
| --- | --- |
| degree 0 → 12 polynomial on 60 noisy points of `sin(3x)` | `E_train` falls monotonically from 0.61 to 0.064; `E_cv` bottoms out at 0.095 (degree 7) and rises to 0.31 at degree 12 |
| test MSE of the CV choice | 0.118 (the best degree on test is 3, at 0.099: CV is itself a noisy estimate) |
| best of 1000 random classifiers on 50 validation points | 72% "accuracy" on labels that are pure coin flips |

## Knowledge gap

**What the slides leave out**

- *Why* `E_gap` shrinks with `n`: concentration inequalities (Hoeffding), the union bound over a finite model class,
  and VC dimension / Rademacher complexity for infinite ones. This is what turns "optimization bias" into a formula:
  with `m` models and `n` validation points the bias is about `sqrt(log(m) / n)`.
- Bias–variance decomposition of squared error, which is the regression version of the trade-off.
- Variance of the cross-validation estimate itself, and why folds are not independent (no unbiased estimator of
  the CV variance exists; see Bengio & Grandvalet 2004). Nested CV for reporting a model that was tuned by CV.
- Leakage: preprocessing (standardizing, feature selection, imputation) has to be fit inside each fold. See
  [20 Evaluation](../20-evaluation/) for leakage, training-serving skew and online evaluation.
- Non-IID settings: grouped CV, time-series splits, covariate shift and importance-weighted validation.
- Double descent is shown but not explained; the modern story involves implicit regularization and the
  minimum-norm interpolant.

**Prerequisites to shore up**

- Probability: expectation, variance, independence; the law of large numbers.
- Asymptotic notation, since CV multiplies cost by `k`.

**Open questions / further study**

- When does a model that interpolates noisy training data still generalize ("benign overfitting")?
- Reading: Shalev-Shwartz & Ben-David, *Understanding Machine Learning*, ch. 2–6; Hastie, Tibshirani & Friedman,
  *ESL* ch. 7; Belkin et al. 2019, "Reconciling modern machine learning practice and the bias-variance trade-off."

## Connections

- → every folder: each demo in this repo reports held-out error, not training error.
- → [20 Evaluation](../20-evaluation/): leakage, training-serving skew, and A/B tests, i.e. how this protocol breaks
  in practice and what replaces it after deployment.
- → [05 Feature selection and regularization](../05-feature-selection-and-regularization/): penalties such as BIC
  and `λ‖w‖²` are cheaper substitutes for cross-validation.
- → [08 Gaussians and Bayesian learning](../08-gaussians-and-bayesian-learning/): empirical Bayes chooses
  hyper-parameters from the marginal likelihood without a validation set.
