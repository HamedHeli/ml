# 01 Decision trees and ensembles

**CPSC 340 L3, L7, L8, L23** · [back to the graph](../README.md)

**What is learned:** a program of nested if-statements on single features (a tree), or a vote / sum of many such
programs (an ensemble).

## Lectures

| Slides | Covers |
| --- | --- |
| `L3 - Trees.pdf` | decision stumps, search and score, greedy recursive splitting, accuracy vs. information gain, `O(nd log n)` stump fitting |
| `L7 - Ensemble.pdf` | norms and distances (for kNN, see 02), ensemble methods, voting, stacking, random forests, bootstrap |
| `L8 - Clustering.pdf` (first part) | random-forest ingredients, extremely randomized trees, Bayesian model averaging (bonus) |
| `L23 - Boosting.pdf` | AdaBoost, gradient tree boosting, XGBoost's regularized regression trees |

## The models

### Decision stump and decision tree

A stump is one rule, "if `x_j > t` predict `a`, else predict `b`". Learning is **search and score**: try every
feature `j` and threshold `t` (only the midpoints between sorted values matter) and keep the best score.
Sorting each feature once and sweeping gives `O(n d log n)`.

A tree is learned by **greedy recursive splitting**: fit a stump, split the data, recurse on each side until a depth
limit or a pure leaf. Greedy means a split is judged only on its immediate gain.

- **Accuracy** is a poor split score: a split can make both children purer without changing either child's
  majority label, so accuracy sees no gain and the tree stops. **Information gain** `H(y) − Σ (n_c/n) H(y_c)` (the
  drop in label entropy) rewards that progress.
- Regression trees predict the mean of `y` in each leaf and split to minimize squared error.
- Depth is the complexity knob: shallow trees underfit, deep trees memorize.

### Random forests (averaging)

Train many deep trees, each on a **bootstrap sample** (n draws with replacement, about 63% unique points) and
considering only a **random subset of about √d features** at each split, then take a majority vote. Deep trees have
small `E_train` and large `E_gap`; averaging trees whose errors are only partly correlated reduces variance, which
shrinks `E_gap` without raising `E_train`.

### Boosting

Boosting attacks the other side of the trade-off: combine many models that each **underfit**, fit sequentially so
each one corrects the current ensemble.

- **AdaBoost** (binary): re-weight examples so the ones the ensemble gets wrong count exponentially more, fit a stump
  on the weighted data, add it with weight `α = ½ log((1 − err)/err)`. It is coordinate descent on the exponential
  loss `Σ exp(−y f(x))`. The base learner must beat 50% weighted accuracy (stumps with accuracy score do; deep trees
  leave no errors to boost).
- **Gradient tree boosting**: each new regression tree is fit to the residuals `y − f(x)`, i.e. the negative
  gradient of the squared loss, and added with a small step `η`. XGBoost adds an L2 penalty on leaf values and
  second-order (Newton) steps for other losses, and scales to huge data.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`decision_tree.py`](decision_tree.py) | stump with accuracy / infogain / squared-error scores and `O(nd log n)` sweeping, weighted examples, greedy tree |
| [`random_forest.py`](random_forest.py) | bootstrap + random feature subsets + majority vote |
| [`boosting.py`](boosting.py) | AdaBoost with stumps; gradient-boosted regression trees with shrinkage |

```bash
python 01-decision-trees-and-ensembles/decision_tree.py
python 01-decision-trees-and-ensembles/random_forest.py
python 01-decision-trees-and-ensembles/boosting.py
```

Results on 8×8 digits (1200 train / 597 test) and two toy problems:

| | train error | test error |
| --- | --- | --- |
| tree, depth 4 (infogain) | 0.296 | 0.369 |
| tree, depth 12 (infogain) | 0.000 | 0.168 |
| random forest, 10 trees | 0.001 | 0.065 |
| random forest, 50 trees | 0.000 | **0.028** |
| AdaBoost on a disc, 1 / 50 / 200 stumps | 0.393 / 0.025 / 0.000 | 0.455 / 0.048 / 0.037 |
| gradient boosting on `x sin x`, 10 / 50 / 200 depth-2 trees | MSE 1.26 / 0.065 / 0.039 | 1.00 / 0.018 / 0.021 |

The forest drops test error six-fold relative to one deep tree while both fit the training set perfectly. The
boosting runs show training error falling monotonically. AdaBoost's test error keeps improving even after its
training error reaches zero, while gradient boosting's test error creeps back up between 50 and 200 trees as the
ensemble starts fitting the noise (its training MSE drops below the noise variance of 0.09).

## Knowledge gap

**What the slides leave out**

- **Why greedy works at all.** Finding the optimal tree is NP-hard; nothing guarantees the greedy tree is close.
  Pruning (cost-complexity pruning in CART) is the usual fix for greedy over-growth and is not covered.
- **Other impurity measures and categorical splits**: Gini impurity, gain ratio (C4.5), multi-way splits, missing
  values (surrogate splits).
- **Random-forest theory**: the variance of an average of `B` estimators with pairwise correlation `ρ` is
  `ρσ² + (1 − ρ)σ²/B`, which is exactly why random feature subsets (lower `ρ`) help. Out-of-bag error as free
  validation; permutation feature importance and its biases.
- **Boosting as functional gradient descent** (Friedman 2001) for arbitrary losses (log-loss, Huber, quantile), and
  why AdaBoost's training margin keeps growing after training error hits zero (the margin theory of Schapire et al.).
- **Modern gradient boosting engineering**: histogram splits (LightGBM), ordered target statistics for categorical
  features (CatBoost), early stopping on a validation set, learning-rate vs. number-of-trees trade-off.
- Calibration: tree ensembles give poorly calibrated probabilities; Platt scaling or isotonic regression fix this.

**Prerequisites to shore up**

- Entropy and information (log base, why `H` is maximized by uniform labels).
- Variance of sums of correlated random variables.

**Open questions / further study**

- Why do boosted trees still beat deep networks on most tabular benchmarks (Grinsztajn et al. 2022)?
- Reading: *ESL* ch. 9, 10, 15; Breiman 2001, "Random Forests"; Chen & Guestrin 2016, "XGBoost."

## Connections

- ← [00 Learning fundamentals](../00-learning-fundamentals/): depth, number of trees and `η` are hyper-parameters
  chosen by validation; forests and boosting attack the two terms of the fundamental trade-off.
- → [02 Naive Bayes and kNN](../02-naive-bayes-and-knn/): the other two "first" classifiers of the course.
- → [04 Linear regression](../04-linear-regression/): gradient boosting is gradient descent, with trees as the steps.
