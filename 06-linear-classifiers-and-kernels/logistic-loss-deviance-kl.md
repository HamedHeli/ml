# Logistic regression's loss: cross-entropy, likelihood, deviance and KL divergence

[back to the folder](README.md) · code that checks every identity below:
[`logistic_deviance.py`](logistic_deviance.py) · related: [exponential families and GLMs](../07-density-estimation-mle-map/README.md#exponential-families),
[KL divergence in variational inference](../12-variational-inference-and-vaes/README.md)

**Where this comes from.** A chat Hamed had about logistic regression as a GLM. It started from "the prediction is
`1/(1 + exp(bx))`, so how do you compare that number with 0 and 1?" and went on to cross-entropy, entropy, KL
divergence and deviance. This page keeps the conclusions, puts them in order, and fixes the places where the chat was
wrong or loose (listed in [Corrections](#corrections-to-the-original-chat)).

**The short version.** Four names, one objective, up to sign and scale:

| Quantity | For `n` binary labels | Optimized by |
| --- | --- | --- |
| log-likelihood `ℓ(β)` | `Σ [yᵢ log p̂ᵢ + (1 − yᵢ) log(1 − p̂ᵢ)]` | maximizing |
| negative log-likelihood (NLL) | `−ℓ(β)` | minimizing |
| (binary) cross-entropy, "log-loss" | `−ℓ(β) / n` | minimizing |
| deviance | `2(ℓ_sat − ℓ(β))`, which for 0/1 labels is `−2ℓ(β)` | minimizing |

All four give the same `β̂`. They differ in what they let you *say* about the fit: cross-entropy connects to
information theory, deviance connects to hypothesis tests.

## 1. The model and the loss

Logistic regression models the probability of the positive class:

```
p̂(x) = P(y = 1 | x) = σ(βᵀx) = 1 / (1 + exp(−βᵀx))
```

Note the **minus sign** in the exponent. With `1/(1 + exp(+βᵀx))`, increasing `βᵀx` would push `p̂` toward 0 instead
of 1. (That model is not wrong, it is the same model with `β` negated, but every sign afterwards would flip.)

The label `y ∈ {0, 1}` is a draw from `Bernoulli(p̂)`, whose probability mass is `p̂^y (1 − p̂)^(1−y)`. Taking the log
and negating gives the loss for one example:

```
L(y, p̂) = −[ y log p̂ + (1 − y) log(1 − p̂) ]
```

**How this "compares p̂ with 0 and 1".** Because `y` is exactly 0 or 1, it switches off one of the two terms:

| True label | Loss | `p̂ → 1` | `p̂ → 0` |
| --- | --- | --- | --- |
| `y = 1` | `−log p̂` | → 0 (confident and right) | → ∞ (confident and wrong) |
| `y = 0` | `−log(1 − p̂)` | → ∞ (confident and wrong) | → 0 (confident and right) |

So there is no thresholding at 0.5 inside the loss. The loss scores the probability itself, and punishes confident
mistakes without bound. (The term the chat began with, "cross-validation", is unrelated: it is the resampling scheme
for estimating test error in [00](../00-learning-fundamentals/). The loss is **cross-entropy**.)

Over a dataset, the cost is the average:

```
J(β) = −(1/n) Σᵢ [ yᵢ log p̂ᵢ + (1 − yᵢ) log(1 − p̂ᵢ) ],    p̂ᵢ = σ(βᵀxᵢ)
```

`J` is convex in `β` (the Bernoulli is an [exponential family](../07-density-estimation-mle-map/README.md#exponential-families),
so its NLL is convex in the natural parameter `βᵀx`), and its gradient is `(1/n) Xᵀ(p̂ − y)`.

**Same thing as the [README](README.md)'s logistic loss.** This folder writes the loss as `log(1 + exp(−y wᵀx))` with
labels `y ∈ {−1, +1}`. Recode `y₀₁ = (y₊₋ + 1)/2` and the two are equal term by term, because
`−log σ(s) = log(1 + e^(−s))` and `−log(1 − σ(s)) = −log σ(−s) = log(1 + e^(s))`. The script checks it on data (they
agree to every printed digit).

## 2. Log-likelihood and cross-entropy

**Cross-entropy** between a "true" distribution `P` and a model `Q` is

```
H(P, Q) = −E_{z ~ P}[ log Q(z) ] = −Σ_z P(z) log Q(z)
```

For classification, the model is a conditional `Q_β(y | xᵢ)`, and for each example the "true" distribution is the
observed label as a point mass, `Pᵢ = onehot(yᵢ)`. Then

```
(1/n) Σᵢ H(Pᵢ, Q_β(· | xᵢ)) = −(1/n) Σᵢ log Q_β(yᵢ | xᵢ)  =  average NLL.
```

The same identity holds for any model fitted by maximum likelihood: replace the unknown data distribution by the
empirical one, and cross-entropy becomes the average NLL. Softmax regression (categorical), Poisson regression, and
least squares (the Gaussian NLL is `(1/2σ²)·SSE + const` for fixed `σ`) are all instances. This is the
[MLE ↔ loss correspondence](../07-density-estimation-mle-map/README.md#mle-and-map-for-supervised-learning-cpsc-340-l24)
from CPSC 340 L24.

**Where they differ in practice**

- **Sign and scale.** Log-likelihood is a sum and is maximized; cross-entropy is usually an average and minimized.
- **Soft targets.** Likelihood is defined for observed data, i.e. hard labels. Cross-entropy accepts any target
  distribution `P`: label smoothing (`P = [0.9, 0.1]` instead of `[1, 0]`), knowledge distillation (`P` = a teacher's
  softmax), or grouped proportions (section 4).
- **Log base.** Statistics uses natural logs (nats); information theory often uses `log₂` (bits). Only the scale changes.

## 3. Entropy and KL divergence

Add and subtract `Σ P log P`:

```
H(P, Q)  =  H(P)  +  KL(P ‖ Q)
cross-entropy  =  entropy of the targets  +  model error
```

where `H(P) = −Σ P(z) log P(z)` and `KL(P ‖ Q) = Σ P(z) log(P(z)/Q(z)) ≥ 0`, with equality only when `Q = P`.

- **Entropy is not the log-likelihood.** `H(P)` has no model in it. It is the irreducible uncertainty of the targets,
  a constant during training. Minimizing cross-entropy over `β` is therefore the same as minimizing `KL(P ‖ Q_β)`.
  `H(P)` equals the NLL only when the model is perfect, `Q = P`, so that the KL term is 0.
- **Hard labels have zero entropy.** For a one-hot `P`, the term at `p = 1` is `1·log 1 = 0`. The term at `p = 0` is
  `0·log 0`, which is not defined by substitution (`0·(−∞)`). It is set to 0 by convention, justified by the limit
  `lim_{p→0⁺} p log p = lim log p / (1/p) = lim (1/p)/(−1/p²) = lim (−p) = 0` (L'Hôpital). So `H(onehot) = 0`, and
  the per-example cross-entropy *is* the KL from the label to the model. Its floor is 0.
- **Soft labels have a positive floor.** With `P = [0.9, 0.1]`, `H(P) ≈ 0.325` nats, and even the perfect model
  `Q = P` has cross-entropy `H(P) > 0`. The script shows this with label smoothing: cross-entropy at the perfect model
  equals `H(P)` exactly, not 0.

## 4. Deviance

In a GLM (R's `glm(..., family = binomial)`), the fit is reported as **deviance**:

```
D = 2 ( ℓ_sat − ℓ(β̂) )
```

The **saturated model** has one free mean per observation, so it reproduces the data exactly: `p̂ᵢ = yᵢ` for 0/1
labels, `p̂_g = y_g / m_g` for grouped counts, `μ̂ᵢ = yᵢ` for Poisson. `D ≥ 0`, and it plays the role that the residual
sum of squares plays in least squares (for a Gaussian GLM, `D` is the SSE, and `D / σ²` is the scaled deviance).

**Deviance is 2 × summed KL from the saturated to the fitted distribution.** For each observation, the
log-likelihood ratio between the saturated and fitted members of the family is a KL divergence between them:

```
D = 2 Σᵢ KL( f(· ; saturated meanᵢ) ‖ f(· ; fitted meanᵢ) )
```

| Data | `ℓ_sat` | Deviance |
| --- | --- | --- |
| ungrouped 0/1 labels | `Σ log 1 = 0` | `D = −2ℓ(β̂) = 2 × NLL = 2n × cross-entropy`, and `= 2 Σ KL(onehot(yᵢ) ‖ Bern(p̂ᵢ))` |
| grouped binomial (`y_g` successes of `m_g`) | `≠ 0` | `D = 2 Σ_g m_g KL( Bern(y_g/m_g) ‖ Bern(p̂_g) )` |
| Poisson counts | `≠ 0` | `D = 2 Σ [yᵢ log(yᵢ/μ̂ᵢ) − (yᵢ − μ̂ᵢ)] = 2 Σ KL( Pois(yᵢ) ‖ Pois(μ̂ᵢ) )` |

The script checks all three to every printed digit.

**Watch the sign.** Probabilities are at most 1, so for discrete data `ℓ ≤ 0`, and NLL and cross-entropy are `≥ 0`.
Deviance must be positive, so for 0/1 data it is `−2 × log-likelihood = +2 × NLL`. Writing "−2 × cross-entropy" or
"−2 × negative log-likelihood" gives a negative number.

**Grouping changes the deviance, not the fit.** Group the same 400 binary labels by their 5 distinct `x` values. The
binomial likelihood differs from the Bernoulli one only by the constant `Σ log C(m_g, y_g)`, so `β̂` is identical. But
the saturated model changes (5 free probabilities instead of 400), and the deviance falls from 467.1 to 2.58.
Differences of deviances between two models fitted to the same data are the same either way, because `ℓ_sat` cancels.
Only those differences carry meaning across groupings.

**Why the factor 2: Wilks' theorem.** For nested models, the likelihood-ratio statistic
`D_small − D_big = 2(ℓ_big − ℓ_small)` is asymptotically `χ²` with degrees of freedom equal to the number of extra
parameters, when the smaller model is true. The factor 2 is what makes the limit a standard `χ²`. The script simulates
2000 datasets with a true slope of 0: the test rejects at the 5% level in 5.4% of them.

**But the residual deviance of 0/1 data is not a goodness-of-fit test.** Comparing a single model's `D` with
`χ²_{n−p}` needs the saturated model to have a fixed number of parameters as `n` grows. With ungrouped binary data it
has `n` of them, so the asymptotics fail, and `D` mostly reflects the `p̂ᵢ` themselves rather than the fit (in the
demo it is 467 against `n − p = 398` for a correctly specified model). Grouped data with large `m_g` do support the
test (demo: 2.58 on 3 df). For ungrouped data, use Hosmer–Lemeshow or calibration plots
([20 Evaluation](../20-evaluation/)).

## Corrections to the original chat

Most of the chat was correct. These are the places that were wrong, garbled or loose:

1. **Sigmoid sign.** The opening formula `1/(1 + exp(bx))` has the wrong sign for the usual parameterization. The chat
   fixed this; it is kept in section 1.
2. **"−2 × cross-entropy" / "−2 × negative log-likelihood".** Both are negative. Deviance for 0/1 data is
   `−2ℓ = +2 × NLL`. The chat fixed this too.
3. **"Entropy is the same as the log-likelihood."** No: cross-entropy is the NLL; entropy is the model-free part. The
   chat fixed this.
4. **`ℓ_sat = −H(P)` is not true in general.** The chat derived "deviance = 2 × KL" by substituting
   `ℓ_sat = −H(P)` and `ℓ_fit = −H(P, Q)`. That works for 0/1 and grouped-proportion data, where the target `P` is a
   point mass or a Bernoulli built from the data. It fails for Poisson: `ℓ_sat = Σ log Pois(yᵢ; yᵢ)` is the
   log-probability of the observed count, not minus the entropy of `Pois(yᵢ)` (the demo gets −385.4 against −495.5).
   The conclusion survives for a different reason. For any exponential family, the log-likelihood ratio between two
   members at the observed `y` equals the KL between the member whose mean is `y` and the other member. So
   `D = 2 Σ KL(saturatedᵢ ‖ fittedᵢ)` holds for binomial and Poisson alike, as in the table in section 4.
5. **Grouped data: `ℓ_sat ≠ 0`.** In the exported chat this line reads "ℓ_sat = 0, so subtracting it matters", which
   contradicts itself. The `≠` was most likely lost in the export. For grouped binomial or Poisson data the saturated
   log-likelihood is nonzero, so `D ≠ 2 × NLL`, but `D = 2 Σ KL` still holds.
6. **"Log-likelihood is negative" holds only for discrete data.** Densities can exceed 1, so a continuous model's
   log-likelihood can be positive and its "cross-entropy" (differential) negative. The demo gets `ℓ = +324` for 100
   Gaussian points with `σ = 0.01`. The positivity argument for deviance still holds there, because `D` is a
   difference `ℓ_sat − ℓ_fit ≥ 0`, not a single log-likelihood.
7. **Two different `P`s.** The chat used `P_emp` (the empirical distribution over the whole dataset) to show
   cross-entropy = average NLL, and then said hard labels give `H(P) = 0`. These are different distributions. For
   classification the `P` with zero entropy is the per-example label distribution `onehot(yᵢ)`. The empirical
   distribution over `n` distinct data points has entropy `log n`, not 0, which is why the floor of an unconditional
   density model's average NLL is not 0.

## Knowledge gap

**What this leaves out**

- **Separable data.** If a hyperplane separates the classes, the logistic NLL keeps decreasing as `‖β‖ → ∞`, so the
  MLE does not exist. R warns "fitted probabilities numerically 0 or 1 occurred". L2 regularization
  ([05](../05-feature-selection-and-regularization/)) or Firth's penalized likelihood gives a finite answer.
- **Newton's method on this loss is IRLS**, iteratively reweighted least squares with weights `p̂(1 − p̂)`. That is how
  `glm` fits it, and what `fit_logistic` in the script does.
- **Proper scoring rules.** Log-loss and the Brier score are *proper*: they are minimized in expectation by the true
  probability. Accuracy and the 0-1 loss are not, which is why we train on log-loss and only threshold afterwards.
- **Deviance residuals** `sign(yᵢ − p̂ᵢ)·√dᵢ`, and over-dispersion (a quasi-likelihood scale factor `D / (n − p)` for
  grouped binomial and Poisson data).

**Prerequisites to shore up**

- Bernoulli and binomial likelihoods; L'Hôpital's rule (for `0 log 0`).
- Jensen's inequality (for `KL ≥ 0`); the [exponential-family](../07-density-estimation-mle-map/README.md#exponential-families)
  identities `∇A = E[T]`, `∇²A = Cov[T]`.

**Further reading**

- McCullagh & Nelder, *Generalized Linear Models*, ch. 2 and 4 (deviance, the saturated model, IRLS).
- Agresti, *Categorical Data Analysis*, ch. 4–5 (grouped vs. ungrouped deviance, goodness of fit).
- Cover & Thomas, *Elements of Information Theory*, ch. 2 (entropy, cross-entropy, KL).
- Hastie, "A closer look at the deviance" (*The American Statistician*, 1987), on deviance as a KL divergence.
