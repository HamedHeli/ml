# 20 Evaluation: offline, online, leakage and training-serving skew

**Beyond the slides; builds on CPSC 340 L4, L5 and CPSC 440 lectures 15, 23, 24** · [back to the graph](../README.md)

**What is learned:** nothing new. This folder is about whether the number you report is the number you will get.
[00](../00-learning-fundamentals/) covers the textbook protocol (train / validation / test, cross-validation,
optimization bias). Here: how that protocol gets broken in practice (**data leakage**), how a model that evaluated
well can still fail in production because its inputs changed (**training-serving skew**), and how to evaluate a model
on real users (**online evaluation**).

## Where the slides touch this

| Slides | What they say |
| --- | --- |
| `L4 - Fund of Learning.pdf`, `L5 - Probalistic Classifier.pdf` | the IID assumption, the golden rule (the test set must not influence training), optimization bias, cross-validation |
| `15-img-gen.pdf` | evaluating generative models: log-likelihood pitfalls, FID, precision / recall |
| `23-rep-learning.pdf`, `24-chat-fair.pdf` | memorization, benchmark contamination, domain shift, spurious correlations |

The slides stop at offline evaluation. Everything below on online testing and skew is extra material.

## The ideas

### Offline evaluation: what the held-out number means

An offline score estimates performance on future data **only if** the held-out set was produced the way production
data will be. Three questions decide whether it was:

- **Is the split like deployment?** Random split for IID data. **Grouped** split when rows share an entity (patient,
  user, document): the same entity must not sit on both sides. **Forward-in-time** split when the model will predict
  the future: train on the past, validate on a later window, ideally with a gap.
- **Is the metric the decision?** Accuracy hides class imbalance. **AUC** measures ranking only and is blind to
  calibration. **Log loss / Brier score** measure calibrated probabilities. Precision / recall at the operating
  threshold measures the actual decision. Report uncertainty (bootstrap or a binomial interval), not one number.
- **Was the test set used once?** Each look that changes the model turns test data into validation data
  ([00](../00-learning-fundamentals/), optimization bias). Public benchmarks reused for years suffer from the same
  bias at the scale of a whole field.

### Data leakage

Leakage is when the model, or your evaluation, uses information that **will not be available at prediction time**.
Offline scores come out too good, and the gap shows up only after deployment.

| Kind | What happens | Example | Fix |
| --- | --- | --- | --- |
| **Target leakage** | a feature is caused by, or recorded after, the outcome | "received antibiotics" when predicting infection; "account closed" when predicting churn | for each feature, ask *when* it becomes known; build features as of the prediction time |
| **Preprocessing leakage** | scaling, imputation, feature selection or oversampling are fit on all rows before splitting | selecting the 20 genes most correlated with the label, then cross-validating | put every fitted step inside the training fold (a pipeline) |
| **Group leakage** | the same entity appears in train and test | two scans of one patient; near-duplicate images; one user's sessions | grouped split; de-duplicate |
| **Temporal leakage** | the future informs the past | random split of time series; features joined with "latest" values instead of point-in-time values | forward-chaining split; point-in-time joins |
| **Test-set / benchmark contamination** | the test examples were seen in training | LLM pre-training corpora that contain benchmark questions | hold out fresh data; check n-gram overlap; keep a private test set |

Warning signs: a score that is too good for the problem, one feature dominating the importances, a large drop from
random-split to time-split validation, or a big gap between offline and online results.

### Training-serving skew

**Skew** is a difference between the features (or labels, or data) a model trained on and what it receives when
served, caused by **the system**, not by the world. Common causes:

- **Two code paths.** Features are computed by a batch job (SQL, Spark) for training and re-implemented in a
  low-latency service for serving; units, defaults, tokenization or rounding differ.
- **Different missing-value handling.** Offline imputes the mean, online sends 0 or `null`.
- **Freshness.** Offline features use values as of the event; online they come from a cache refreshed daily.
- **Point-in-time errors.** Training rows were joined with feature values from *after* the event (this is also
  leakage), so serving, which can only see the past, looks different.
- **Feedback loops.** The model's own decisions change the data it is later trained on (a fraud model blocks the
  transactions that would have become positive labels).

**Distribution shift** (covariate shift, label shift, concept drift) is the related problem where the world changes
and the pipeline is correct. The detectors overlap, but the fixes differ: skew is a bug to fix; drift calls for
retraining or reweighting.

**Prevention:** share one feature implementation between training and serving (a feature store or a single library);
**log the features exactly as served** and train on those logs; check feature parity on a sample of requests; monitor
input distributions (PSI, KS tests) and the score distribution against the training baseline.

### Online evaluation

Offline metrics are proxies. The question that matters is causal: *does deploying the model improve the outcome we
care about?* Only an experiment on real traffic answers it.

- **A/B test.** Randomly assign units (users, not requests, when effects persist) to control and treatment, compare
  a pre-registered primary metric, and watch **guardrail metrics** (latency, errors, revenue) for harm. Size the
  test in advance from the baseline rate and the **minimum detectable effect**: `n ≈ (z_{α/2} + z_β)² (p₁q₁ + p₂q₂) / δ²`
  per arm.
- **Pitfalls.** **Peeking** (stopping when the p-value first dips below 0.05) inflates false positives; use a fixed
  horizon or sequential methods (alpha spending, always-valid p-values). Also: novelty and primacy effects, sample-ratio
  mismatch (the split is not 50/50, so assignment is broken), interference between units (marketplaces, social
  networks), and testing many metrics at once.
- **Before the A/B test.** **Shadow mode** runs the new model on live traffic without acting on it: this catches skew
  and latency problems at no risk. **Canary** releases send a small share of traffic first.
- **Interleaving** (for ranking): mix both rankers' results into one list for the same user and count which model's
  results get clicked. It needs 10–100× fewer users than an A/B test.
- **Off-policy evaluation.** Estimate a new policy's reward from logs of the old one with **inverse propensity
  scoring**: `V̂ = (1/n) Σ 1[a_i = π_new(x_i)] r_i / π_old(a_i | x_i)`. It only works if the old policy logged its
  action probabilities and sometimes explored. **Bandits** go a step further and adapt the traffic split while they
  learn.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`leakage.py`](leakage.py) | feature selection outside vs. inside CV on random labels; a target-leaking feature; random vs. forward split on a drifting time series |
| [`skew.py`](skew.py) | a fraud model with a batch and an online feature pipeline; unit, imputation and staleness bugs; feature-parity checks and PSI; real population drift for contrast |
| [`online.py`](online.py) | two-proportion z-test and sample size; power by simulation; peeking in A/A tests; naive replay vs. IPS off-policy evaluation |

```bash
python 20-evaluation/leakage.py      # about 1 s
python 20-evaluation/skew.py
python 20-evaluation/online.py
```

**Leakage** (`leakage.py`)

| Experiment | Leaky evaluation | Honest evaluation | Truth |
| --- | --- | --- | --- |
| random labels, 5000 noise features, top-20 selected | CV accuracy **0.870** (selected on all rows) | 0.530 (selected inside each fold) | 0.5 |
| infection model with an "antibiotics given" feature | test accuracy **0.935** | 0.673 without the feature | 0.600 when deployed: no better than always predicting "no infection" (0.599) |
| 5-NN on date features, predict the next 30 days (MSE, mean of 50 series) | random-split validation **3.11** | forward-split validation 17.0 | 23.1 on the real future |

The leaky selection finds 20 features that happen to correlate with these particular noise labels, and the "test"
folds were part of that search. The antibiotics model is the best model offline and the worst in use.

**Training-serving skew** (`skew.py`, 20,000 served requests; PSI against the training features)

| Scenario | AUC | log loss | mean score / fraud rate | feature parity mismatch | PSI (amount, velocity, age) |
| --- | --- | --- | --- | --- | --- |
| no skew | 0.864 | 0.329 | 0.181 / 0.185 | 0% | 0.00, 0.00, 0.00 |
| amount sent in cents | **0.864** | **2.034** | **0.838** / 0.185 | 100% | **10.6**, 0.00, 0.00 |
| missing velocity filled with 0 | 0.827 | 0.368 | 0.157 / 0.185 | 19% | 0.00, **2.72**, 0.00 |
| velocity from a daily cache | 0.808 | 0.389 | 0.182 / 0.185 | 33% | 0.00, **0.00**, 0.00 |
| population drift (older accounts), no bug | 0.879 | 0.213 | 0.092 / 0.094 | 0% | 0.00, 0.00, **6.12** |

What to take from the table:
- **AUC cannot see the unit bug.** Multiplying by 100 shifts every log-amount equally, so the ranking is unchanged,
  but the scores are miscalibrated (mean score 0.84 for a fraud rate of 0.19). A threshold set offline would flag
  almost everything.
- **Distribution monitoring cannot see the stale cache.** The cached values have the same distribution (PSI 0.00)
  but the wrong value for each request. Only a parity check that computes both pipelines on the same logged requests
  catches it.
- **PSI also fires on real drift**, where the model is still fine. An alert says *something changed*, not that
  something broke.

**Online evaluation** (`online.py`)

| Experiment | Result |
| --- | --- |
| A/B test, conversion 10% vs. 11% | 14,749 users per arm for 80% power (simulated power 0.795); one run: lift 0.0071, 95% CI [0.0001, 0.0142], p = 0.048 |
| A/A test (no real difference), 30 days | false-positive rate **0.044** analysed once at the end, **0.275** when checked daily and stopped at the first p < 0.05 |
| off-policy CTR of a personalised policy, from logs of an ε-greedy policy | truth 0.145; naive replay of matching rows **0.104**; IPS **0.146** |

The single A/B run is a reminder of what 80% power means: the true lift is 0.010, the estimate is 0.007, and the
result was barely significant. Naive replay is biased because the old policy rarely showed item 2, so the type-1 users
who benefit most from the new policy are under-represented among the matching rows; IPS reweights them by
`1 / π_old`.

## Knowledge gap

**What the slides leave out**

- Everything in this folder except the train / validation / test protocol: leakage beyond "don't touch the test set,"
  any discussion of deployment, and online experiments.
- **Calibration**: reliability diagrams, expected calibration error, Platt scaling and isotonic regression; why AUC
  and calibration are separate questions (shown by the unit bug above).
- **Uncertainty in metrics**: bootstrap intervals, paired tests for comparing two models on one test set
  (McNemar, paired bootstrap), and the variance of cross-validation.
- **Variance reduction in A/B tests** (CUPED: regress out a pre-experiment covariate), sequential testing, and
  multiple-testing corrections.
- **Causal inference** behind online tests: potential outcomes, why randomization identifies the effect, and what
  breaks under interference.
- **Off-policy estimators** beyond IPS: self-normalized IPS, the direct method, doubly robust estimators, and the
  variance problems when propensities are small.
- **Monitoring and label delay**: in fraud or credit, labels arrive weeks later, so production quality must be
  inferred from proxies and input monitoring.

**Prerequisites to shore up**

- Hypothesis tests, p-values, confidence intervals, statistical power ([00](../00-learning-fundamentals/) and an intro
  statistics course).
- Importance sampling ([11](../11-monte-carlo-and-mcmc/)): IPS is importance sampling with the old policy as the
  proposal.

**Open questions / further study**

- How do you evaluate a model whose deployment changes the data (performative prediction, feedback loops)?
- When offline and online metrics disagree, which offline metric should you have used?
- Reading: Kaufman et al. 2012, "Leakage in data mining"; Kapoor & Narayanan 2023, "Leakage and the reproducibility
  crisis in ML-based science"; Sculley et al. 2015, "Hidden technical debt in machine learning systems"; Google,
  "Rules of Machine Learning" (rules 29–37 on training-serving skew); Kohavi, Tang & Xu, *Trustworthy Online
  Controlled Experiments*; Li et al. 2011, "Unbiased offline evaluation of contextual-bandit-based news article
  recommendation"; Chip Huyen, *Designing Machine Learning Systems*, ch. 6–9.

## Connections

- ← [00 Learning fundamentals](../00-learning-fundamentals/): the IID assumption and the golden rule; leakage and skew
  are the ways they break in practice.
- ← [05 Feature selection](../05-feature-selection-and-regularization/): feature selection is a training step, so it
  belongs inside cross-validation.
- ← [11 Monte Carlo](../11-monte-carlo-and-mcmc/): inverse propensity scoring is importance sampling.
- ← [18 Diffusion](../18-diffusion-and-image-generation/): evaluating generative models, where held-out likelihood and
  FID can both mislead.
- ← [19 LLMs](../19-representation-learning-and-llms/): benchmark contamination is test-set leakage, and domain shift
  is the world-side version of skew.
