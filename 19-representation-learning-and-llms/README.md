# 19 Representation learning, large language models and fairness

**CPSC 440 lectures 23, 24 · CPSC 340 L35 (LLM part), L36 (societal part)** · [back to the graph](../README.md)

**What is learned:** general-purpose features from unlabelled data, by solving a *pretext task* (predict masked words,
predict the next token, match two views of the same image); then, for chatbots, a policy fine-tuned on demonstrations
and human preferences.

## Lectures

| Slides | Covers |
| --- | --- |
| `23-rep-learning.pdf` | training Transformers at scale, data, progress in language modeling, memorization, BERT-style features for downstream tasks, autoencoders and denoising as pretext tasks, relative-patch, jigsaw and colourization pretext tasks, contrastive learning (SimCLR), what the contrastive loss does, information-theoretic view and its limits, CLIP |
| `24-chat-fair.pdf` | pre-training vs. post-training, supervised fine-tuning and its labour, toxicity filtering and its biases, RLHF (reward models, KL-regularized policy optimization, DPO / GRPO), reward hacking, LLMs as role-play, failures, bias in NLP and vision models, limits of computational fairness, spurious correlations, domain shift, costs (economy, water, energy) |
| `L35 - Transformers.pdf` (second half), `L36 - Generative Modelling.pdf` (first half) | LLMs, few-shot prompting, prompt engineering and jailbreaks, "does the model understand language?"; racially biased and sexist algorithms, energy costs |

## The ideas

### Pretext tasks and transfer

Train on a task whose labels come free with the data, keep the network, and reuse its features: fit a linear
classifier on them (a **linear probe**) or fine-tune. Language: **masked language modeling** (BERT: hide 15% of tokens,
predict them; good features for classification) and **next-token prediction** (GPT: good generators). Vision:
denoising and colourization, predicting the relative position of patches, solving jigsaws, and contrastive learning.

### Contrastive learning

Make two random augmentations of each image; for a batch of `n` images with normalized embeddings, minimize the
**InfoNCE** loss

`−log exp(z_iᵀz_i'/τ) / Σ_j exp(z_iᵀz_j/τ)`,

i.e. classify which of the `n` candidates is your partner view. `log n − loss` lower-bounds the mutual information
between the views, but the slides stress that this does not explain why it works: what the representation keeps is
decided by the **augmentations** (it becomes invariant to whatever they change), and by architecture and
optimization. **CLIP** applies the same loss to (image, caption) pairs, aligning vision and language in one embedding
space; it is the text encoder behind text-to-image diffusion ([18](../18-diffusion-and-image-generation/)).

### From language model to chatbot

1. **Pre-training**: next-token prediction on a huge corpus.
2. **Supervised fine-tuning**: imitate curated (prompt, good response) pairs, much of it written by low-paid
   annotators.
3. **RLHF**: collect pairwise human preferences, fit a reward model `r̂(x, y)`, and optimize the policy to maximize
   `E[r̂(x, y)] − β KL(π ‖ π_base)`. The KL term limits **reward hacking** (maximizing a learned network's output leaves
   the region where it is accurate). DPO and GRPO optimize the same kind of objective without a separate RL loop.

### Failures and fairness

Models learn correlations, not concepts: they are non-robust to domain shift, memorize and regurgitate training data,
inflate benchmark scores through contamination, and absorb social biases from data, labels, and filtering (toxicity
filters flag African-American English and queer content at higher rates). Computational fairness criteria (e.g.
demographic parity, equalized odds) conflict with each other and cannot fix a problem whose framing is wrong. Training
and serving large models has real energy, water and labour costs.

## Code in this folder

[`contrastive.py`](contrastive.py): SimCLR-style contrastive learning in JAX on 8×8 digits. Augmentations are random
one-pixel shifts, 20% pixel dropout and Gaussian noise; the encoder is a 64-128-64-32 MLP; the loss is symmetric
InfoNCE with `τ = 0.2`. Pre-training uses 1400 images with no labels; then a linear probe is trained on 1, 5 or 20
labelled images per class.

```bash
python 19-representation-learning-and-llms/contrastive.py        # about 1 min on a CPU
```

Linear-probe test accuracy (mean of 5 random draws of the labelled set):

| features | 1 label / class | 5 / class | 20 / class |
| --- | --- | --- | --- |
| raw pixels (64-d) | 0.563 | 0.840 | **0.934** |
| PCA (32-d) | 0.351 | 0.748 | 0.909 |
| random encoder (32-d) | 0.457 | 0.706 | 0.838 |
| contrastive encoder (32-d) | **0.730** | **0.874** | 0.923 |

InfoNCE falls from 5.59 to 3.33 nats (batch 256, `log n` = 5.55). With one label per class, the contrastive features are
17 points better than raw pixels; with 20 labels per class raw pixels catch up, since on tiny clean digits pixels are
already good features. Representation learning pays off most when labels are scarce.

## Knowledge gap

**What the slides leave out**

- **Collapse and how to avoid it**: why contrastive methods need negatives, and how non-contrastive methods (BYOL,
  SimSiam, DINO, VICReg) avoid the trivial constant embedding without them.
- **Masked image modeling** (MAE) and joint-embedding predictive architectures, the vision analogues of BERT.
- **Fine-tuning practice**: full fine-tuning vs. linear probing vs. parameter-efficient methods (LoRA, adapters), and
  catastrophic forgetting.
- **RL foundations**: the slides define MDPs and point to other courses; policy gradients, PPO, and the derivation of
  DPO from the KL-regularized objective are not covered.
- **Evaluation of LLMs**: perplexity vs. downstream benchmarks, contamination detection, LLM-as-judge and its biases.
- **Retrieval-augmented generation, tool use and in-context learning**: how deployed systems actually use LLMs, and
  what is known about why few-shot prompting works.
- **Formal fairness**: the impossibility results (Kleinberg et al. 2016; Chouldechova 2017) behind "limits of
  computational fairness," and causal notions of fairness.
- **Privacy**: membership inference, training-data extraction, and differential privacy as the principled defense (the
  slides note it is not yet practical for LLMs).

**Prerequisites to shore up**

- Mutual information and its variational lower bounds; softmax cross-entropy ([06](../06-linear-classifiers-and-kernels/)).
- Transformers ([17](../17-sequence-models-and-transformers/)).

**Open questions / further study**

- Why does contrastive learning work when the mutual-information story does not explain it (Tschannen et al. 2020)?
- Do LLMs learn world models or surface statistics, and how would we tell? (The question L35 ends on.)
- Reading: Chen et al. 2020 (SimCLR); Radford et al. 2021 (CLIP); Devlin et al. 2019 (BERT); Ouyang et al. 2022
  (InstructGPT); Rafailov et al. 2023 (DPO); Bender et al. 2021, "On the dangers of stochastic parrots"; Barocas, Hardt &
  Narayanan, *Fairness and Machine Learning* (free online).

## Connections

- ← [17 Transformers](../17-sequence-models-and-transformers/): the architecture being pre-trained.
- ← [12 Autoencoders](../12-variational-inference-and-vaes/) and [09 PCA](../09-latent-factor-models/): earlier pretext tasks
  (reconstruct the input); posterior collapse shows likelihood is not the same as a useful representation.
- ← [02 kNN](../02-naive-bayes-and-knn/): contrastive embeddings are learned distance metrics.
- ← [00 Learning fundamentals](../00-learning-fundamentals/): memorization, benchmark contamination and domain shift are
  violations of the train/test protocol and the IID assumption.
- ↔ [18 Diffusion](../18-diffusion-and-image-generation/): CLIP text embeddings condition image generation.
