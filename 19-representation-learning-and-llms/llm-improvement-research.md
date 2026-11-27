# How people are improving LLMs now (November 2026)

[back to the folder](README.md) · architecture details: [LLM architecture design](../17-sequence-models-and-transformers/llm-architecture-design.md)

**The question.** What are people researching right now to make neural networks and LLMs better? Is it making models
smaller, making them produce more output, or something else?

**The short answer: both, and several other things.** From 2020 to about 2024 the main recipe was "more parameters,
more data, more compute" (the scaling laws of Kaplan et al. 2020 and Hoffmann et al. 2022). That still works, but it
is expensive and good web text is running out. So most current gains come from seven directions:

| # | Direction | One-line idea | Makes the model… |
| --- | --- | --- | --- |
| 1 | [Better training data](#1-better-training-data-and-distillation) | filter, synthesize and distil the data the model learns from | better at the same size |
| 2 | [Smaller, cheaper models](#2-smaller-and-cheaper-models-quantization-pruning-mixture-of-experts) | quantization, pruning, mixture of experts | cheaper at the same quality |
| 3 | [Test-time compute](#3-test-time-compute-letting-the-model-think-longer) | let the model think longer or try several answers | better by producing *more* output |
| 4 | [RL post-training](#4-reinforcement-learning-after-pretraining) | reward the model for correct answers, not for imitation | better at reasoning, math and code |
| 5 | [Architecture](#5-architecture-cheaper-attention-hybrids-multi-token-prediction) | cheaper attention, state-space hybrids, multi-token prediction | able to use long context cheaply |
| 6 | [Systems efficiency](#6-systems-efficiency-same-model-faster) | kernels, speculative decoding, KV-cache tricks, low precision | faster and cheaper, same outputs |
| 7 | [Agents and tools](#7-agents-tools-and-retrieval) | retrieval, code execution, multi-step agent training | able to do tasks the weights alone can't |

Directions 2 and 3 pull in opposite ways ("smaller" vs. "more output"), and the current frontier uses both: a
cheaper model per token that spends more tokens on hard questions.

## A worked example: the two things Hamed saw

**Gemini 4 Neon** (Google DeepMind, released 18 Nov 2026). Artificial Analysis puts it 3rd on its Intelligence
Index with a score of 61 at the `xhigh` thinking level, 1 point above Gemini 4 Carbon and behind GPT-6.2 Astra and
Claude Fable 5.2. Google's announcement gives no architecture, size or training details. It talks about **thinking
levels** (four of them, with `xhigh` adding "deeper, multi-step reasoning") and **cost per task**. Artificial Analysis
measured that Neon uses about 15% *more* output tokens per task than Carbon, and that its lower cost per task comes
from lower token prices and higher output speed (about 110 vs. 57 tokens/s). So, from outside, the visible levers are
direction 3 (spend more tokens thinking) and direction 6 (serve tokens faster and cheaper). What happened inside
training is not public, which is typical of closed models.

**"Perturbing Inputs Improves Pretraining Distillation"** (Cai, Dong, Papailiopoulos & Garg, arXiv 2611.11598, Nov
2026). This is direction 1. In ordinary distillation a small *student* is trained to match a big *teacher*'s
next-token distribution on clean text. Here, for an early part of pretraining, both see *perturbed* inputs (added
token noise, two documents blended together, or neighbouring tokens blended), and the student still matches the
teacher. The rest of the token budget is ordinary distillation.

- Results: 0.025 nats lower loss than standard distillation for a 135M student, which the abstract says is 1.45 times
  the gain that distillation itself gives over plain next-token training. For a 0.5B student the gain is 0.020 nats,
  and the zero-shot average over seven tasks goes from 52.5% to 53.5%.
- Why it helps is not settled. The ablations say most of the gain comes from perturbing the *teacher's* input: the
  teacher's answers on inputs it would never see in clean text carry information about how it generalizes, which
  clean text does not show. The authors call it a proof of concept; whether it holds at frontier scale is open.
- Course link: this is the same move as **data augmentation** in contrastive learning ([README](README.md)) and
  **denoising** as a pretext task ([12](../12-variational-inference-and-vaes/), [18](../18-diffusion-and-image-generation/)):
  show the model corrupted inputs so it learns a function that behaves well around the data, not only on it.

## 1. Better training data and distillation

**Why it matters.** For a fixed model and compute budget, the data is the biggest lever left. Chinchilla showed that
most models of 2020–2021 were trained on too little data for their size; after that, the question became which
tokens to train on.

- **Filtering and deduplication.** Build a classifier for "educational" or "high-quality" text and keep only what it
  scores highly. FineWeb (Penedo et al. 2024) and DataComp-LM (Li et al. 2024) showed that the filter, not the model,
  can be worth several benchmark points. Course link: this is a learned importance weighting of the training
  distribution, so the train/test assumptions in [00](../00-learning-fundamentals/) and [20](../20-evaluation/) apply
  (filtered data can drift from what users actually ask).
- **Synthetic data.** Have a strong model write textbook-style text, problems with worked solutions, or code with
  tests. *Textbooks Are All You Need* (Gunasekar et al. 2023, the phi-1 paper) is the early example.
- **Knowledge distillation.** Train the student on the teacher's full next-token distribution instead of the one-hot
  next token (Hinton et al. 2015). Gemma 2 (2024) used this in pretraining for its small models. The loss is the
  cross-entropy to soft targets, i.e. a KL divergence ([06](../06-linear-classifiers-and-kernels/)).
- **On-policy distillation.** Ordinary distillation trains on the teacher's text, but at test time the student
  generates its own text and goes off the teacher's distribution. On-policy methods let the student generate, then
  ask the teacher how it would have scored each token (GKD, Agarwal et al. 2023; MiniLLM, Gu et al. 2023). In 2026
  this is very active: self-distillation with privileged information (OPSD, Zhao et al. 2026), distilling a long
  system prompt into weights (OPCD, Ye et al. 2026), and a survey (Song & Zheng 2026) that frames it as reverse-KL
  minimization on the student's own samples, the same objective shape as KL-regularized RL (direction 4).
- **Perturbed-input distillation** (Cai et al. 2026, above).

## 2. Smaller and cheaper models: quantization, pruning, mixture of experts

**Why it matters.** Serving cost scales with the bytes and FLOPs touched per token. Most "Flash", "Mini" and "Lite"
models come from this line of work, usually combined with distillation from a larger sibling.

- **Quantization.** Store weights (and sometimes activations and the KV cache) in 8, 4 or fewer bits. Post-training
  methods: GPTQ (Frantar et al. 2022) solves a small least-squares problem per layer to choose rounded weights; AWQ
  (Lin et al. 2023) protects the few weight channels that large activations hit. Training in low precision is the
  newer step: NVIDIA trained a 12B model on 10T tokens in 4-bit NVFP4 with FP8-level loss (2025), and *Full-Stack FP4*
  (Ding et al. 2026) also quantizes optimizer states and attention. BitNet b1.58 (Ma et al. 2024) goes to ternary
  weights {−1, 0, 1}.
- **Low-rank adaptation.** LoRA and QLoRA (Dettmers et al. 2023) fine-tune a 4-bit model by learning a low-rank update
  `ΔW = AB`. Course link: the same low-rank factorization as PCA and recommenders ([09](../09-latent-factor-models/)).
- **Pruning.** Remove weights (SparseGPT, Frantar & Alistarh 2023) or whole layers, heads and channels, then distil
  to recover (Minitron, Muralidharan et al. 2024). Course link: L1 sparsity and feature selection
  ([05](../05-feature-selection-and-regularization/)), applied after training instead of during it.
- **Mixture of experts (MoE).** Replace the MLP with many expert MLPs and a router that sends each token to a few of
  them, so total parameters grow while compute per token does not (Shazeer et al. 2017; Switch Transformer, Fedus et
  al. 2021; Mixtral 2024; DeepSeek-V3 2024). Course link: a **mixture model** ([10](../10-mixture-models-and-em/)) whose
  gating network plays the role of the responsibilities, but trained by backprop with hard top-k routing instead of
  EM. Most large open models of 2025–2026 are MoE (families C, D and E in the
  [architecture page](../17-sequence-models-and-transformers/llm-architecture-design.md)).

## 3. Test-time compute: letting the model think longer

**Why it matters.** This is the "increase the output" direction, and it works: on math, code and science problems,
accuracy rises with the number of tokens spent at inference. Snell et al. (2024) found that, for some problems,
spending inference compute well beats a model 14 times larger.

- **Chain of thought** (Wei et al. 2022): ask for the reasoning steps before the answer. Reasoning models are trained
  to do this at length (direction 4), and "thinking levels" like Gemini's `high`/`xhigh` set the token budget.
- **Sample many, then choose.** Self-consistency (Wang et al. 2022) samples several reasoning paths and takes the
  majority answer. Course link: a Monte Carlo estimate of `argmax_a p(a | x)` with the reasoning path marginalized
  out ([11](../11-monte-carlo-and-mcmc/)).
- **Verifiers.** Score candidate answers with a learned reward model or a checker. Process reward models score each
  step, not only the final answer (Lightman et al. 2023, *Let's Verify Step by Step*). Best-of-N with a verifier is a
  crude form of sampling from `p(y | x) · reward`.
- **Budget control.** s1 (Muennighoff et al. 2025) showed a 1,000-example fine-tune plus "budget forcing" (stopping or
  extending the thinking) is enough to get a scaling curve. Course link: beam search and decoding in
  [17](../17-sequence-models-and-transformers/) are the older, cheaper relatives.

The cost: tokens are not free, and longer thinking means slower answers. Much current work (and the Neon example
above) is about getting the same accuracy with fewer thinking tokens.

## 4. Reinforcement learning after pretraining

**Why it matters.** Pretraining and supervised fine-tuning are imitation: the model learns to produce text like the
data. RL optimizes a reward instead, so the model can find reasoning strategies that are not in the data. This is
what turned base models into the reasoning models of 2025–2026.

- **RLHF** (InstructGPT, Ouyang et al. 2022) and **DPO** (Rafailov et al. 2023), covered in the
  [folder README](README.md): maximize a learned reward with a KL penalty to the base model. Constitutional AI (Bai et
  al. 2022) replaces most human labels with AI feedback (RLAIF).
- **RL with verifiable rewards (RLVR).** Use rewards that a program can check: a math answer matches, the unit tests
  pass. Tülu 3 (Lambert et al. 2024) named it; DeepSeek-R1 (2025) showed that RL with such rewards alone produces
  long chains of thought, self-checking and backtracking. **GRPO** (from DeepSeekMath, Shao et al. 2024) drops the
  value network: sample a group of answers per question and use each answer's reward minus the group mean as its
  advantage. Course link: a baseline for variance reduction, the same idea as control variates in Monte Carlo
  ([11](../11-monte-carlo-and-mcmc/)).
- **Reward hacking** is the main failure: the policy finds what the reward checks, not what it means (tests that pass
  without solving the problem). The KL term and verifier design are the defences.
- **Open questions.** Does RL teach new skills or only make the base model pick its best existing behaviour more
  often? How to give rewards for tasks without a checker (writing, research)? The survey of Zhang et al. (2025, the
  "agentic RL" survey) covers the current map.

## 5. Architecture: cheaper attention, hybrids, multi-token prediction

**Why it matters.** Attention costs `O(n²)` in sequence length and its KV cache grows with `n`, which caps context
length and makes long reasoning traces expensive. The [architecture page](../17-sequence-models-and-transformers/llm-architecture-design.md)
covers this in detail from 77 real models; the research directions are:

- **State-space and linear-attention layers.** Mamba (Gu & Dao 2023), Mamba-2 (Dao & Gu 2024) and Gated DeltaNet (Yang
  et al. 2024) keep a fixed-size state, so cost is linear in length. Course link: these are RNNs
  ([17](../17-sequence-models-and-transformers/)) with a linear state update, the deterministic cousin of the HMM
  forward recursion ([13](../13-markov-chains-and-hmms/)).
- **Hybrids.** Pure linear models are worse at exact recall, so most new designs keep a few full-attention layers
  (Jamba 2024; Qwen3-Next; MiniCPM-SALA 2026, which mixes sparse and linear attention 1:3 and reaches 1M tokens).
  Bae et al. (2025) compare hybrid layouts systematically. HydraHead (Tan et al. 2026) mixes per *head* instead of per
  layer, keeping full attention only for heads that do retrieval.
- **Sparse attention.** Attend to a learned subset of past tokens (Native Sparse Attention, Yuan et al. 2025).
- **Multi-token prediction.** Predict the next several tokens from each position (Gloeckle et al. 2024). It gives a
  richer training signal and extra heads that can draft tokens for speculative decoding (direction 6). DeepSeek-V3
  uses it.
- **Optimizers.** Muon (scaled up by Liu et al. 2025) orthogonalizes the update matrix and is being adopted as an AdamW
  replacement for large runs.

## 6. Systems efficiency: same model, faster

**Why it matters.** None of this changes what the model computes, but it decides what is affordable, and therefore
how much test-time compute (direction 3) a provider can sell.

- **IO-aware kernels.** FlashAttention (Dao et al. 2022) and FlashAttention-3 (Shah et al. 2024) compute exact
  attention without writing the `n × n` matrix to slow memory.
- **Speculative decoding** (Leviathan et al. 2022; EAGLE, Li et al. 2024). A small draft model proposes several tokens;
  the big model checks them in one pass and keeps the prefix it agrees with. Course link: the accept/reject rule is
  exactly **rejection sampling** ([11](../11-monte-carlo-and-mcmc/)), so the output distribution is the big model's,
  not an approximation.
- **KV-cache management.** PagedAttention / vLLM (Kwon et al. 2023) stores the cache in pages like virtual memory;
  GQA, MLA and KV quantization shrink it.
- **Low-precision arithmetic** (FP8, FP4) for training and serving, as in direction 2.

## 7. Agents, tools and retrieval

**Why it matters.** Many tasks need facts the model does not have, exact computation, or many steps. Giving the
model tools moves that work out of the weights.

- **Retrieval-augmented generation** (RAG, Lewis et al. 2020): fetch relevant documents and put them in the context.
  Course link: a learned-embedding nearest-neighbour lookup ([02](../02-naive-bayes-and-knn/), with the embeddings
  from contrastive training in this folder).
- **Tool use and acting.** ReAct (Yao et al. 2022) interleaves reasoning and actions; Toolformer (Schick et al. 2023)
  teaches the model when to call an API.
- **Training agents with RL.** Long multi-step tasks (software engineering on SWE-bench, Jimenez et al. 2023; app and
  terminal tasks) are now trained with RL on outcome rewards. Pu et al. (2026) report that a simple outcome-only,
  on-policy, KL-anchored recipe trained a 14B model to the top of the AppWorld leaderboard and raised a 9B model on
  SWE-bench Verified by 16.6 points. The Gemini 4 Neon announcement ranks itself on agent benchmarks (Coding Agent
  Index, Terminal-Bench), which shows where the labs now compete.

## How to read new claims

1. **Ask which direction a gain comes from.** A leaderboard jump can come from a better model, from more thinking
   tokens, or from a cheaper price at the same quality. Artificial Analysis reports tokens per task and cost per task
   for this reason.
2. **Check the scale.** Many papers, including the perturbed-distillation one, test on models under 1B parameters.
   Methods that help small models sometimes vanish at scale.
3. **Check the evaluation** ([20](../20-evaluation/)). Benchmark contamination, LLM-as-judge bias and the variance across
   seeds are large compared with the 0.5–2 point gains most papers report. The perturbed-distillation paper reports
   three seeds and a bootstrap standard error, which is better than most.

## Knowledge gap

- **What the course slides leave out.** Slides 23–24 cover pre-training, SFT, RLHF and DPO/GRPO by name. They do not
  cover distillation, quantization, mixture of experts, test-time scaling, verifiers, RL with verifiable rewards,
  state-space hybrids, speculative decoding or agents. This page is the bridge.
- **Prerequisites to shore up.** KL divergence and cross-entropy ([06](../06-linear-classifiers-and-kernels/)); policy
  gradients and baselines (not in either course: Sutton & Barto ch. 13); rejection and importance sampling
  ([11](../11-monte-carlo-and-mcmc/)).
- **Open questions.** Why perturbed inputs help distillation; whether RL creates new capabilities or reweights existing
  ones; how far test-time compute can substitute for model size; what to train on once human-written text is used up.

## References

All arXiv links checked against the arXiv API on 2026-11-20.

**Worked example**
- Cai, Dong, Papailiopoulos & Garg (2026). Perturbing Inputs Improves Pretraining Distillation. [arXiv:2611.11598](https://arxiv.org/abs/2611.11598)
- Google DeepMind (18 Nov 2026). [Introducing Gemini 4 Neon](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/)
- Artificial Analysis (18 Nov 2026). [Gemini 4 Neon intelligence gain](https://artificialanalysis.ai/articles/gemini-4-neon-intelligence-gain)

**Scaling**
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

**1. Data and distillation**
- Penedo et al. (2024). The FineWeb Datasets. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- Li et al. (2024). DataComp-LM. [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
- Gunasekar et al. (2023). Textbooks Are All You Need. [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)
- Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size. [arXiv:2408.00118](https://arxiv.org/abs/2408.00118)
- Agarwal et al. (2023). On-Policy Distillation of Language Models (GKD). [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- Gu et al. (2023). MiniLLM: On-Policy Distillation of Large Language Models. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- Zhao et al. (2026). Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs. [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)
- Ye et al. (2026). On-Policy Context Distillation for Language Models. [arXiv:2602.12275](https://arxiv.org/abs/2602.12275)
- Song & Zheng (2026). A Survey of On-Policy Distillation for Large Language Models. [arXiv:2604.00626](https://arxiv.org/abs/2604.00626)

**2. Smaller models**
- Frantar et al. (2022). GPTQ. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al. (2023). AWQ. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Dettmers et al. (2023). QLoRA. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Ma et al. (2024). The Era of 1-bit LLMs (BitNet b1.58). [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- NVIDIA (2025). Pretraining Large Language Models with NVFP4. [arXiv:2509.25149](https://arxiv.org/abs/2509.25149)
- Ding et al. (2026). Full-Stack FP4. [arXiv:2607.04422](https://arxiv.org/abs/2607.04422)
- Frantar & Alistarh (2023). SparseGPT. [arXiv:2301.00774](https://arxiv.org/abs/2301.00774)
- Muralidharan et al. (2024). Compact Language Models via Pruning and Knowledge Distillation (Minitron). [arXiv:2407.14679](https://arxiv.org/abs/2407.14679)
- Shazeer et al. (2017). The Sparsely-Gated Mixture-of-Experts Layer. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Fedus, Zoph & Shazeer (2021). Switch Transformers. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Jiang et al. (2024). Mixtral of Experts. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

**3. Test-time compute**
- Wei et al. (2022). Chain-of-Thought Prompting. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Wang et al. (2022). Self-Consistency. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Lightman et al. (2023). Let's Verify Step by Step. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- Snell et al. (2024). Scaling LLM Test-Time Compute Optimally. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
- Muennighoff et al. (2025). s1: Simple test-time scaling. [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)

**4. RL post-training**
- Ouyang et al. (2022). InstructGPT. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Bai et al. (2022). Constitutional AI. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- Rafailov et al. (2023). Direct Preference Optimization. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Shao et al. (2024). DeepSeekMath (GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Lambert et al. (2024). Tülu 3 (RLVR). [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
- DeepSeek-AI (2025). DeepSeek-R1. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Zhang et al. (2025). The Landscape of Agentic Reinforcement Learning for LLMs: A Survey. [arXiv:2509.02547](https://arxiv.org/abs/2509.02547)

**5. Architecture**
- Gu & Dao (2023). Mamba. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Dao & Gu (2024). Transformers are SSMs (Mamba-2). [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- Yang et al. (2024). Gated Delta Networks. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Lieber et al. (2024). Jamba. [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
- Bae et al. (2025). Hybrid Architectures for Language Models. [arXiv:2510.04800](https://arxiv.org/abs/2510.04800)
- MiniCPM Team (2026). MiniCPM-SALA. [arXiv:2602.11761](https://arxiv.org/abs/2602.11761)
- Tan et al. (2026). HydraHead. [arXiv:2606.20097](https://arxiv.org/abs/2606.20097)
- Yuan et al. (2025). Native Sparse Attention. [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- Gloeckle et al. (2024). Better & Faster LLMs via Multi-token Prediction. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
- Liu et al. (2025). Muon is Scalable for LLM Training. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

**6. Systems**
- Dao et al. (2022). FlashAttention. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Shah et al. (2024). FlashAttention-3. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Leviathan, Kalman & Matias (2022). Speculative Decoding. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Li et al. (2024). EAGLE. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
- Kwon et al. (2023). PagedAttention (vLLM). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

**7. Agents and tools**
- Lewis et al. (2020). Retrieval-Augmented Generation. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Yao et al. (2022). ReAct. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Schick et al. (2023). Toolformer. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Jimenez et al. (2023). SWE-bench. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- Pu et al. (2026). Explore More, Drift Less: Outcome-Only RL Can Suffice for Long-Horizon Interactive Agents. [arXiv:2609.01245](https://arxiv.org/abs/2609.01245)
