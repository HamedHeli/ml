# Inside a modern LLM block: what each mechanism does, where it sits, and what breaks without it

[back to the folder](README.md) · the design process and the gallery evidence: [LLM architecture
design](llm-architecture-design.md) · numerical checks: [`llm_block_mechanisms.py`](llm_block_mechanisms.py) · the
blocks in general: [building blocks](../15-neural-networks/building-blocks.md)

[LLM architecture design](llm-architecture-design.md) shows *that* every LLM since 2024 makes the same handful of
choices (residual stream, pre-norm, RMSNorm, gated MLP, RoPE, no dropout and no biases, shared K/V heads), and gives
each an evidence level. This page goes one level down. For each mechanism it answers four questions:

1. **What is it?** The exact computation.
2. **Where does it sit?** Its position in the block, because most of these only work in one place.
3. **What role does it play?** The problem it solves.
4. **What breaks without it?** Shown with a small experiment where one fits, from `llm_block_mechanisms.py`.

Then come five related mechanisms that the same block depends on: the `1/√d` score scale and causal mask, QK-norm,
depth-scaled initialization, the final norm, and tied embeddings.

```bash
python 17-sequence-models-and-transformers/llm_block_mechanisms.py    # about 2 min on a CPU
```

The experiments are toys: MLP stacks of width 64, random attention inputs, linear regression. They isolate each
mechanism; they are not LLM training runs. Where a claim rests on published LLM-scale results instead, the section
says so, and the evidence marks (●●● established, ●●○ supported, ●○○ convention) are the ones from the design page.

## The block, with every mechanism in its place

A Llama-3-style decoder. The numbers in the right column are the sections below.

```text
tokens ──► x = Embed(tokens)                       (nothing added for position: see ⑤)          ⑫
for layer in 1..N:
    ┌ h = RMSNorm(x)                                pre-norm, RMSNorm                         ② ③
    │ q, k, v = h·W_q, h·W_k, h·W_v                 no biases; k, v have g < h heads (GQA)     ⑦ ⑧
    │ q, k = RMSNorm(q), RMSNorm(k)                 QK-norm (optional, per head)              ⑩
    │ q, k = RoPE(q, pos), RoPE(k, pos)             position enters here, on q and k only     ⑤
    │ a = softmax(q·kᵀ / √d_h + causal mask) · v    score scale and mask                      ⑨
    └ x = x + a·W_o                                 residual add; W_o init scaled by 1/√(2N)   ① ⑪
    ┌ h = RMSNorm(x)                                pre-norm                                  ②
    │ u = SiLU(h·W_gate) ⊙ (h·W_up)                 gated MLP (SwiGLU)                        ④
    └ x = x + u·W_down                              residual add; W_down init scaled          ① ⑪
x = RMSNorm(x)                                      final norm                                ⑫
logits = x·W_out    (W_out = Embedᵀ in small models: tied embeddings)                         ⑫
dropout: none, anywhere                                                                       ⑥
```

Two things to notice before the details. First, the residual stream `x` is only ever *added to*: no norm, no
dropout, no activation touches it between the embedding and the final norm. Second, everything that touches
position (RoPE), head sharing (GQA) and logit control (QK-norm, `1/√d_h`) happens inside attention, on `q` and `k`.
Neither changes `v` or the residual stream.

## ① Residual stream ●●●

**What.** Each sub-layer computes a correction `F(x)` and adds it: `x ← x + F(Norm(x))`. The running sum `x` is the
*residual stream*. It is the only thing passed from one layer to the next.

**Where.** Around every sub-layer: one add after attention, one after the MLP. It starts at the embedding and ends at
the final norm.

**Role.** Two jobs.

- *Optimization.* The derivative of `x + F(x)` with respect to `x` is `I + ∂F/∂x`. The identity term carries the
  gradient from the loss to every layer without passing through any weight matrix, so depth does not shrink it (He
  et al. 2016). At initialization the network is close to the identity, and each layer only has to learn a small
  correction.
- *Communication.* Because every sub-layer reads from and writes to the same vector, the stream works as a shared
  memory. An attention head in layer 3 can write a feature that an MLP in layer 20 reads, with nothing in between
  having to preserve it (Elhage et al. 2021, "A mathematical framework for Transformer circuits"). Interpretability
  work on LLMs is built on this view.

**Without it.** 48 MLP layers, standard initialization:

| | gradient at the input ÷ gradient at the output | output RMS ÷ input RMS |
| --- | --- | --- |
| plain `x = F(x)` | 7.5 × 10⁻¹⁵ | 7.7 × 10⁻¹⁵ |
| residual `x = x + F(x)` | 1.3 | 1.2 |

Without the residual, both the signal and the gradient shrink by a constant factor per layer and are gone after 48
layers. Trained for 400 steps (table in ② below), the plain 24-layer stack learns nothing at learning rate 10⁻³ and
diverges at 10⁻².

## ② Pre-normalization (and the final norm) ●●●

**What.** Normalize the *input* of each sub-layer, inside its branch: `x + F(Norm(x))`. The original 2017
Transformer did the opposite, post-norm: `Norm(x + F(x))`, which puts the norm on the stream itself.

**Where.** Two norms per block: in front of attention and in front of the MLP. Because the stream is never
normalized inside the stack, one more norm is needed at the very end, before the output head.

**Role.** Each sub-layer sees inputs at a fixed scale, whatever has piled up in the stream. The stream itself stays a
pure sum, so ① keeps working. With post-norm, the gradient to a lower layer must pass through every norm above it,
and the top layers get large gradients at initialization. Xiong et al. (2020) showed that with post-norm the
gradient on the last layer does not shrink with depth, while with pre-norm it scales like `1/√N`. That is why
post-norm needs a learning-rate warm-up and pre-norm mostly does not.

**Without it.** 24-layer stacks trained with Adam at a constant learning rate, no warm-up, on a regression whose
target has variance 0.43 (loss at step 0 → mean of the last 20 of 400 steps):

| Wiring | lr 10⁻³ | lr 10⁻² |
| --- | --- | --- |
| plain `x = F(x)` | 0.42 → 0.43 (learns nothing) | diverges (10¹²) |
| residual, no norm | 2.14 → 0.061 | diverges (10⁴) |
| post-norm `Norm(x + F(x))` | 1.42 → 0.26 | 1.42 → 0.43 (learns nothing) |
| **pre-norm `x + F(Norm(x))`** | 1.44 → 0.068 | **1.44 → 0.052** |

Gradient on the top layer's weights at initialization, depth 6 / 24 / 96: post-norm 5.1 / 4.8 / 5.1, pre-norm
3.1 / 1.8 / 0.9. This reproduces the Xiong et al. scaling.

Read the table as: the residual add makes depth trainable, and the pre-norm makes it trainable at a large learning
rate. Post-norm without warm-up is the only normalized wiring that degrades as the learning rate grows.

**The cost.** In pre-norm, nothing ever shrinks the stream, so its size grows with depth (see ⑪). A late layer's
output is then small relative to the stream and changes it less. At scale this shows up as very large activations in
a few channels. The extra norms in step 11 of the [design page](llm-architecture-design.md) (sandwich norm, OLMo's norm after
the sub-layer) put some of that control back without returning to post-norm.

## ③ RMSNorm ●●○

**What.** `RMSNorm(x) = g ⊙ x / √(mean(x²) + ε)`, one learned gain vector `g`. LayerNorm also subtracts the mean
and adds a learned bias: `g ⊙ (x − mean(x)) / std(x) + b`.

**Where.** Wherever a norm appears: the two pre-norms, the final norm, and QK-norm.

**Role.** Fix the scale of the vector going into a sub-layer. The checks on a 4096-dimensional vector:

| Property | Result |
| --- | --- |
| scale invariance: `RMSNorm(7.3x) = RMSNorm(x)` | max difference 2 × 10⁻⁶ |
| `LayerNorm(x) = RMSNorm(x − mean(x))` | exact |
| adding 3 to every feature | changes RMSNorm by up to 3.6, LayerNorm by 5 × 10⁻⁷ |
| gradient direction vs. `x` | cosine 4 × 10⁻⁸ |

So LayerNorm is RMSNorm plus a centring step. RMSNorm keeps the mean component and LayerNorm removes it. The last
row is a consequence of scale invariance: the gradient is always orthogonal to `x`, so the layer before a norm gets
no signal about the length of its output, only its direction.

**Without it (that is, with LayerNorm).** Nothing breaks. Zhang & Sennrich (2019) found the re-centring is not what
makes LayerNorm work and RMSNorm matches its quality. It needs one reduction over the vector instead of two, and one
learned vector instead of two, and they report 7–64% faster runs on their models. At LLM scale norms are
memory-bound, so the saving is real. This is a cost choice, not a fix. Without *any* norm, the "residual, no norm"
row in ② diverges at the larger learning rate.

## ④ Gated MLP (SwiGLU) ●●○

**What.** `MLP(h) = W_down · (SiLU(W_gate h) ⊙ W_up h)`. Two input projections: one passes through SiLU and acts
as a gate, the other is linear, and they are multiplied elementwise.

**Where.** The channel mixer: the second sub-layer of every block, after its own pre-norm. In MoE models, each expert
is one of these.

**Role.** The MLP is where each token is transformed on its own, and it holds about two thirds of a dense model's
parameters. Each hidden unit works as a key–value memory slot (Geva et al. 2021): its input weights detect a
pattern, its output column writes a feature into the stream. The gate makes each slot *conditional*: unit `i` writes
`SiLU(gate_i) · up_i`, a product of two learned features of the input. A plain MLP can only write `σ(a_i)`. Products
let a unit say "feature *a* matters only when *b* is present" directly. The [building-blocks
notebook](../15-neural-networks/building_blocks.ipynb) (§4) measures about 20× lower error on `x₁x₂` with the gated
version.

**Where the 8/3 comes from.** A gated MLP has three matrices instead of two. To keep parameters equal to a 4·d GELU
MLP, the hidden width drops to `8/3 · d`. For `d = 4096`: 10923 hidden units, 134.2M parameters, the same as the
4·d GELU MLP (ratio 1.000). Llama 3 8B goes wider than that, to 14336 = 3.5·d.

**Without it.** A plain GELU MLP of the same size works, slightly worse. Shazeer (2020) found lower perplexity for GLU
variants at equal compute, and Narang et al. (2021) found the gain transfers to other codebases. The gate is a
quality-per-parameter gain, not a requirement: NVIDIA's Nemotron uses squared ReLU instead.

## ⑤ RoPE: rotary position embedding ●●●

**What.** Split each query and key vector of a head into pairs of coordinates. Rotate pair `i` at position `m` by
the angle `m·θᵢ`, with `θᵢ = base^(−2i/d_h)`. Base 10 000 originally; Llama 3 uses 500 000.

**Where.** Inside attention, on `q` and `k` only, after the projections (and after QK-norm if there is one), before
the dot product. Not on `v`. Nothing is added to the embedding or the stream.

**Role.** Attention without position information treats the context as a set: shuffle the tokens and each output is
the same. RoPE puts position into the one quantity that needs it, the score. Rotations satisfy
`R(m)ᵀ R(n) = R(n − m)`, so

`score(m, n) = (R(m) q)ᵀ (R(n) k) = qᵀ R(n − m) k`

depends on the tokens' *offset*, not on where they are. Rotations also keep lengths, so the scale of `q` and `k` is
untouched. The checks: shift every position by 1000 and the 64 × 64 score matrix changes by 3 × 10⁻¹³ with RoPE,
and by 22.8 with a sinusoidal embedding added to the input (typical score size 8.5). RoPE changes `|q|` by
9 × 10⁻¹⁶.

**Without it (with a learned or added absolute embedding, as in GPT-2).**

- The model must learn relative position separately for every absolute position. A learned table also has no rows
  beyond the training length.
- The position vector is mixed into every feature of the stream, in every layer, and the model must spend capacity
  to separate "what" from "where".
- Long-context extension is hard. With RoPE it is a change of frequencies: position interpolation (Chen et al. 2023),
  YaRN (Peng et al. 2023), or a larger base.

The KV cache stores keys *after* rotation. A key computed at position `n` never needs to be recomputed.

## ⑥ No dropout in pretraining ●●○

**What.** Dropout probability 0 everywhere: on attention weights, on the residual branches, on the embedding.

**Where.** It is an absence, so: nowhere. GPT-2 had it in three places (attention probabilities, each branch before
the residual add, the embedding).

**Role of the absence.** Dropout is a regularizer. It trades a little extra bias and noise for less overfitting.
Overfitting needs repeated data: the model has to see the same examples often enough to memorise them. LLM
pretraining sees most tokens once, so the training loss on the next batch *is* a held-out loss. There is nothing for
dropout to fight, and it only costs signal and throughput. The toy version, linear regression with 100 features and
dropout `p = 0.2` (test MSE, noise floor 0.25):

| Regime | no dropout | dropout |
| --- | --- | --- |
| 80 examples, each seen 2000 times | 1.38 | **0.77** |
| 20 000 examples, each seen once | **0.253** | 0.286 |

The same dropout halves the error when data repeats and makes it 13% worse when it does not. PaLM trained without
dropout, and Liu et al. (2025) measured worse downstream results with dropout in single-epoch pretraining.

**When it comes back.** Whenever data repeats: multi-epoch training on a small corpus, fine-tuning a large model on a
small dataset (LoRA dropout), and the small models elsewhere in this repo. The rule depends on the data regime, not
the architecture. Weight decay usually stays on in LLM pretraining. In that setting it is generally read as an
optimization tool (it sets the effective learning rate of weights that feed a norm), not as protection against
overfitting.

## ⑦ No biases ●●○

**What.** No `+ b` in any linear layer (Q, K, V, O, gate, up, down) and no bias in the norms.

**Where.** Everywhere. The few exceptions are deliberate: Qwen2 kept a bias on Q, K and V, and GPT-OSS keeps
attention biases next to its learned sink logit.

**Role of the absence.** Three reasons, from strongest to weakest.

1. **Some biases do nothing at all.** A key bias adds `q·b_k` to every score in a row. Softmax ignores a constant
   added to all of its inputs, so the output does not change: measured difference 7 × 10⁻¹⁶. A value bias passes
   through attention unchanged, because the weights sum to 1. It becomes a constant `b_v W_o` added to every output,
   which is the same as one output bias (measured 9 × 10⁻¹⁶ away from that). A bias in front of RMSNorm or LayerNorm
   is partly undone by the norm.
2. **They add almost no capacity.** If every linear layer of a Llama-3-8B-shaped model had a bias, the biases would
   be 0.017% of the weights.
3. **Stability at scale.** PaLM reported that removing all biases increased training stability for large models.
   That is the lab's own observation, with no published ablation, so this reason is the weakest of the three.

**Without it (with biases).** Small models train fine with biases: GPT-2 had them. The cost is mostly redundant
parameters, plus whatever instability PaLM saw at scale.

## ⑧ Shared K/V heads: MQA and GQA ●●● (GQA), ●●○ (MLA)

**What.** Multi-head attention (MHA) gives each of the `h` query heads its own key and value heads. Grouped-query
attention (GQA, Ainslie et al. 2023) has only `g < h` key/value heads, and each is shared by `h/g` query heads.
Multi-query attention (MQA, Shazeer 2019) is `g = 1`. `g = h` is ordinary MHA, which the code checks: the outputs
agree to 7 × 10⁻¹⁶.

**Where.** In the K and V projections of attention: `W_k` and `W_v` have `g·d_h` output columns instead of
`h·d_h`. Queries, `W_o` and everything outside attention are unchanged. MLA (DeepSeek) goes further and caches one
compressed latent vector per token, which is expanded into per-head K and V when it is used.

**Role.** It cuts *inference* cost, not training cost. When generating, each new token attends to the K and V of
every earlier token. These are cached, and the whole cache is read from memory once per generated token. With MHA that
read, not the arithmetic, limits speed and batch size. For Llama 3 8B's shapes (32 layers, 32 query heads, head dim
128, bf16):

| | KV cache per token | at 128k tokens of context |
| --- | --- | --- |
| MHA, 32 KV heads | 512 KiB | 64 GiB |
| **GQA, 8 KV heads (Llama 3)** | **128 KiB** | **16 GiB** |
| MQA, 1 KV head | 16 KiB | 2 GiB |

The model's own weights are about 15 GiB in bf16. With MHA, the cache of a single 128k-token sequence would be four
times the size of the model.

**Why sharing costs little quality.** The heads' keys and values turn out to be redundant: many heads need the same
"what is at this position" information, and differ mainly in what they *ask* (the queries). Ainslie et al. found GQA
with 8 groups close to MHA quality at close to MQA speed. MQA alone loses some quality and was less stable, which is
why the middle setting won.

**Without it.** The model is just as good, but serving it costs several times more memory and bandwidth per user.
That is why all 38 models of 2026 in the gallery share K/V in some form.

## Related mechanisms the block relies on

### ⑨ The `1/√d_h` score scale and the causal mask ●●●

**What and where.** Inside attention: `softmax(q·kᵀ / √d_h + M)`, where `M` is 0 on and below the diagonal and −∞
above it.

**Role of the scale.** If `q` and `k` have unit-variance entries, `q·k` has standard deviation `√d_h`. Without the
division, the softmax over 512 keys becomes nearly one-hot as heads get wider:

| `d_h` | score std | entropy without scale | largest weight | entropy with `/√d_h` | largest weight |
| --- | --- | --- | --- | --- | --- |
| 16 | 4.0 | 2.15 | 0.45 | 5.74 | 0.027 |
| 64 | 8.0 | 0.85 | 0.72 | 5.75 | 0.026 |
| 256 | 16.1 | 0.38 | 0.85 | 5.74 | 0.027 |

(uniform attention has entropy ln 512 = 6.24.) A saturated softmax has a near-zero gradient, so heads stuck in it
stop learning. The scale keeps the initial attention spread out at any head size.

**Role of the mask.** Next-token training predicts token `t+1` at every position at once. Without the mask, position
`t` could read token `t+1` and the task would be trivial. The mask is also what makes a KV cache possible: a past
token's K and V never depend on future tokens, so they can be computed once and kept.

### ⑩ QK-norm ●●○

**What and where.** An RMSNorm on each head's `q` and `k`, after the projections and before RoPE (Henry et al. 2020;
Dehghani et al. 2023).

**Role.** The `1/√d_h` scale controls scores only at initialization. During training `W_q` and `W_k` can grow, and the
scores grow with the product of their scales. In the check, multiplying both by `s`:

| `s` | max logit | attention entropy | with QK-norm: max logit | entropy |
| --- | --- | --- | --- | --- |
| 1 | 5.4 | 5.09 | 3.9 | 5.09 |
| 4 | 86 | 0.34 | 3.9 | 5.09 |
| 16 | 1380 | 0.02 | 3.9 | 5.09 |

Without QK-norm, attention collapses to one key per query. With it, the logits are bounded by `√d_h · g_q · g_k`
whatever the weights do, and only the learned gains can sharpen attention. This is the "attention logit growth"
instability that Wortsman et al. (2023) reproduced at small scale. At large scale it shows up as loss spikes.

**Without it.** Most models still train at small scale. At large scale it is cheap insurance, and about half the
gallery states it.

### ⑪ Depth-scaled initialization of the residual writes ●●○

**What and where.** The two matrices that write into the stream, `W_o` in attention and `W_down` in the MLP, are
initialized smaller, by `1/√(2N)` for `N` blocks (GPT-2 did this; nanoGPT and many codebases still do). Other schemes
zero-initialize them or follow μP.

**Role.** In a pre-norm stack the stream is a sum of `2N` branch outputs. With independent branches of similar size,
its variance grows like `2N`. Measured on 48 pre-norm layers: the stream's RMS ends 4.7× larger than it started with
standard init, and 1.1× with the scaled init. Bounding the stream keeps each layer's update a similar fraction of the
total and leaves the final norm less to undo. It is the initialization-time version of the depth problem in ②.

### ⑫ Final norm and tied embeddings ●●● / ●○○

**Final norm.** Because pre-norm never normalizes the stream, the output head would otherwise see a vector whose
size depends on depth and on training. One RMSNorm after the last block fixes that. Every pre-norm model has it.

**Tied embeddings.** The output head `W_out` (vocabulary × d) can reuse the input embedding matrix. That saves
`V·d` parameters: 525M for a 128k vocabulary at `d = 4096`, which is a large share of a 1B model and a small one of a
70B model. So small models tie (Llama 3.2 1B/3B, Gemma) and large ones mostly don't. This is a budget rule, not a
mechanism (●○○).

## How the pieces depend on each other

- **① and ② go together.** The residual add makes depth trainable. Pre-norm keeps that add clean while still giving
  every sub-layer a normalized input. Post-norm puts the norm back on the path that ① kept clear.
- **② creates the need for ⑪, ⑩ and the extra norms.** Since nothing normalizes the stream, it grows. Depth-scaled
  init bounds it at the start, and QK-norm and sandwich norms bound what grows during training.
- **③ is safe because of ②.** The pre-norm only needs to fix the scale of a sub-layer's input. Centring was never the
  important part.
- **⑤ and ⑧ meet in the cache.** The cache holds rotated keys for `g` heads. GQA sets how many vectors are stored per
  token, and RoPE means they never need recomputing.
- **⑥ and ⑦ are the same kind of decision.** Both remove something that served small models trained for many epochs
  and that only costs something at one epoch over trillions of tokens.

## Summary

| # | Mechanism | Where it sits | Role | Without it | Evidence |
| --- | --- | --- | --- | --- | --- |
| ① | residual stream | an add after each sub-layer | identity path for gradients; shared memory between layers | signal and gradient vanish with depth | ●●● |
| ② | pre-norm + final norm | input of each sub-layer, inside the branch; once at the end | fixed-scale inputs without touching the stream | post-norm needs warm-up, degrades at high lr | ●●● |
| ③ | RMSNorm | every norm | scale normalization at lower cost | LayerNorm works too, slower | ●●○ |
| ④ | gated MLP (SwiGLU) | the channel mixer | multiplicative, conditional features | a plain MLP, slightly worse per parameter | ●●○ |
| ⑤ | RoPE | rotates q and k inside attention | scores depend on relative offset only | absolute positions, poor length extension | ●●● |
| ⑥ | no dropout | — | nothing to regularize in one pass | noise and lost throughput | ●●○ |
| ⑦ | no biases | — | redundant or inert parameters removed | wasted parameters; less stable at scale (PaLM) | ●●○ |
| ⑧ | GQA / MQA / MLA | K and V projections | smaller KV cache, faster decoding | 4–8× more serving memory | ●●● / ●●○ |
| ⑨ | `1/√d_h`, causal mask | the attention scores | softmax not saturated; no peeking ahead | heads saturate; trivial training task | ●●● |
| ⑩ | QK-norm | q and k, before RoPE | bounded logits through training | attention collapse, loss spikes at scale | ●●○ |
| ⑪ | depth-scaled init | `W_o`, `W_down` | bounded stream at init | stream grows like √(2N) | ●●○ |
| ⑫ | tied embeddings | output head | saves `V·d` params | nothing in large models | ●○○ |

## Knowledge gap

**What the slides leave out.** The course covers the 2017 block (`22-attention.pdf`, `L35`). It mentions RoPE and
layer norm but not pre- vs. post-norm, RMSNorm, gating, the dropout change, biases, GQA, QK-norm or initialization
scaling.

**Limits of the checks.** The training table in ② uses MLP stacks without attention, and one seed. It shows the
direction the papers report, not their size. The dropout and bias checks are exact or nearly so. The GQA numbers are
arithmetic.

**Open questions.** Why GLU variants work is still not explained well. Shazeer's own paper attributes it to "divine
benevolence". Whether the pre-norm stream growth should be fixed by norms (Gemma, OLMo), by initialization, or by
changing the residual itself (DeepSeek V4's hyper-connections) is not settled.

**Reading**

- Residual stream: He et al. 2016; Elhage et al. 2021, "A mathematical framework for Transformer circuits".
- Norms: Xiong et al. 2020, "On layer normalization in the Transformer architecture"; Liu et al. 2020,
  "Understanding the difficulty of training Transformers"; Zhang & Sennrich 2019 (RMSNorm).
- MLP: Shazeer 2020, "GLU variants improve Transformer"; Geva et al. 2021, "Transformer feed-forward layers are
  key-value memories"; Narang et al. 2021.
- Positions: Su et al. 2021 (RoPE); Chen et al. 2023 (position interpolation); Peng et al. 2023 (YaRN).
- Dropout and biases: Chowdhery et al. 2022 (PaLM); Liu et al. 2025, "Drop dropout on single-epoch language model
  pretraining".
- K/V sharing: Shazeer 2019, "Fast Transformer decoding: one write-head is all you need"; Ainslie et al. 2023 (GQA);
  DeepSeek-AI 2024, DeepSeek-V2 (MLA).
- Logits and init: Vaswani et al. 2017; Henry et al. 2020, "Query-key normalization for Transformers"; Dehghani et
  al. 2023; Wortsman et al. 2023; Radford et al. 2019 (GPT-2, residual init scaling).
