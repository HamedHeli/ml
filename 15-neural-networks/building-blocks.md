# Building blocks and proven architectures

[back to the folder](README.md) · intuition notebook: [`building_blocks.ipynb`](building_blocks.ipynb)

Most of the architectures that work today are made from about ten blocks, assembled in a few patterns. This page
lists the blocks, what each one is for, and how the well-known architectures combine them. The notebook runs a small
experiment for each block.

## The one skeleton behind most modern networks

```
x ──► Norm ──► token mixer ──► (+) ──► Norm ──► channel mixer ──► (+) ──► next block
 │                              ▲  │                               ▲
 └──────────── residual ────────┘  └─────────── residual ──────────┘
```

- The **residual stream** `x` runs straight through the network. Each block reads from it and *adds* its output
  to it: `x ← x + F(Norm(x))`.
- A **token mixer** moves information *between* positions (words, pixels, time steps). Attention, convolution,
  recurrence and state-space layers are all token mixers.
- A **channel mixer** transforms each position *on its own*: an MLP, a gated MLP (GLU), or a mixture of experts.
- A **normalization** layer in front of each mixer keeps its input at a fixed scale.

"Attention + norm + MLP" is the Transformer's version. Swap the token mixer and you get the other families:

| Architecture | Token mixer | Channel mixer | Norm |
| --- | --- | --- | --- |
| Transformer (GPT, BERT, ViT) | multi-head self-attention | MLP / SwiGLU | LayerNorm / RMSNorm |
| ResNet, ConvNeXt | convolution (3×3, or 7×7 depthwise) | 1×1 convolutions (= a per-pixel MLP) | BatchNorm / LayerNorm |
| MLP-Mixer | an MLP applied across positions | MLP | LayerNorm |
| Mamba | selective state-space scan (+ a short convolution) | gating inside the same block | RMSNorm |
| Mixture-of-experts LLM (Mixtral, DeepSeek-V3) | attention | router + many expert MLPs | RMSNorm |

### A note on the usual summary of the Transformer block

The common summary is: attention gives better token representations, the normalizer keeps representations from
drifting far from each other, and the MLP converts the representation based on the context by asking questions. Most
of this is right. Three refinements make it sharper:

1. **Attention is the only place tokens exchange information.** Each token's output is a weighted average of all
   tokens' **value** vectors. The weights are `softmax(QKᵀ/√d)`, i.e. how well the token's query matches each key.
   The MLP then works on each token separately, with the same weights for every token.
2. **LayerNorm rescales each token's own features; it does not bring tokens closer to each other.** For one token it
   subtracts the mean of that token's features and divides by their standard deviation, then applies a learned scale
   and shift. Two tokens that point in different directions still do afterwards (the notebook shows `[1,2,3,4]` and
   `[4,3,2,1]` coming out as exact opposites). Its purpose is **optimization**: every layer sees inputs of a predictable
   size, so deep stacks train stably and are less sensitive to initialization and learning rate.
3. **The residual connections are the missing piece.** The block is `x + Attention(Norm(x))`, then
   `x + MLP(Norm(x))`. Each sub-layer *adds a correction* to the residual stream instead of replacing it. Without
   residuals, deep Transformers do not train (notebook §1).

The "MLP asks questions" picture can be made precise. Each hidden unit's input weights are a **key** that detects a
pattern in the token. The matching column of the output matrix is a **value** that the unit writes into the residual
stream when the pattern fires. MLP layers act as a large key–value memory (Geva et al. 2021). That is where much of a
language model's factual knowledge is stored.

## The blocks

### Channel mixers (per position)

| Block | What it computes | Why it works | Where |
| --- | --- | --- | --- |
| **Dense / MLP** | `W₂ σ(W₁x)`, usually expanding `d → 4d → d` | universal approximator; the expansion gives room for many feature detectors | everywhere |
| **Activations** | ReLU `max(0, a)`; GELU and SiLU/Swish (smooth versions) | ReLU's gradient is 1 when active, so it doesn't vanish; smooth versions optimize slightly better | ReLU in CNNs; GELU/SiLU in Transformers |
| **Gated MLP (GLU, SwiGLU)** | `W₂ (SiLU(W₁x) ⊙ W₃x)` | multiplicative interactions: "feature *a* matters only if *b* is present" (notebook §4: about 20× lower error on `x₁x₂`) | LLaMA, PaLM, Mistral and most LLMs since 2022 |
| **Mixture of experts** | a router picks top-*k* of *E* expert MLPs per token | parameters grow *E*-fold while compute per token stays fixed | Switch Transformer, Mixtral, DeepSeek-V3 |
| **1×1 convolution** | an MLP applied at each pixel | changes channel count cheaply; mixes channels | ResNet bottlenecks, Inception, MobileNet |

### Token mixers (between positions)

| Block | What it computes | Inductive bias | Cost in length *n* | Where |
| --- | --- | --- | --- | --- |
| **Convolution** | the same small filter slid over every position | locality + weight sharing ⇒ translation equivariance | `O(n)`; receptive field grows with depth | CNNs, audio (WaveNet), U-Nets |
| **Depthwise / separable convolution** | one filter per channel, then a 1×1 conv | same, at a fraction of the parameters | `O(n)` | MobileNet, EfficientNet, ConvNeXt, Mamba's short conv |
| **Dilated convolution** | a filter with gaps | exponentially growing receptive field | `O(n)` | WaveNet, segmentation |
| **Pooling / strided conv** | downsample | builds a hierarchy of scales | `O(n)` | CNN encoders, U-Nets |
| **Self-attention** | `softmax(QKᵀ/√d)V`: every token reads every token, weighted by content | none about order; content-based, all-to-all | `O(n²)` | Transformers |
| **Causal (masked) attention** | each token reads only earlier ones | autoregressive generation | `O(n²)`; `O(n)` per new token with a KV cache | GPT-style decoders |
| **Cross-attention** | queries from one sequence, keys and values from another | conditioning on another input | `O(n·m)` | translation decoders, text-to-image U-Nets |
| **Multi-head, multi-query, grouped-query** | several attention patterns in parallel; MQA/GQA share keys and values across heads | different heads track different relations; MQA/GQA shrink the KV cache | same | all Transformers; GQA in LLaMA 2/3 |
| **RNN / LSTM / GRU** | a hidden state updated token by token; LSTM/GRU **gates** decide what to keep | order; recency | `O(n)` but sequential in training | seq2seq before 2017, small on-device models |
| **State-space / Mamba** | a linear recurrence `h_t = A_t h_{t−1} + B_t x_t`; "selective" = `A_t, B_t` depend on the input | order; compressive memory; gates as in LSTMs | `O(n)`, trained with a parallel scan | Mamba, Jamba and other hybrids |
| **Graph message passing** | each node aggregates its neighbours' messages | the graph's structure; permutation invariance | `O(edges)` | molecules, social networks, AlphaFold's pair stack |

### Structure and stability

| Block | What it does | Why it matters |
| --- | --- | --- |
| **Residual connection** `x + F(x)` | each block learns a correction | identity is easy, gradients flow to early layers; enabled 100+ layers (ResNet 2016) |
| **BatchNorm** | normalize each feature over the batch | made deep CNNs trainable; depends on batch statistics, awkward for sequences and small batches |
| **LayerNorm** | normalize each example's (token's) features | batch-independent; the Transformer default |
| **RMSNorm** | LayerNorm without subtracting the mean | cheaper, works as well; most current LLMs |
| **GroupNorm** | normalize groups of channels per example | small-batch vision, diffusion U-Nets |
| **Pre-norm vs. post-norm** | `x + F(Norm(x))` vs. `Norm(x + F(x))` | pre-norm keeps the residual path clean and trains deep stacks without careful warm-up |
| **Dropout, stochastic depth** | randomly drop units or whole residual branches in training | regularization; stochastic depth for very deep nets |
| **Initialization (He, Glorot; zero-init last layer of a branch)** | choose weight scales that preserve variance | without it deep plain nets explode or vanish (notebook §2); zero-init makes each block start as the identity |

### Inputs, positions and conditioning

| Block | What it does | Where |
| --- | --- | --- |
| **Token embedding** | a lookup table from token id to vector; often shared with the output layer ("weight tying") | every language model |
| **Patch embedding** | cut an image into 16×16 patches and project each to a vector | ViT, DiT |
| **Positional encoding** | sinusoidal or learned vectors added to the tokens | original Transformer, BERT, ViT |
| **RoPE (rotary)** | rotate queries and keys by position-dependent angles so that scores depend only on the relative offset | LLaMA and most current LLMs |
| **ALiBi, relative position biases** | add a distance-dependent bias to attention scores | T5, BLOOM; helps longer contexts |
| **Time / noise-level embedding** | sinusoidal features of the diffusion step → MLP | every diffusion model |
| **FiLM / AdaLN** | a condition predicts per-channel scale and shift (`γ ⊙ x + β`) | diffusion U-Nets and DiT, style transfer |
| **Classifier-free conditioning** | train with the condition randomly dropped, combine both predictions at sampling time | text-to-image models |

## Proven architectures as compositions

| Architecture | Built from | Key idea | Folder |
| --- | --- | --- | --- |
| **MLP** | dense + activation | learned features | [15](README.md) |
| **LeNet → AlexNet → VGG** | conv + ReLU + pooling, then dense | hierarchy of local features | [16](../16-convolutional-networks/) |
| **ResNet** | conv–BN–ReLU blocks with residual connections; bottleneck 1×1–3×3–1×1 | depth through residuals | [16](../16-convolutional-networks/) |
| **MobileNet / EfficientNet** | depthwise separable convs, inverted bottlenecks, squeeze-and-excitation (channel gating) | accuracy per FLOP | – |
| **ConvNeXt** | depthwise 7×7 conv + LayerNorm + MLP (4× expansion) + residual | a CNN redesigned with Transformer-block habits; matches ViTs | – |
| **U-Net** | conv encoder (downsample) + decoder (upsample) + skip connections at each scale (+ attention at low resolution) | global context *and* pixel-level detail | [18](../18-diffusion-and-image-generation/) |
| **Autoencoder / VAE** | encoder → bottleneck → decoder | compression, generation | [12](../12-variational-inference-and-vaes/) |
| **Seq2seq RNN + attention** | LSTM encoder, LSTM decoder, attention over encoder states | translation before 2017 | [17](../17-sequence-models-and-transformers/) |
| **Transformer encoder (BERT)** | embeddings + [self-attention + MLP] × L, no mask | bidirectional context | [17](../17-sequence-models-and-transformers/) |
| **Transformer decoder (GPT, LLaMA)** | embeddings + RoPE + [causal attention + SwiGLU] × L, RMSNorm, pre-norm | next-token prediction at scale | [17](../17-sequence-models-and-transformers/) |
| **Encoder–decoder (T5, Whisper)** | encoder + decoder with cross-attention | sequence-to-sequence | [17](../17-sequence-models-and-transformers/) |
| **Vision Transformer (ViT), DiT** | patch embedding + Transformer encoder (DiT: AdaLN conditioning) | Transformers for images and diffusion | [16](../16-convolutional-networks/), [18](../18-diffusion-and-image-generation/) |
| **CLIP, two-tower models** | separate encoders mapped to one embedding space, contrastive loss | match images and text | [19](../19-representation-learning-and-llms/) |
| **Mamba / hybrid SSM** | [selective SSM + gating] × L, sometimes interleaved with attention | linear-time long context | [17](../17-sequence-models-and-transformers/) |
| **Mixture-of-experts Transformer** | attention + routed expert MLPs | capacity without compute | – |
| **Graph neural network** | message passing + MLP per node, residuals | learning on graphs | [14](../14-graphical-models/) (graphs, not GNNs) |

## How to choose (rules of thumb)

- **Start from the skeleton**: residual stream, pre-norm, token mixer, channel mixer. The common recipes (AdamW,
  warm-up, cosine decay) are tuned for it.
- **Pick the token mixer by data:** grids with local structure (images, audio) → convolutions, or attention on
  patches when data are plentiful. Sequences where any position may matter → attention. Very long sequences →
  state-space layers or hybrids. Graphs → message passing.
- **Inductive bias trades against data:** convolutions win with little data; Transformers catch up and overtake as
  data grow (ViT beat ResNets only after pre-training on 14M–300M images).
- **Scale the channel mixer** (wider MLPs, GLU, or experts) for more capacity; scale depth only with residuals and
  normalization in place.

## Knowledge gap

**What the slides leave out:** almost all of this page. The slides cover dense layers, ReLU, skip connections (L31),
convolution (L32–33), RNNs, LSTMs and attention (CPSC 440 lectures 21–22). Normalization layers, gated MLPs, mixtures
of experts, state-space models, RoPE, conditioning mechanisms and the design history of CNNs after AlexNet are not
covered.

**Reading:** He et al. 2016, "Deep residual learning"; Ioffe & Szegedy 2015 (BatchNorm); Ba, Kiros & Hinton 2016
(LayerNorm); Xiong et al. 2020, "On layer normalization in the Transformer architecture" (pre-norm); Vaswani et al.
2017; Shazeer 2020, "GLU variants improve Transformer"; Geva et al. 2021, "Transformer feed-forward layers are
key-value memories"; Su et al. 2021 (RoPE); Ronneberger et al. 2015 (U-Net); Liu et al. 2022 (ConvNeXt); Tolstikhin et
al. 2021 (MLP-Mixer); Gu & Dao 2023 (Mamba); Fedus, Zoph & Shazeer 2022 (Switch Transformer); Prince,
*Understanding Deep Learning* ch. 10–13 (free online).
