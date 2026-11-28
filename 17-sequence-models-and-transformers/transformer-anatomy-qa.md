# Transformer anatomy, question by question: gates, MLPs, encoders, the KV cache and Llama 3 8B's shapes

[back to the folder](README.md) · each mechanism in depth: [inside a modern LLM block](llm-block-mechanisms.md) ·
positions and decoding: [how RoPE positions tokens](rope-positions.md) · numerical checks:
[`transformer_anatomy.py`](transformer_anatomy.py)

Written from a chat on 2026-11-27. The questions were:

1. Is a gated MLP a gate that decides whether a weight matrix gets used for a token?
2. Is a regular MLP `σ(W₁x) + b`?
3. Are `W_up` and `W_down` an encoder and a decoder?
4. Are chat LLMs decoder-only? Is BERT an encoder because it predicts masked words, and was it trained for NER?
5. Does a chatbot get its vectors from BERT and then decode them?
6. Does every token have its own `W_Q`, `W_K`, `W_V`, and is that what the KV cache stores?
7. Is Llama 3 8B 32 copies of one block, with every K/V head shared by 32 query heads?
8. Is the model width, 4096, the size of the largest matrix?

Short answers: (1) no, that is Mixture of Experts; a SwiGLU gate scales hidden features softly and uses every
weight; (2) the bias goes inside the activation, and there are two layers: `W₂ σ(W₁x + b₁) + b₂`; (3) no, it is
closer to a key–value memory, and it expands rather than compresses; (4) yes, and what makes BERT an encoder is
bidirectional attention; NER is a fine-tuning task; (5) no, the decoder makes its own vectors from its own
embedding table; (6) the weights are shared, each token gets its own vectors, and the cache stores past keys and
values; (7) yes to 32 blocks, but each K/V head serves 4 query heads; (8) no, 4096 is the width of the residual
stream, and the MLP and vocabulary matrices are much wider.

```bash
python 17-sequence-models-and-transformers/transformer_anatomy.py     # under a second, NumPy only
```

The checks use small random matrices, except the Llama 3 8B numbers, which come from the model's published
configuration. The parameter count they give, 8.030 B, matches Meta's published 8.03 B, which is a check that the
shapes in [Q8](#8-is-the-model-width-4096-the-largest-matrix) are right.

## The block these questions are about

One layer of a decoder-only LLM, Llama 3 style (the annotated version with every mechanism is
[here](llm-block-mechanisms.md#the-block-with-every-mechanism-in-its-place)):

```text
x = Embed(tokens)                                   one 4096-vector per token: the residual stream
repeat 32 times, each with its own weights:
    x = x + Attention(RMSNorm(x))                   tokens read from earlier tokens   (Q6, Q7)
    x = x + W_down · (SiLU(W_gate h) ⊙ W_up h)      each token on its own, h = RMSNorm(x)   (Q1–Q3)
logits = Unembed(RMSNorm(x_last))                   one score per vocabulary entry, then sample   (Q5)
```

This is a **pre-norm residual block**: normalize, transform, add back to the stream. Almost every LLM since 2020
uses it.

## 1. Is a gated MLP a gate on which weights are used?

No. In a gated MLP like SwiGLU, every weight is multiplied by `x` for every token. The gate works on the *hidden
activations*, not on which weights get used. With a hidden size `h` (Llama 3 8B: 14336), in
`W_down · (SiLU(W_gate x) ⊙ W_up x)`:

- **`W_up x`** gives `h` candidate features: the content.
- **`SiLU(W_gate x)`** gives `h` gate values, one per feature, also computed from `x`.
- **`⊙`** multiplies them element by element, so feature `j` is scaled by gate `j`. A gate near 0 mutes that
  feature *for this token*.
- **`W_down`** maps the `h` gated features back to the model width.

So the gate is soft, it changes from token to token, and it acts one feature at a time. It decides how much of each
hidden feature passes through. "Gate" is a loose word here: SiLU is not a 0-to-1 switch. It dips to −0.28 for
negative inputs and grows without bound for positive ones (`SiLU(4) = 3.93`), so the gate can also amplify. The
original GLU used a sigmoid, which is a true 0-to-1 gate.

At the same hidden width, three matrices cost 1.5× the compute of two. In practice the hidden width is shrunk to
about `8/3 · d` so that parameters and compute match a plain `4d` MLP (see
[④ in the mechanisms note](llm-block-mechanisms.md#-gated-mlp-swiglu-)). The benefit is expressiveness: the output
is a product of two linear views of `x`, and that trains better per parameter than a single activation.

**What you had in mind is Mixture of Experts (MoE).** A router scores `E` expert MLPs (each usually a SwiGLU block)
and sends each token to only the top `k`, for example 2 of 64. The other experts' weights are never multiplied by
that token, and that is where the compute saving comes from.

| check 1 | SwiGLU | MoE, top 2 of 64 |
| --- | --- | --- |
| fraction of the MLP weights a token is multiplied by | 1 | 2/64 = 0.031 |
| what the gate decides | how much of each of `h` features | which `k` whole MLPs run |
| hard or soft | soft, continuous | hard choice (the chosen ones are weighted softly) |

With random weights, the two tokens in check 1 nearly close (|gate| < 0.05) almost entirely different sets of
features (overlap 0.02), which is the "per token, per feature" point in numbers.

## 2. Is a regular MLP `σ(W₁x) + b`?

Close, but the bias goes *inside* the activation, and a Transformer's MLP has two layers:

```text
MLP(x) = W₂ · σ(W₁x + b₁) + b₂
```

- `W₁` expands `x` to a hidden size `h`, often `4d`. `σ` is the nonlinearity. `W₂` projects back down to `d`.
- The original Transformer used ReLU for `σ`; GPT-2 and BERT used GELU. SiLU on its own is uncommon; it mostly
  appears inside SwiGLU.
- Modern LLMs drop the biases (see [⑦](llm-block-mechanisms.md#-no-biases-)).

**SwiGLU** replaces `σ(W₁x)` with `SiLU(W_gate x) ⊙ W_up x` and keeps the down-projection. So it is the same
two-layer shape with a gated hidden layer in the middle. "Up" and "down" refer to the *dimension* (`d → h → d`),
not to a position in the network.

## 3. Are `W_up` and `W_down` an encoder and a decoder?

Only loosely, and the names are worth avoiding, for two reasons:

- In Transformers, "encoder" and "decoder" mean whole stacks of layers (BERT is an encoder, GPT is a decoder).
  `W_up` and `W_down` live inside *every* block of both.
- It is the reverse of an autoencoder ([L34](../12-variational-inference-and-vaes/)). An autoencoder squeezes `x`
  into a *smaller* code; the MLP expands `x` into a *bigger* hidden layer (`d → ~4d`) and projects it back.

A better picture is a **key–value memory** (Geva et al. 2021). Each row of `W_up` (and `W_gate`) is a pattern
detector, a "key" that fires when `x` looks like something, such as "a capital-city context". The matching column
of `W_down` is the "value" that is added to the residual stream when that detector fires, such as "boost *Paris*".
So the MLP looks up stored associations and adds them in.

The "encode into features, decode back" intuition is closest to how **sparse autoencoders** are used to *interpret*
these hidden features, which is a tool built on top of a trained model, not part of it.

## 4. Are chat LLMs decoder-only? What makes BERT an encoder?

**Chat LLMs.** GPT-2 and GPT-3, Llama, Mistral, Qwen and DeepSeek publish decoder-only architectures. Google's
Gemini reports describe Transformer decoders. Anthropic (Claude) and OpenAI (GPT-4 and later) have not published
their architectures. They generate text one token at a time like the others and are generally assumed to be
decoder-only, but that is an inference, not a published fact.

**BERT.** It was trained with *masked language modelling* (predict hidden words) plus a next-sentence objective. But
what makes it an encoder is the attention, not the objective:

- **Encoder:** every token attends to the tokens on both sides (bidirectional attention). That is what makes masked
  language modelling possible: the hidden word is predicted from context on both sides.
- **Decoder:** a causal mask means each token sees only earlier tokens. That fits next-token prediction, which is
  what lets the model generate.

Check 2 makes the difference concrete: change token 5 of 8. In the decoder, the outputs at positions 0–4 do not move
at all (difference exactly 0). In the encoder, they change by up to 0.34.

**NER** (named-entity recognition) is not how BERT was trained. It is a downstream task you *fine-tune* BERT for,
like classification and extractive question answering. Encoders suit tasks that need the whole input at once;
decoders are built to generate.

## 5. Does a chatbot get its vectors from BERT?

No. A decoder-only chatbot has no encoder and no BERT step. It builds its own vectors:

1. **Tokenize** the prompt and look up each token in the model's *own* embedding table, learned in training.
2. **Run them through `N` decoder blocks** (causal attention + MLP). Each block refines every token's vector in the
   residual stream. Because of the causal mask, a vector only ever "knows" the tokens to its left.
3. **Take the last token's final vector**, multiply it by the output (unembedding) matrix to get one score per
   vocabulary entry, softmax the scores into probabilities, and sample the next token.
4. **Append that token and repeat**, using the KV cache ([Q6](#6-does-every-token-have-its-own-w_q-w_k-w_v-what-does-the-kv-cache-store)).

So understanding and generating happen in the same stack. The early layers do "understanding" work much as an
encoder would, only with causal attention.

**Where the idea does exist:**

- **Encoder–decoder models** (the original 2017 Transformer, T5, many translation models): an encoder reads the
  input, and the decoder attends to the encoder's vectors through *cross-attention* while it generates.
- **Retrieval (RAG) and search:** BERT-like embedding models turn documents into vectors to *find* relevant text. That
  text is then pasted into the chatbot's prompt as ordinary tokens, not passed in as vectors.

## 6. Does every token have its own `W_Q`, `W_K`, `W_V`? What does the KV cache store?

Two separate ideas.

**The weight matrices are shared.** Every token uses the same `W_Q`, `W_K`, `W_V` in a given layer. Each token gets
its *own vectors* from them: `qᵢ = W_Q xᵢ`, `kᵢ = W_K xᵢ`, `vᵢ = W_V xᵢ`.

**The KV cache is a speed trick for generation.** To generate token `t`, its query `q_t` is compared with the keys
of all earlier tokens, and their values are mixed. Those earlier `kᵢ` and `vᵢ` never change: the causal mask means
later tokens cannot affect them, and RoPE has already rotated the keys by their positions
([RoPE note §3](rope-positions.md#3-generation-prefill-the-kv-cache-and-a-counter)). So instead of recomputing them at
every step, they are **stored**, in every layer and every K/V head. Each new token computes only its own `q`, `k`,
`v`, appends its `k` and `v` to the cache, and attends over the cache. Past queries are never needed again, so
there is no Q cache.

Check 3 prefills 6 tokens and decodes 10 more one at a time. The result matches a full recompute to 1.1 × 10⁻¹⁵,
with 16 token projections instead of 121.

- **Without the cache**, step `t` redoes the projections and MLPs of all `t` earlier tokens, so generating `n`
  tokens costs about `n²/2` token passes instead of `n`. (The attention scores themselves still grow with the
  context: each new token compares against all cached keys.)
- **The cost is memory:** layers × K/V heads × head size × 2 (K and V) × bytes, per token. At long contexts this is
  many GB, and it often limits context length and batch size more than the weights do.

**The idea of sharing does exist, one level up.** **Multi-query and grouped-query attention (MQA, GQA)** let several
query heads *share* one key/value head: the queries differ, the K and V are shared. Llama 2 70B and all Llama 3
models use GQA, which shrinks the KV cache 4–8× (details in
[⑧](llm-block-mechanisms.md#-shared-kv-heads-mqa-and-gqa--gqa--mla)).

## 7. Is Llama 3 8B 32 copies of one block? Do 32 query heads share each K/V head?

**The first part is right.** A *layer* is one full pre-norm block: norm → attention → add, norm → MLP → add. Llama
3 8B stacks 32 of them, each with its own weights. Layers run *in sequence*: layer 2 reads what layer 1 wrote.
Heads run *side by side* inside one layer's attention, on the same input, and their outputs are concatenated and
mixed by `W_O`.

**The second part is off.** There are **32 query heads in total, sharing 8 K/V heads**, so each K/V head serves a
group of **4 query heads**:

| per layer | heads | size each | total |
| --- | --- | --- | --- |
| queries | 32 | 128 | 4096 (= model width) |
| keys | 8 | 128 | 1024 |
| values | 8 | 128 | 1024 |

Query heads 0–3 use K/V head 0, heads 4–7 use K/V head 1, and so on (check 4 prints the full map). Each query head
still asks its own question; four of them look at the same keys and values.

**Why:** a 4× smaller KV cache. Per token, 32 layers × 8 heads × 128 × 2 (K and V) = 65,536 numbers = 128 KiB in
16-bit. At the 8,192-token training context that is 1 GiB, against 4 GiB if every query head had its own K/V. At
Llama 3.1's 128K context, it is 16 GiB against 64 GiB.

## 8. Is the model width, 4096, the largest matrix?

No. **Model width (`d_model`)** is the length of each token's vector in the residual stream, the vector that flows
from layer to layer. Every block takes in 4096 numbers per token and gives back 4096. It is the "bus width" every
block plugs into. The matrices are shaped around it, and several are much larger (check 5, from the published
config):

| matrix | shape (in × out) | parameters | per |
| --- | --- | --- | --- |
| `W_Q`, `W_O` | 4096 × 4096 | 16.8 M each | layer |
| `W_K`, `W_V` | 4096 × 1024 (8 K/V heads × 128) | 4.2 M each | layer |
| `W_gate`, `W_up` | 4096 × 14336 (MLP hidden) | 58.7 M each | layer |
| `W_down` | 14336 × 4096 | 58.7 M | layer |
| embedding | 128,256 (vocab) × 4096 | 525 M | model |
| output (unembedding) | 4096 × 128,256 | 525 M | model, not tied to the embedding |

(PyTorch stores each weight transposed, as out × in, so `k_proj.weight` is listed as 1024 × 4096.)

Adding up: 41.9 M attention + 176.2 M MLP per layer, so the MLP is 81% of each layer. Times 32 layers, plus 1.05 B
for the two embedding matrices, gives **8.030 B parameters**, about 15 GiB in bf16.

## Corrections to the original chat

- The chat called SiLU's output a gate that is "roughly 0 (closed) or roughly passthrough (open)". SiLU is not
  bounded: it dips to −0.28 and grows linearly for large inputs, so it scales and can amplify. A sigmoid gate (the
  original GLU) is the one that stays between 0 and 1.
- The chat said the gated MLP "costs extra compute rather than saving any, because there are three matrices instead
  of two". That is true at equal hidden width, but in practice the width is cut to about `8/3 · d` to keep cost
  equal. Llama 3 8B goes wider (3.5 · d), so it does spend more.
- The chat said GPT is decoder-only "the same as" Llama and DeepSeek. That is published for GPT-2 and GPT-3; GPT-4's
  architecture, like Claude's, is not.
- The chat called the waste without a KV cache "roughly quadratic". More precisely, the *recomputed projections and
  MLPs* grow quadratically in the number of generated tokens; attention over the context grows quadratically either
  way.

## References

- Vaswani et al. 2017, *Attention Is All You Need* (encoder–decoder, ReLU MLP).
- Devlin et al. 2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Raffel et al. 2020, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5).
- Radford et al. 2019, *Language Models are Unsupervised Multitask Learners* (GPT-2); Brown et al. 2020 (GPT-3).
- Shazeer 2020, *GLU Variants Improve Transformer*; Dauphin et al. 2017 (GLU).
- Geva et al. 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*.
- Shazeer et al. 2017, *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*; Fedus et
  al. 2022, *Switch Transformers*.
- Shazeer 2019, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA); Ainslie et al. 2023 (GQA).
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Gemini Team 2023, *Gemini: A Family of Highly Capable Multimodal Models*.
- Llama Team 2024, *The Llama 3 Herd of Models*; `config.json` of `meta-llama/Meta-Llama-3-8B`.
- Bricken et al. 2023, *Towards Monosemanticity* (sparse autoencoders on MLP features).
