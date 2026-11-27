# 17 Sequence models: RNNs, LSTMs, attention and Transformers

**CPSC 440 lectures 21, 22 · CPSC 340 L35** · [back to the graph](../README.md)

**What is learned:** a map from a sequence of tokens to predictions (the next token, a label, another sequence), with
a learned mechanism for carrying information across positions: a recurrent state, gated memory, or attention.

## Lectures

| Slides | Covers |
| --- | --- |
| `21-rnns.pdf` | HMMs for text, word representations, RNNs, inference and learning, deep and bidirectional RNNs, seq2seq for translation, beam search, tokenization and byte-pair encoding, exponential forgetting, state-space models, LSTMs, PixelRNN, image and video captioning |
| `22-attention.pdf` | problems with RNNs, attention over encoder states, keys / queries / values, multi-modal attention, the Transformer, self-attention, position encodings and RoPE, multi-head attention, MLPs, layer norm, encoder (BERT) and decoder (GPT-1 to GPT-4), Transformers vs. RNNs / SSMs, vision Transformers |
| `L35 - Transformers.pdf` (CPSC 340) | attention as a building block, convolutions and dilated convolutions for sequences, the Transformer block, stacking blocks, vision Transformers, LLMs, few-shot prompting and prompt engineering, jailbreaks, "does the model understand?" |

## The models

### Recurrent neural networks

`h_t = tanh(W_h h_{t−1} + W_x x_t)`, `ŷ_t = softmax(V h_t)`. A discriminative cousin of the HMM with a deterministic,
continuous state; inference is one forward pass. Trained by backpropagation through time. Because the gradient through
`T` steps multiplies `T` Jacobians, it **vanishes or explodes exponentially** in the time lag; gradient clipping fixes
explosion, not vanishing. Deep (stacked) and bidirectional variants; **seq2seq** encodes a source sequence into a state
and decodes a target sequence of different length (translation).

### LSTMs

Add a memory cell `c_t` with soft gates: `c_t = f_t ⊙ c_{t−1} + i_t ⊙ g_t`, `h_t = o_t ⊙ tanh(c_t)`. With the forget gate
near 1 the cell is copied forward unchanged, so information and gradients survive long lags.

### Tokenization and decoding

**Byte-pair encoding** learns a vocabulary by repeatedly merging the most frequent adjacent pair: common words become
one token, rare words split into pieces, nothing is out-of-vocabulary. Tokenization choices visibly affect model
behaviour (spelling, arithmetic). **Beam search** keeps the `B` best partial outputs at each step, between greedy
(`B = 1`) and exact search (`B = V^T`).

### Attention and Transformers

**Attention**: `softmax(qKᵀ/√d) V`, a soft lookup of values by matching a query against keys. First added so a seq2seq
decoder could look back at all encoder states instead of one bottleneck vector. **Self-attention** makes every token a
query, key and value (`Q = XW_q`, …): every position reads from every other in one step, at `O(T²d)` cost but fully
parallel.

A **Transformer block** is `X + MultiHeadAttention(LayerNorm(X))` then `X + MLP(LayerNorm(X))`. Attention is
permutation-equivariant, so **position encodings** (sinusoidal, learned, or rotary RoPE) inject order. A causal mask
gives a decoder (GPT, trained to predict the next token); no mask gives an encoder (BERT, trained to fill in masked
tokens). Vision Transformers apply the same blocks to image patches.

**Modern LLM architectures:** [LLM architecture design](llm-architecture-design.md) codes the 77 readable models of
Raschka's LLM Architecture Gallery (GPT-2 to mid-2026), sorts them into eight families, compares them in one table,
and turns the trends into a step-by-step design process: why each layer is there (pre-norm, RMSNorm, RoPE, SwiGLU,
GQA/MLA, sliding-window and linear-attention hybrids, MoE, QK-norm, no dropout), with the strength of the evidence
for each.

**Inside the block:** [inside a modern LLM block](llm-block-mechanisms.md) takes the settled choices one at a time
(residual stream, pre-norm, RMSNorm, SwiGLU, RoPE, no dropout, no biases, GQA, plus the `1/√d` scale, causal mask,
QK-norm, depth-scaled init, final norm and tied embeddings): what each computes, where it sits, the role it plays
and what breaks without it, each with a small numerical check.

**Positions and RoPE:** [how RoPE positions tokens](rope-positions.md) answers where position 0 is, how the counter
and the KV cache work during generation, why each token gets 64 angles at 64 speeds (the clock with many hands, and
why Llama 3 raised the base to 500 000), a worked sentence, and context extension (position interpolation, NTK, YaRN).

**Worked example:** [one token through one Transformer block](worked-example.md) follows "tree" in "I am the tree I saw
yesterday" through Q/K/V, softmax, the weighted sum of Values, residual + LayerNorm and the 4 → 16 → 4 MLP, with numbers
you can check by hand.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`rnn.py`](rnn.py) | vanilla RNN and LSTM with `lax.scan`, gradient clipping, a "recall the marked first token after `T` distractors" task |
| [`transformer.py`](transformer.py) | causal multi-head self-attention, sinusoidal position encodings, pre-norm blocks with GELU MLPs, the same recall task, attention inspection |
| [`tokens_and_decoding.py`](tokens_and_decoding.py) | byte-pair encoding (learn merges, encode new words), beam search vs. greedy vs. exact decoding |
| [`llm_architectures.py`](llm_architectures.py) | the gallery's architectures as data, the family classifier, adoption counts by period, the comparison table and the figures in [llm-architecture-design.md](llm-architecture-design.md) |
| [`llm_block_mechanisms.py`](llm_block_mechanisms.py) | one check per mechanism in [llm-block-mechanisms.md](llm-block-mechanisms.md): signal through depth, post- vs pre-norm training, RMSNorm properties, score scaling, QK-norm, RoPE's relative property, dropout regimes, inert attention biases, GQA and KV-cache sizes |
| [`rope_positions.py`](rope_positions.py) | the checks in [rope-positions.md](rope-positions.md): per-pair frequency table, clock uniqueness, relative-only scores, KV-cache decoding vs full recompute, the worked sentence, PI / NTK / YaRN angle ranges |
| [`worked_example.py`](worked_example.py) | the hand calculation in [worked-example.md](worked-example.md): attention, residual + LayerNorm, 4 → 16 → 4 MLP |

```bash
python 17-sequence-models-and-transformers/rnn.py                   # about 2 min on a CPU
python 17-sequence-models-and-transformers/transformer.py           # about 1 min
python 17-sequence-models-and-transformers/tokens_and_decoding.py
python 17-sequence-models-and-transformers/rope_positions.py      # under a second
python 17-sequence-models-and-transformers/worked_example.py      # instant
python 17-sequence-models-and-transformers/llm_block_mechanisms.py # about 2 min
python 17-sequence-models-and-transformers/llm_architectures.py   # instant; rewrites the CSV and figures/llm_*.png
```

Recall the first of `T` tokens (8 symbols, chance = 0.125), test accuracy after the same training budget:

| `T` | RNN | LSTM | Transformer |
| --- | --- | --- | --- |
| 20 | 1.000 | 1.000 | not run |
| 80 | 0.875 | **1.000** | not run |
| 150 | 0.129 | 0.119 | **1.000** |

The trained Transformer's last token puts 0.998 of one layer-2 head's attention on position 0 (uniform would be
0.007): it reads the answer directly instead of carrying it through 150 steps.

| Other results | |
| --- | --- |
| BPE, 15 merges on a small corpus | `lowest` → one token; `slowly` → `s`, `low`, `l`, `y`; unseen `unwiderable` → `u n wide r a b l e` |
| finding the most probable length-6 sequence in 200 random Markov models | greedy 66%, beam 2: 88%, beam 5: 97% |

## Knowledge gap

**What the slides leave out**

- **The LSTM needed a forget-gate bias to work here.** With the common bias of 1, the LSTM failed at `T = 40` in this
  experiment; with a bias of 3 it solved `T = 80`. Initialization details like this (and chrono-initialization) decide
  whether gated RNNs learn long dependencies at all; the slides present LSTMs as simply solving the problem.
- **GRUs**, and why gating is a learned, input-dependent version of the state-space recurrence.
- **Training Transformers**: learning-rate warm-up, AdamW, pre-norm vs. post-norm stability (see
  [LLM architecture design](llm-architecture-design.md) for how today's LLMs place their norms), mixed precision, and the
  KV-cache that makes autoregressive decoding `O(T)` per token.
- **Efficient attention** for long contexts (FlashAttention, sparse/linear attention) and modern state-space models
  (S4, Mamba), which the slides mention as an RNN revival without detail.
- **Scaling laws** (Kaplan et al. 2020; Hoffmann et al. 2022): loss falls as a power law in parameters, data and compute,
  which is the quantitative reason "huge models" work.
- **Sampling** from language models (temperature, top-k, nucleus sampling) as opposed to beam search, and why beam
  search produces bland, repetitive text in open-ended generation.
- **What attention heads compute**: induction heads and in-context learning mechanisms (Olsson et al. 2022).

**Prerequisites to shore up**

- Softmax and its Jacobian; matrix shapes of batched multi-head attention (`n × h × T × d_h`).
- Backpropagation through time and why products of Jacobians vanish ([15](../15-neural-networks/)).

**Open questions / further study**

- Does next-token prediction lead to "understanding"? L35 closes on this question, and
  [19](../19-representation-learning-and-llms/) continues it.
- Reading: Vaswani et al. 2017, "Attention is all you need"; Hochreiter & Schmidhuber 1997; Karpathy, "The
  unreasonable effectiveness of recurrent neural networks" and nanoGPT; Phuong & Hutter 2022, "Formal algorithms for
  Transformers"; Prince, *Understanding Deep Learning* ch. 12.

## Connections

- ← [13 Markov chains and HMMs](../13-markov-chains-and-hmms/): RNNs replace the discrete hidden chain with a learned
  continuous state; beam search and Viterbi solve the same decoding problem for different models.
- ← [15 Neural networks](../15-neural-networks/): the same layers and training, unrolled over time.
- ← [09 word2vec](../09-latent-factor-models/): learned token embeddings are the input layer.
- ↔ [16 CNNs](../16-convolutional-networks/): convolutions are local, fixed-size attention; ViTs bring attention to images.
- → [19 Representation learning and LLMs](../19-representation-learning-and-llms/): pre-training, BERT vs. GPT objectives,
  fine-tuning and RLHF.
