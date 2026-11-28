# How RoPE positions tokens: indices, angles, the KV cache and long contexts

[back to the folder](README.md) · the short version: [⑤ RoPE in inside a modern LLM block](llm-block-mechanisms.md#-rope-rotary-position-embedding-)
· numerical checks: [`rope_positions.py`](rope_positions.py)

Written from a chat on 2026-11-27. The questions were:

1. Is a token's position counted from the start of the sentence, the paragraph or the chapter?
2. How does that position get into inference, when the model generates one word after another?
3. With a 128K training window, is 360° divided into 128 000 parts, so each token gets one angle?
4. What does that look like for one sentence, "She was there all the time, before he got there"?

Short answers: (1) none of those, it is the index in the context window; (2) a counter that goes up by one per
generated token, with rotated keys kept in the KV cache; (3) no, each token gets 64 angles that turn at 64 different
speeds, and the window length is not in the formula at all; (4) worked through [below](#a-worked-sentence).

```bash
python 17-sequence-models-and-transformers/rope_positions.py      # under a second, NumPy only
```

## 1. A position is an index in the context window

A token's position is its index among the tokens the model is processing right now, starting at 0 with the first
token fed in. The model has no built-in notion of sentence, paragraph or chapter. If a system prompt comes first,
position 0 is the system prompt's first token, and the user's question starts wherever the system prompt ended.

Structure reaches the model only through tokens: a newline, a `#` heading, a "Chapter 3". The model learns to read
those, the same way it learns to read any other word.

Two refinements from training practice:

- Pretraining packs several documents into one long sequence. Some recipes, Llama 3 among them, add an attention
  mask so a token cannot attend across a document boundary. The index still just counts through the sequence.
- Because RoPE scores depend only on offsets (section 2), moving the whole text later in the window does not change
  how its tokens attend to each other. So "counted from where" matters less than it seems.

## 2. The rotation, and why it gives relative position

Take one attention head with `d_h = 128`. Split the query and the key into 64 pairs of coordinates. For a token at
position `m`, rotate pair `i` by the angle

`m · θᵢ`,  with  `θᵢ = base^(−2i/d_h)`,  `i = 0 … 63`.

Base 10 000 in the original RoFormer paper (Su et al. 2021); 500 000 in Llama 3. The values `v` are not rotated, and
nothing is added to the token embedding.

A rotation by `a` followed by the inverse of a rotation by `b` is a rotation by `a − b`: `R(m)ᵀ R(n) = R(n − m)`. So
the attention score between a query at `m` and a key at `n` is

`score(m, n) = (R(m) q)ᵀ (R(n) k) = qᵀ R(n − m) k`.

You only ever hand the model absolute indices, but the score sees only the distance `m − n`. Check 3 in the script
puts the same `q`, `k` 5 tokens apart at positions 10, 60 000 and 131 000: the three scores agree to 3 × 10⁻¹².

## 3. Generation: prefill, the KV cache and a counter

1. **Prefill.** The prompt's `N` tokens get positions `0 … N−1`, all at once. Every layer computes their keys and
   values, rotates the keys by their positions, and stores both in the KV cache.
2. **Decode a token.** The new token gets position `N`. Its query and key are rotated by `N`. Its key and value are
   appended to the cache, and its query attends over the whole cache.
3. **Repeat** with `N+1`, `N+2`, … The position is a counter that goes up by one per generated token.

Nothing in the cache is ever recomputed: a key at position `n` was rotated by `n` when it was made, and that stays
correct forever. This is a real advantage over schemes that add position to the input, and it is why the cache
stores keys *after* rotation. Check 4 prefills 12 tokens and decodes 8 more at positions 12 … 19; the outputs match
a full recompute to 1.6 × 10⁻¹⁵.

With batching, each sequence keeps its own counter. A left-padded short prompt either gets its positions shifted to
start at 0 at its first real token or keeps the offset, which does not matter for scores between its own tokens.

## 4. One token, 64 angles: the clock with many hands

The most common misreading: "with a 128K window, 360° is split into 128 000 steps, one per token". That is not how
it works. The window length appears nowhere in `m · θᵢ`. Each pair turns at a fixed speed `θᵢ` radians per token,
decided before training starts, and the speeds fall geometrically from pair 0 to pair 63.

Check 1, with `d_h = 128` and a 131 072-token window:

| base | pair `i` | degrees per token | tokens per full turn | turns in 128K |
| --- | ---: | ---: | ---: | ---: |
| 10 000 | 0 | 57.3 | 6.3 | 20 861 |
| 10 000 | 32 | 0.57 | 628 | 209 |
| 10 000 | 63 | 0.0066 | 54 410 | 2.4 |
| 500 000 | 0 | 57.3 | 6.3 | 20 861 |
| 500 000 | 16 | 2.15 | 167 | 784 |
| 500 000 | 32 | 0.081 | 4 443 | 29.5 |
| 500 000 | 48 | 0.0031 | 118 143 | 1.1 |
| 500 000 | 63 | 0.00014 | 2 559 196 | 0.05 |

Think of 64 clock hands. The fastest is a second hand: it turns 1 radian per token and comes around every 6.3 tokens,
so on its own it cannot tell offset 0 from offset 44 (check 2: distance 0.018, where 2 is the largest possible). The
slowest is an hour hand that barely moves inside the window; it gives coarse "near or far" information. No single
hand is unique, but the set is: over every offset from 1 to 131 071, the offset whose 64 angles come closest to
offset 0 is offset 1, the neighbour. Nothing far away masquerades as near.

The fast hands carry fine, local order ("the word right before me"); the slow hands carry distance at the scale of
paragraphs and pages. A head can learn to put weight in its `q` and `k` on whichever pairs suit its job.

**What the window length does constrain.** If the slowest hand completes a full turn inside the window, two tokens
a full turn apart look the same to it, and far looks like near. With base 10 000 the slowest hand turns every
54 410 tokens, about 2.4 times inside 128K. Raising the base to 500 000 stretches that to 2.56M tokens, so inside
128K it covers 5% of a turn. That is the whole reason for Llama 3's larger base.

**Long-range decay.** With `q = k`, the score falls as the distance grows, because the hands drift out of phase
(check 5, `q = k` = all ones, base 500 000):

| offset | 0 | 1 | 16 | 256 | 4 096 | 16 384 | 65 536 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 128.0 | 125.2 | 93.4 | 71.6 | 49.0 | 17.1 | 5.8 |

This is a mild default preference for nearby tokens. Learned `q` and `k` can override it.

## A worked sentence

"She was there all the time, before he got there", one token per word for readability (a real BPE tokenizer
splits differently, e.g. " there" with its leading space, but the bookkeeping is the same). Base 500 000; angles in
degrees, taken mod 360 (check 6):

| pos | token | pair 0 | pair 8 | pair 16 | pair 63 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | She | 0.0 | 0.0 | 0.00 | 0.0000 |
| 1 | was | 57.3 | 11.1 | 2.15 | 0.0001 |
| 2 | there | 114.6 | 22.2 | 4.31 | 0.0003 |
| 3 | all | 171.9 | 33.3 | 6.46 | 0.0004 |
| 4 | the | 229.2 | 44.4 | 8.62 | 0.0006 |
| 5 | time | 286.5 | 55.6 | 10.77 | 0.0007 |
| 6 | , | 343.8 | 66.7 | 12.93 | 0.0008 |
| 7 | before | 41.1 | 77.8 | 15.08 | 0.0010 |
| 8 | he | 98.4 | 88.9 | 17.24 | 0.0011 |
| 9 | got | 155.7 | 100.0 | 19.39 | 0.0013 |
| 10 | there | 213.0 | 111.1 | 21.55 | 0.0014 |

What to read from it:

- **The two "there"s are the same word with different rotations.** Same embedding, so the same `q` and `k` before
  RoPE; after RoPE the fast pair sits at 114.6° for position 2 and 213.0° for position 10. That difference is how a
  head can tell them apart.
- **Pair 0 has already wrapped by position 7** (7 radians = 401° → 41.1°). The table shows why one hand is not
  enough: position 7 is near position 1 on pair 0 (41° vs 57°), but far on pair 8 (78° vs 11°).
- **"got" sees distances, not positions.** Its query is 1 from "he", 2 from "before", 7 from the first "there",
  9 from "She". Attention scores from "got" depend only on those numbers and on the tokens' content.
- **Shift the sentence, nothing changes.** Put a 100-token system prompt in front: "She" moves to 100 and "got" to
  109, every angle in the table changes, but every offset is the same and the got → he score is identical to 10
  digits (1.1338736668 both times).
- **Generating it.** With "She was there all the time, before he" as the prompt, prefill gives those 9 tokens
  positions 0 … 8 and caches their rotated keys. The model generates "got" at position 9: its query is rotated by
  9 and compared with the 9 cached keys plus its own. Then "there" at position 10, against 11 keys. The two words
  at 0 and 8 were never re-rotated.
- **Causality.** "there" at position 2 never sees "he" or "got", because of the causal mask, not because of RoPE.
  RoPE says how far; the mask says what is allowed.

## 5. Going past the training length

Say a model was trained on `L` = 8K tokens with base 10 000 and you want to run it on 32K (scale `s = 4`). The fast
pairs are fine: they turned thousands of times in training and have seen every angle. The slow pairs are the
problem: in training they never completed a turn, so past 8K they reach angles the model has never seen. Without
any fix, 14 of the 64 pairs go out of range (check 7) and quality usually collapses past `L`.

| method | idea | fast pairs | slow pairs | out-of-range pairs at 4× (check 7) |
| --- | --- | --- | --- | ---: |
| none | keep counting | unchanged | out of range | 14 |
| position interpolation (Chen et al. 2023) | use `m / s` instead of `m`: squeeze 32K into the trained range | slowed 4×: neighbours get harder to tell apart | slowed 4× | 0 |
| "NTK-aware" base scaling (bloc97, 2023) | raise the base to `base · s^(d/(d−2))` | unchanged | the slowest slowed exactly 4×, middle ones less | 13 |
| YaRN (Peng et al. 2023) | per pair: interpolate pairs that turned < 1 time in training, leave pairs that turned > 32 times, blend between; plus a small logit temperature | unchanged | slowed 4× | 0 |
| larger base at pretraining (Llama 3: 500 000) | make the slow hands slow enough from the start | unchanged | never wrap inside the window | n/a |

All of them work much better with a short fine-tune at the new length. Llama 3.1 reached 128K with a YaRN-style
per-band rule: pairs with wavelength under 2K tokens unchanged, over 8K slowed 8×, a smooth blend in between.

"Out of range" here means the pair reaches an angle it never reached in training. NTK scaling leaves 13 pairs
slightly over their trained range at 4×, which is one reason it is usually used "dynamically" (scale grows only as
the context does) or with fine-tuning.

**Sliding windows.** Another option is to never let the model see more than `L` tokens: drop the oldest as new
ones arrive. Because scores depend only on offsets, the cached keys stay valid as the counter keeps going, and the
offsets the model sees never exceed `L`. Implementations often re-index the kept tokens from 0 anyway: very large
`m` loses precision when angles are computed in low-precision arithmetic, and StreamingLLM (Xiao et al. 2023)
keeps the first few "attention sink" tokens and assigns positions inside the cache.

## Corrections to the original chat

- The chat said a sliding window "re-indexes what's kept". It can, but it does not have to: with RoPE, keeping the
  counter going gives the same scores in exact arithmetic. Re-indexing is about numerical precision and attention
  sinks, not correctness.
- The chat's frequency formula and numbers are right. The one-angle-per-token picture from question 3 is the thing
  to drop: 64 angles per token per head, at fixed speeds independent of the window length.

## References

- Su et al. 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
- Chen et al. 2023, *Extending Context Window of Large Language Models via Positional Interpolation*.
- bloc97 2023, "NTK-Aware Scaled RoPE" (Reddit r/LocalLLaMA post; the idea behind dynamic NTK in Hugging Face).
- Peng et al. 2023, *YaRN: Efficient Context Window Extension of Large Language Models*.
- Xiao et al. 2023, *Efficient Streaming Language Models with Attention Sinks*.
- Llama Team 2024, *The Llama 3 Herd of Models* (base 500 000, document masks, 128K extension).
