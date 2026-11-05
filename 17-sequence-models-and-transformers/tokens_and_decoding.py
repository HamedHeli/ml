"""Byte-pair encoding and beam search (CPSC 440 lecture 21).

Byte-pair encoding (BPE)
    Start from characters (or bytes). Repeatedly find the most frequent adjacent
    pair of tokens in the corpus and merge it into a new token. The learned merge
    list is the tokenizer: frequent words become single tokens, rare words split
    into pieces, and nothing is ever out-of-vocabulary.

Beam search
    A sequence model gives p(y_1..y_T) = prod_t p(y_t | y_<t). Greedy decoding
    takes the argmax at each step, which can miss the most probable sequence.
    Exact search is O(V^T). Beam search keeps the B best partial sequences at each
    step: B = 1 is greedy, B = V^T is exact.

Run:  python 17-sequence-models-and-transformers/tokens_and_decoding.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

CORPUS = (
    "low lower lowest newer newest wider widest slow slower slowest "
    "the newest model is lower and wider than the slowest model "
) * 20


def learn_bpe(text, n_merges):
    words = Counter(text.split())
    vocab = {tuple(w) + ("</w>",): c for w, c in words.items()}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for toks, c in vocab.items():
            for a, b in zip(toks[:-1], toks[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        new_vocab = {}
        for toks, c in vocab.items():
            out, i = [], 0
            while i < len(toks):
                if i < len(toks) - 1 and (toks[i], toks[i + 1]) == best:
                    out.append(toks[i] + toks[i + 1])
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            new_vocab[tuple(out)] = c
        vocab = new_vocab
    return merges


def bpe_encode(word, merges):
    toks = list(word) + ["</w>"]
    for a, b in merges:
        i, out = 0, []
        while i < len(toks):
            if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return toks


def beam_search(log_next, T, B, V):
    """log_next(prefix tuple) -> log-probs over V tokens. Returns (best sequence, log prob)."""
    beams = [((), 0.0)]
    for _ in range(T):
        cand = [(seq + (v,), lp + lv) for seq, lp in beams for v, lv in enumerate(log_next(seq))]
        beams = sorted(cand, key=lambda c: -c[1])[:B]
    return beams[0]


def demo():
    merges = learn_bpe(CORPUS, 15)
    print("First BPE merges:", ["".join(m) for m in merges[:10]])
    for w in ["lowest", "newer", "slowly", "unwiderable"]:
        print(f"  {w:>12} -> {bpe_encode(w, merges)}")

    # random first-order "language models": how often does each beam width find the most probable sequence?
    V, T, trials = 5, 6, 200
    hits = {1: 0, 2: 0, 5: 0}
    gap = {1: 0.0, 2: 0.0, 5: 0.0}
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        P0 = rng.dirichlet(np.ones(V) * 0.5)
        A = rng.dirichlet(np.ones(V) * 0.3, V)
        log_next = lambda seq: np.log(P0) if not seq else np.log(A[seq[-1]])
        _, exact_lp = beam_search(log_next, T, V ** T, V)      # beam as wide as the space = exact
        for B in hits:
            _, lp = beam_search(log_next, T, B, V)
            hits[B] += np.isclose(lp, exact_lp)
            gap[B] += (exact_lp - lp) / trials
    print(f"Decoding length-{T} sequences from {trials} random Markov models (vocabulary {V}):")
    for B in hits:
        print(f"  beam width {B}{' (greedy)' if B == 1 else '         '}: finds the most probable sequence "
              f"{hits[B] / trials:.0%} of the time, average log-prob shortfall {gap[B]:.3f}")


if __name__ == "__main__":
    demo()
