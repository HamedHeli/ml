"""Numerical checks for rope-positions.md: how RoPE turns a token's index into angles.

    frequency_table      per-pair speed, wavelength and turns over a 128K window, for base 10 000 and 500 000
    clock_uniqueness     one fast pair repeats every ~6 tokens; all 64 pairs together never repeat inside 128K
    relative_only        score(m, n) depends on m - n alone, even at positions near 128K
    kv_cache_decode      prefill + one-token-at-a-time decoding with cached rotated keys = full recompute
    score_vs_distance    with q = k, the score falls off with distance (RoPE's long-range decay)
    sentence_example     word positions in "She was there all the time, before he got there" and their angles
    context_extension    position interpolation, NTK base scaling and YaRN: which pairs see angles never seen
                         in training when a model trained on 8K is run on 32K

NumPy only, a few seconds on a CPU.
"""

from __future__ import annotations

import numpy as np

D_HEAD = 128


def freqs(d=D_HEAD, base=10000.0):
    """theta_i = base^(-2i/d), i = 0 .. d/2 - 1, in radians per token."""
    return base ** (-np.arange(0, d, 2) / d)


def rope(x, pos, theta):
    """Rotate consecutive pairs (x[2i], x[2i+1]) of the last axis by pos * theta_i.

    Llama's code splits the vector into halves instead of interleaving; that only permutes coordinates.
    """
    ang = np.asarray(pos, dtype=float)[:, None] * theta[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=float)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def frequency_table(d=D_HEAD, bases=(10000.0, 500000.0), window=131072, pairs=(0, 16, 32, 48, 63)):
    rows = []
    for b in bases:
        th = freqs(d, b)
        for i in pairs:
            rows.append({"base": b, "pair": i, "rad_per_token": float(th[i]),
                         "deg_per_token": float(np.degrees(th[i])),
                         "wavelength_tokens": float(2 * np.pi / th[i]),
                         "turns_in_window": float(window * th[i] / (2 * np.pi))})
    return rows


def clock_uniqueness(d=D_HEAD, base=500000.0, window=131072):
    """Distance between the rotation 'fingerprint' of position 0 and of every offset 1 .. window - 1.

    Because only offsets matter, position 0 vs offset k covers every pair of positions (m, m + k).
    Fingerprint = the unit vectors (cos, sin) of every pair's angle.
    """
    th = freqs(d, base)
    k = np.arange(1, window)
    ang = k[:, None] * th[None, :]
    # |e^{i a} - 1|^2 = 2 - 2 cos a, summed over pairs
    dist_all = np.sqrt((2 - 2 * np.cos(ang)).sum(1))
    dist_fast = np.sqrt(2 - 2 * np.cos(ang[:, 0]))
    near_fast = int(k[np.argmin(dist_fast[:100])])            # first near-repeat of the fastest hand
    return {"fast_pair_min_dist": float(dist_fast.min()),
            "fast_pair_near_repeat_offset": near_fast,
            "fast_pair_dist_at_that_offset": float(dist_fast[near_fast - 1]),
            "all_pairs_min_dist": float(dist_all.min()),
            "all_pairs_argmin_offset": int(k[np.argmin(dist_all)]),
            "max_possible_dist": float(np.sqrt(4 * d / 2))}


def relative_only(d=D_HEAD, base=500000.0, seed=0):
    """Same (q, k) pair placed at offsets 5 apart: near the start, mid-window, near 128K."""
    rng = np.random.default_rng(seed)
    q, k = rng.normal(size=(1, d)), rng.normal(size=(1, d))
    th = freqs(d, base)
    out = {}
    for m in (10, 60000, 131000):
        out[m] = float((rope(q, [m + 5], th) @ rope(k, [m], th).T)[0, 0])
    return {"scores": out, "spread": max(out.values()) - min(out.values())}


def attention(q, K, V):
    s = q @ K.T / np.sqrt(q.shape[-1])
    w = np.exp(s - s.max(-1, keepdims=True))
    return (w / w.sum(-1, keepdims=True)) @ V


def kv_cache_decode(T_prompt=12, T_new=8, d=D_HEAD, base=500000.0, seed=0):
    """Toy single-head causal attention. The 'next token' vectors are random: we only check the position bookkeeping.

    Full recompute: every step rebuilds q, k, v for all tokens and rotates them by 0 .. t.
    Cached: prefill rotates the prompt's keys once; each new token gets position N, N+1, ... and only its own
    q and k are rotated. Old cache entries are never touched.
    """
    rng = np.random.default_rng(seed)
    T = T_prompt + T_new
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    th = freqs(d, base)

    # full recompute, causal
    pos = np.arange(T)
    Q, K, V = rope(X @ Wq, pos, th), rope(X @ Wk, pos, th), X @ Wv
    full = np.stack([attention(Q[t:t + 1], K[:t + 1], V[:t + 1])[0] for t in range(T)])

    # prefill + decode with a KV cache
    p = np.arange(T_prompt)
    K_cache, V_cache = rope(X[:T_prompt] @ Wk, p, th), X[:T_prompt] @ Wv
    outs = [attention(rope(X[t:t + 1] @ Wq, [t], th), K_cache[:t + 1], V_cache[:t + 1])[0] for t in range(T_prompt)]
    positions_used = []
    for t in range(T_prompt, T):                      # one new token per step
        positions_used.append(t)
        k_new, v_new = rope(X[t:t + 1] @ Wk, [t], th), X[t:t + 1] @ Wv
        K_cache, V_cache = np.vstack([K_cache, k_new]), np.vstack([V_cache, v_new])
        outs.append(attention(rope(X[t:t + 1] @ Wq, [t], th), K_cache, V_cache)[0])
    return {"max_diff": float(np.abs(np.stack(outs) - full).max()), "decode_positions": positions_used}


def score_vs_distance(d=D_HEAD, base=500000.0, offsets=(0, 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536)):
    """q = k = all ones: score(offset) = 2 * sum_i cos(offset * theta_i). 'Same content' gets less credit far away."""
    th = freqs(d, base)
    return {int(o): float(2 * np.cos(o * th).sum()) for o in offsets}


SENTENCE = ["She", "was", "there", "all", "the", "time", ",", "before", "he", "got", "there"]


def sentence_example(words=SENTENCE, d=D_HEAD, base=500000.0, pairs=(0, 8, 16, 63), shift=100):
    """One token per word for readability (a real tokenizer splits differently). Angle of a few pairs per word,
    and the offsets that 'got' sees. Shifting the sentence (e.g. a 100-token system prompt in front) changes every
    angle but no offset."""
    th = freqs(d, base)
    rows = [{"pos": m, "word": w, "deg": {i: float(np.degrees(m * th[i]) % 360) for i in pairs},
             "deg_shifted": {i: float(np.degrees((m + shift) * th[i]) % 360) for i in pairs}}
            for m, w in enumerate(words)]
    g = words.index("got")
    offsets = {f"{w}@{m}": g - m for m, w in enumerate(words[:g + 1])}
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(1, d)), rng.normal(size=(1, d))
    sc = lambda off: float((rope(q, [g + off], th) @ rope(k, [8 + off], th).T)[0, 0])
    return {"rows": rows, "offsets_from_got": offsets, "got_he_score": sc(0), "got_he_score_shifted": sc(shift)}


# ---------------------------------------------------------------- context extension

def pi_freqs(theta, s):
    """Position interpolation (Chen et al. 2023): positions m -> m / s, i.e. every frequency / s."""
    return theta / s


def ntk_freqs(d, base, s):
    """'NTK-aware' scaling (bloc97, 2023): base -> base * s^(d / (d - 2)). The slowest pair slows by exactly s,
    the fastest pair is unchanged."""
    return freqs(d, base * s ** (d / (d - 2)))


def yarn_freqs(d, base, s, L_train, alpha=1.0, beta=32.0):
    """YaRN 'NTK-by-parts' (Peng et al. 2023). r = turns a pair makes in the training window.
    r < alpha: interpolate fully (theta / s). r > beta: leave alone. Between: blend linearly.
    (YaRN also scales the logits by (0.1 ln s + 1)^2; that does not change the angles.)"""
    th = freqs(d, base)
    r = L_train * th / (2 * np.pi)
    gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)
    return (1 - gamma) * th / s + gamma * th


def context_extension(d=D_HEAD, base=10000.0, L_train=8192, L_new=32768):
    """For each method, the largest angle each pair reaches over L_new tokens, compared with training.

    A pair is 'out of distribution' if it never completed a full turn in training (so it only ever saw angles
    up to L_train * theta) and now goes past that. Pairs that turned many times in training have seen every angle.
    """
    s = L_new / L_train
    th = freqs(d, base)
    seen = np.minimum(L_train * th, 2 * np.pi)                 # angle range seen in training, per pair
    methods = {"none": th, "PI": pi_freqs(th, s), "NTK": ntk_freqs(d, base, s), "YaRN": yarn_freqs(d, base, s, L_train)}
    out = {}
    for name, t in methods.items():
        reach = np.minimum(L_new * t, 2 * np.pi)
        unseen = reach > seen * (1 + 1e-9)
        out[name] = {"pairs_out_of_range": int(unseen.sum()),
                     "fast_pair_speed_ratio": float(t[0] / th[0]),        # 1 = local resolution kept
                     "slow_pair_speed_ratio": float(t[-1] / th[-1])}
    return out


def main():
    print("1. Per-pair speeds, d_head = 128, over a 131072-token window")
    print(f"   {'base':>8} {'pair':>4} {'deg/token':>11} {'tokens per turn':>16} {'turns in 128K':>14}")
    for r in frequency_table():
        print(f"   {r['base']:>8.0f} {r['pair']:>4d} {r['deg_per_token']:>11.5f} "
              f"{r['wavelength_tokens']:>16,.0f} {r['turns_in_window']:>14,.2f}")

    c = clock_uniqueness()
    print("\n2. Clock with 64 hands (base 500 000, offsets 1 .. 131071)")
    print(f"   fastest hand alone: near-repeat at offset {c['fast_pair_near_repeat_offset']}"
          f" (distance {c['fast_pair_dist_at_that_offset']:.3f}, 0 = identical, max 2)")
    print(f"   all 64 hands: closest offset {c['all_pairs_argmin_offset']}, distance {c['all_pairs_min_dist']:.3f}"
          f" (max possible {c['max_possible_dist']:.1f})")

    r = relative_only()
    print("\n3. Same q, k, 5 tokens apart, at three places in the window")
    for m, v in r["scores"].items():
        print(f"   key at {m:>6d}, query at {m + 5:>6d}: score {v:.10f}")
    print(f"   spread {r['spread']:.1e}")

    kv = kv_cache_decode()
    print("\n4. Prefill 12 tokens, then decode 8 one at a time with a KV cache")
    print(f"   positions given to new tokens: {kv['decode_positions']}")
    print(f"   max difference from full recompute: {kv['max_diff']:.1e}")

    print("\n5. Score for identical q = k = 1 at growing distance (base 500 000)")
    for o, v in score_vs_distance().items():
        print(f"   offset {o:>6d}: {v:7.1f}")

    se = sentence_example()
    print("\n6. 'She was there all the time, before he got there' (one token per word, base 500 000)")
    print(f"   {'pos':>3} {'word':>7} {'pair 0':>8} {'pair 8':>8} {'pair 16':>8} {'pair 63':>8}   (degrees, mod 360)")
    for r in se["rows"]:
        d = r["deg"]
        print(f"   {r['pos']:>3d} {r['word']:>7} {d[0]:>8.1f} {d[8]:>8.1f} {d[16]:>8.2f} {d[63]:>8.4f}")
    print(f"   offsets seen by 'got': {se['offsets_from_got']}")
    print(f"   got->he score {se['got_he_score']:.10f}; with 100 tokens in front {se['got_he_score_shifted']:.10f}")

    print("\n7. Trained on 8K (base 10 000), run on 32K (s = 4)")
    print(f"   {'method':>6} {'pairs out of range':>19} {'fast pair speed':>16} {'slow pair speed':>16}")
    for name, v in context_extension().items():
        print(f"   {name:>6} {v['pairs_out_of_range']:>19d} {v['fast_pair_speed_ratio']:>16.3f}"
              f" {v['slow_pair_speed_ratio']:>16.3f}")


if __name__ == "__main__":
    main()
