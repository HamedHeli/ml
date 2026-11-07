"""One token through one Transformer block, by hand.

Reproduces the numbers in `worked-example.md`: the sentence "I am the tree I saw yesterday", 4-dimensional
token representations, and the journey of "tree" through attention, residual + LayerNorm, and a 4 -> 16 -> 4 MLP.
The attention weights and Value vectors are illustrative (chosen by hand, not computed from learned Q and K).
"""
import numpy as np

TOKENS = ["I", "am", "the", "tree", "I", "saw", "yesterday"]
X_TREE = np.array([2.0, 3.0, 4.0, 5.0])
ATTN_TREE = np.array([0.5, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0])
VALUES = np.array([
    [2, 3, 4, 5],   # I (1)
    [1, 2, 1, 3],   # am
    [2, 1, 2, 1],   # the
    [3, 2, 1, 2],   # tree
    [1, 1, 1, 1],   # I (5)
    [2, 2, 2, 2],   # saw
    [1, 2, 3, 4],   # yesterday
], dtype=float)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V, causal=True):
    """Scaled dot-product attention softmax(QK^T / sqrt(d_k)) V for one head; rows are tokens."""
    scores = Q @ K.T / np.sqrt(K.shape[1])
    if causal:
        scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -np.inf)
    A = softmax(scores)
    return A @ V, A


def layer_norm(x, gamma=1.0, beta=0.0, eps=0.0):
    """Normalize each row (one token) across its features; eps=0 to match the hand calculation."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


def mlp(x, W1, b1, W2, b2):
    """Position-wise MLP: d -> d_ff -> d, applied to each row (token) with the same weights."""
    return gelu(x @ W1.T + b1) @ W2.T + b2


def demo():
    np.set_printoptions(precision=3, suppress=True)
    out = ATTN_TREE @ VALUES                       # step 5: weighted sum of Values
    res = X_TREE + out                             # step 6: residual
    mu, var = res.mean(), res.var()
    normed = layer_norm(res)
    print("attention output for 'tree':", out)
    print("after residual:            ", res)
    print(f"mean {mu:.2f}, variance {var:.2f}")
    print("after LayerNorm:           ", normed)

    rng = np.random.default_rng(0)                 # steps 7-9 with random (untrained) weights
    W1, b1 = rng.normal(size=(16, 4)) / 2, np.zeros(16)
    W2, b2 = rng.normal(size=(4, 16)) / 4, np.zeros(4)
    h = mlp(normed, W1, b1, W2, b2)
    print("MLP output shape:", h.shape, "-> block output:", layer_norm(normed + h))

    # The MLP is applied to each token independently: one row at a time gives the same answer as all rows at once.
    Xs = rng.normal(size=(7, 4))
    together = mlp(Xs, W1, b1, W2, b2)
    one_by_one = np.stack([mlp(x, W1, b1, W2, b2) for x in Xs])
    print("MLP per token == MLP on all tokens:", np.allclose(together, one_by_one))

    # A real head computes the weights from learned projections; with a causal mask, "tree" (position 4)
    # gets exactly zero weight on positions 5-7, like the illustrative weights above.
    WQ, WK, WV = (rng.normal(size=(4, 4)) for _ in range(3))
    _, A = attention(Xs @ WQ, Xs @ WK, Xs @ WV)
    print("causal attention weights of 'tree':", A[3])


if __name__ == "__main__":
    demo()
