"""Message passing in chains: Viterbi decoding, forward-backward, and hidden Markov
models trained by EM (CPSC 440 lectures 17 and 20).

HMM: hidden states z_t form a Markov chain (pi, A); each emits x_t ~ B[z_t].
    p(x, z) = pi[z_1] B[z_1, x_1] prod_t A[z_{t-1}, z_t] B[z_t, x_t]

Viterbi (decoding, argmax_z p(z | x)) is dynamic programming:
    M_1(j) = pi_j B_j(x_1),  M_t(j) = B_j(x_t) max_i M_{t-1}(i) A_ij,
    then backtrack the argmaxes. O(T k^2) instead of O(k^T).

Forward-backward (all marginals p(z_t | x)):
    alpha_t(j) = B_j(x_t) sum_i alpha_{t-1}(i) A_ij       ("forward messages")
    beta_t(i)  = sum_j A_ij B_j(x_{t+1}) beta_{t+1}(j)     ("backward messages")
    p(z_t = j | x) proportional to alpha_t(j) beta_t(j);  p(x) = sum_j alpha_T(j).
Messages are normalized at every step to avoid underflow.

Baum-Welch = EM for HMMs: the E-step is forward-backward (expected counts of
states and transitions); the M-step re-estimates pi, A, B from those counts.

Run:  python 13-markov-chains-and-hmms/hmm.py
"""

from __future__ import annotations

import itertools

import numpy as np


def forward_backward(x, pi, A, B):
    T, k = len(x), len(pi)
    alpha, beta, c = np.zeros((T, k)), np.ones((T, k)), np.zeros(T)
    a = pi * B[:, x[0]]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = B[:, x[t]] * (alpha[t - 1] @ A)
        c[t] = a.sum()
        alpha[t] = a / c[t]
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, x[t + 1]] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    xi = (alpha[:-1, :, None] * A[None] * (B[:, x[1:]].T * beta[1:])[:, None, :]) / c[1:, None, None]
    return gamma, xi, np.log(c).sum()                 # marginals, pair marginals, log p(x)


def viterbi(x, pi, A, B):
    T, k = len(x), len(pi)
    logA, logB = np.log(A), np.log(B)
    M = np.log(pi) + logB[:, x[0]]
    back = np.zeros((T, k), int)
    for t in range(1, T):
        S = M[:, None] + logA
        back[t] = S.argmax(0)
        M = S.max(0) + logB[:, x[t]]
    z = [int(M.argmax())]
    for t in range(T - 1, 0, -1):
        z.append(int(back[t, z[-1]]))
    return np.array(z[::-1]), M.max()


def baum_welch(xs, k, m, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    pi = np.full(k, 1 / k)
    A = rng.dirichlet(np.ones(k) * 5, k)
    B = rng.dirichlet(np.ones(m) * 5, k)
    hist = []
    for _ in range(iters):
        P0, NA, NB, ll = np.full(k, 1e-3), np.full((k, k), 1e-3), np.full((k, m), 1e-3), 0.0
        for x in xs:
            g, xi, l = forward_backward(x, pi, A, B)
            P0 += g[0]
            NA += xi.sum(0)
            for t, s in enumerate(x):
                NB[:, s] += g[t]
            ll += l
        hist.append(ll)
        pi, A, B = P0 / P0.sum(), NA / NA.sum(1, keepdims=True), NB / NB.sum(1, keepdims=True)
    return pi, A, B, hist


def sample_hmm(T, pi, A, B, rng):
    z = [rng.choice(len(pi), p=pi)]
    for _ in range(T - 1):
        z.append(rng.choice(len(pi), p=A[z[-1]]))
    z = np.array(z)
    x = np.array([rng.choice(B.shape[1], p=B[s]) for s in z])
    return z, x


def demo():
    rng = np.random.default_rng(0)
    # "occasionally dishonest casino": fair die vs. loaded die that rolls 6 half the time
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.1, 0.9]])
    B = np.vstack([np.full(6, 1 / 6), [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])

    z, x = sample_hmm(8, pi, A, B, rng)
    brute = {}
    for zz in itertools.product(range(2), repeat=8):
        zz = np.array(zz)
        brute[tuple(zz)] = (np.log(pi[zz[0]]) + np.log(A[zz[:-1], zz[1:]]).sum() + np.log(B[zz, x]).sum())
    z_bf = max(brute, key=brute.get)
    lse = np.logaddexp.reduce(list(brute.values()))
    z_v, _ = viterbi(x, pi, A, B)
    g, _, ll = forward_backward(x, pi, A, B)
    marg_bf = sum(np.exp(v - lse) * np.array(k) for k, v in brute.items())
    print("Checks against brute-force enumeration of all 2^8 state sequences:")
    print(f"  Viterbi path matches: {tuple(z_v) == z_bf}; log p(x) forward {ll:.4f} vs. brute force {lse:.4f}; "
          f"max |p(z_t = loaded | x) error| {np.abs(g[:, 1] - marg_bf).max():.1e}")

    z, x = sample_hmm(2000, pi, A, B, rng)
    z_v, _ = viterbi(x, pi, A, B)
    g, _, _ = forward_backward(x, pi, A, B)
    print(f"Casino, 2000 rolls: Viterbi state accuracy {np.mean(z_v == z):.3f}, "
          f"max-marginal accuracy {np.mean(g.argmax(1) == z):.3f}, always-fair baseline {np.mean(z == 0):.3f}")

    seqs = [sample_hmm(300, pi, A, B, rng)[1] for _ in range(10)]
    pi_h, A_h, B_h, hist = baum_welch(seqs, 2, 6, iters=200)
    order = np.argsort(B_h[:, 5])
    print(f"Baum-Welch on 10 x 300 rolls (log-likelihood {hist[0]:.0f} -> {hist[-1]:.0f}, monotone: {np.all(np.diff(hist) > -1e-6)})")
    print(f"  learned A  {np.round(A_h[np.ix_(order, order)], 3).tolist()}  (true {A.tolist()})")
    print(f"  learned p(6 | state) {np.round(B_h[order, 5], 3)}  (true [0.167, 0.5])")


if __name__ == "__main__":
    demo()
