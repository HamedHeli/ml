"""Markov chains (CPSC 440 lecture 16).

Chain rule + Markov assumption:
    p(x_1, ..., x_T) = p(x_1) prod_t p(x_t | x_{t-1})
Homogeneous chain: one transition matrix A[a, b] = p(x_t = b | x_{t-1} = a) for all t.

Learning (MLE) is counting:  A[a, b] = #(a -> b) / #(a -> anything),  pi = first-state frequencies.
Ancestral sampling: sample x_1 ~ pi, then x_t ~ A[x_{t-1}].
Chapman-Kolmogorov: exact marginals p(x_t) = p(x_{t-1}) A, O(T k^2).
Stationary distribution: p* = p* A, the limit of the marginals if the chain is
irreducible and aperiodic. PageRank is the stationary distribution of a random
surfer who follows a random link with probability d and jumps anywhere otherwise.

Run:  python 13-markov-chains-and-hmms/markov_chain.py
"""

from __future__ import annotations

import numpy as np

TEXT = (
    "the quick brown fox jumps over the lazy dog. a markov chain only remembers the last state, "
    "so the text it makes looks locally right and globally wrong. the chain rule of probability "
    "writes any joint distribution as a product of conditionals, and the markov assumption keeps "
    "only the most recent one. the stationary distribution is where the chain spends its time "
    "after it forgets where it started. pagerank is the stationary distribution of a random surfer."
)


class MarkovChain:
    def __init__(self, alpha=0.1):
        self.alpha = alpha            # Dirichlet smoothing on each row

    def fit(self, seqs, k):
        self.k = k
        C = np.full((k, k), self.alpha)
        p0 = np.full(k, self.alpha)
        for s in seqs:
            p0[s[0]] += 1
            np.add.at(C, (s[:-1], s[1:]), 1)
        self.A = C / C.sum(1, keepdims=True)
        self.pi = p0 / p0.sum()
        return self

    def log_likelihood(self, s):
        return np.log(self.pi[s[0]]) + np.log(self.A[s[:-1], s[1:]]).sum()

    def sample(self, T, rng, x0=None):
        x = [rng.choice(self.k, p=self.pi) if x0 is None else x0]
        for _ in range(T - 1):
            x.append(rng.choice(self.k, p=self.A[x[-1]]))
        return np.array(x)

    def marginals(self, T, p1=None):
        p = self.pi if p1 is None else p1
        out = [p]
        for _ in range(T - 1):
            p = p @ self.A
            out.append(p)
        return np.array(out)


def stationary(A, iters=10000, tol=1e-12):
    p = np.full(len(A), 1 / len(A))
    for _ in range(iters):
        q = p @ A
        if np.abs(q - p).max() < tol:
            return q
        p = q
    return p


def pagerank(adj, d=0.85):
    n = len(adj)
    out = adj.sum(1, keepdims=True)
    A = np.where(out > 0, adj / np.maximum(out, 1), 1 / n)       # dangling pages jump uniformly
    return stationary(d * A + (1 - d) / n)


def demo():
    chars = sorted(set(TEXT))
    idx = {c: i for i, c in enumerate(chars)}
    seq = np.array([idx[c] for c in TEXT])
    mc = MarkovChain(0.01).fit([seq], len(chars))
    rng = np.random.default_rng(3)
    print("Character-level homogeneous Markov chain, a sample:")
    print("  " + "".join(chars[i] for i in mc.sample(120, rng, x0=idx["t"])))

    # career chain from the lecture's toy example (states and numbers made up here)
    states = ["undergrad", "grad school", "industry", "academia"]
    A = np.array([[0.2, 0.3, 0.5, 0.0],
                  [0.0, 0.6, 0.3, 0.1],
                  [0.0, 0.02, 0.97, 0.01],
                  [0.0, 0.0, 0.1, 0.9]])
    mc2 = MarkovChain()
    mc2.k, mc2.A, mc2.pi = 4, A, np.array([1.0, 0, 0, 0])
    M = mc2.marginals(30)
    S = np.array([mc2.sample(30, rng) for _ in range(5000)])
    print("Chapman-Kolmogorov marginals at t=10 vs. Monte Carlo from 5000 ancestral samples:")
    for j, s in enumerate(states):
        print(f"  {s:>11}: exact {M[9, j]:.3f}, Monte Carlo {np.mean(S[:, 9] == j):.3f}")
    print(f"  stationary distribution {np.round(stationary(A), 3)}")

    adj = np.array([[0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0]], float)
    print(f"PageRank of a 5-page web (page 2 has the most in-links): {np.round(pagerank(adj), 3)}")


if __name__ == "__main__":
    demo()
