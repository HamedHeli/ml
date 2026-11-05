"""Directed acyclic graphical models (Bayesian networks) and d-separation
(CPSC 440 lecture 19).

A DAG model factorizes the joint as  p(x) = prod_j p(x_j | x_pa(j)).
Markov chains, naive Bayes, mixtures and HMMs are all special cases.

D-separation reads conditional independences off the graph: A is independent of
B given E if every undirected path between them is blocked, where a path is
blocked at a node v if
    * chain   a -> v -> b  or fork a <- v -> b,  and v is observed, or
    * collider a -> v <- b, and neither v nor any descendant of v is observed.
(The collider rule is "explaining away".) We implement it as reachability over
(node, direction) states, the "Bayes ball" algorithm.

Tabular DAG: each conditional p(x_j | x_pa(j)) is a table, learned by counting;
sampling is ancestral (parents before children, in topological order).

Run:  python 14-graphical-models/dag.py
"""

from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np


class DAG:
    def __init__(self, parents: dict[str, list[str]]):
        self.parents = parents
        self.nodes = list(parents)
        self.children = {v: [c for c in self.nodes if v in parents[c]] for v in self.nodes}

    def topological_order(self):
        order, seen = [], set()

        def visit(v):
            if v in seen:
                return
            for p in self.parents[v]:
                visit(p)
            seen.add(v)
            order.append(v)

        for v in self.nodes:
            visit(v)
        return order

    def d_separated(self, a, b, given=()):
        """True if a and b are d-separated given the observed set."""
        given = set(given)
        anc = set()                                   # observed nodes and their ancestors
        stack = list(given)
        while stack:
            v = stack.pop()
            if v not in anc:
                anc.add(v)
                stack.extend(self.parents[v])
        # state: (node, arrived_from_child?)  -- "up" means we came from a child
        q = deque([(a, True)])
        visited = set()
        while q:
            v, up = q.popleft()
            if (v, up) in visited:
                continue
            visited.add((v, up))
            if v == b:
                return False
            if up and v not in given:                 # chain/fork through an unobserved node
                q.extend((p, True) for p in self.parents[v])
                q.extend((c, False) for c in self.children[v])
            elif not up:
                if v not in given:                    # chain continues downward
                    q.extend((c, False) for c in self.children[v])
                if v in anc:                          # collider whose descendant is observed
                    q.extend((p, True) for p in self.parents[v])
        return True


class TabularDAG:
    """Binary variables, conditional probability tables learned by counting (with Laplace smoothing)."""

    def __init__(self, dag: DAG, alpha=1.0):
        self.dag, self.alpha = dag, alpha

    def fit(self, data: dict[str, np.ndarray]):
        self.cpt = {}
        for v in self.dag.nodes:
            pa = self.dag.parents[v]
            table = {}
            for vals in product([0, 1], repeat=len(pa)):
                m = np.ones(len(data[v]), bool)
                for p, val in zip(pa, vals):
                    m &= data[p] == val
                table[vals] = (data[v][m].sum() + self.alpha) / (m.sum() + 2 * self.alpha)
            self.cpt[v] = table
        return self

    def sample(self, n, rng):
        out = {}
        for v in self.dag.topological_order():
            pa = self.dag.parents[v]
            probs = np.array([self.cpt[v][tuple(int(out[p][i]) for p in pa)] for i in range(n)])
            out[v] = (rng.uniform(size=n) < probs).astype(int)
        return out


ALARM = DAG({"burglary": [], "earthquake": [], "alarm": ["burglary", "earthquake"],
             "john_calls": ["alarm"], "mary_calls": ["alarm"]})


def demo():
    g = ALARM
    print("D-separation in the alarm network (burglary -> alarm <- earthquake, alarm -> calls):")
    for a, b, e in [("burglary", "earthquake", []), ("burglary", "earthquake", ["alarm"]),
                    ("burglary", "earthquake", ["john_calls"]), ("john_calls", "mary_calls", []),
                    ("john_calls", "mary_calls", ["alarm"]), ("burglary", "john_calls", ["alarm"])]:
        print(f"  {a} _|_ {b} | {e or '{}'}: {g.d_separated(a, b, e)}")

    # verify "explaining away" numerically with a tabular model
    rng = np.random.default_rng(0)
    true = TabularDAG(g)
    true.cpt = {"burglary": {(): 0.1}, "earthquake": {(): 0.2},
                "alarm": {(0, 0): 0.01, (0, 1): 0.6, (1, 0): 0.9, (1, 1): 0.95},
                "john_calls": {(0,): 0.05, (1,): 0.9}, "mary_calls": {(0,): 0.01, (1,): 0.7}}
    S = true.sample(200000, rng)
    a1 = S["alarm"] == 1
    print("Explaining away, from 200k ancestral samples:")
    print(f"  p(burglary | alarm)              = {S['burglary'][a1].mean():.3f}")
    print(f"  p(burglary | alarm, earthquake)  = {S['burglary'][a1 & (S['earthquake'] == 1)].mean():.3f}")
    print(f"  corr(burglary, earthquake) = {np.corrcoef(S['burglary'], S['earthquake'])[0, 1]:+.3f}, "
          f"given alarm = {np.corrcoef(S['burglary'][a1], S['earthquake'][a1])[0, 1]:+.3f}")
    learned = TabularDAG(g).fit({k: v[:5000] for k, v in S.items()})
    print(f"  CPT p(alarm | b=1, e=0) learned from 5000 samples: {learned.cpt['alarm'][(1, 0)]:.3f} (true 0.9)")


if __name__ == "__main__":
    demo()
