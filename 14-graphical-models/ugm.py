"""Undirected graphical models (Markov random fields) (CPSC 440 lecture 20).

    p(x) = (1/Z) prod_c phi_c(x_c),   Z = sum_x prod_c phi_c(x_c)

Independence is graph separation: x_A _|_ x_B | x_C if every path from A to B
passes through C. Z is a sum over exponentially many states, so inference and
learning need approximations unless the graph is a tree (or has small treewidth).

Ising model (binary pairwise UGM, x_i in {-1, +1}):
    log p(x) = sum_i h_i x_i + sum_{(i,j) in E} w_ij x_i x_j - log Z

Implemented:
* Gibbs sampling: p(x_i = +1 | rest) = sigmoid(2 (h_i + sum_j w_ij x_j)).
* ICM (iterated conditional modes): set each x_i to its most likely value given its
  neighbours; a coordinate-wise decoder that finds a local optimum. Used here to
  denoise a binary image with a grid Ising prior.
* Pseudo-likelihood learning: maximize sum_i log p(x_i | x_neighbours), which
  avoids Z entirely and is consistent.
* Graphical lasso: for Gaussian UGMs, the graph is the sparsity pattern of the
  precision matrix Theta. Maximize log det Theta - tr(S Theta) - lambda ||Theta||_1
  (solved here with ADMM).

Run:  python 14-graphical-models/ugm.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def grid_neighbours(H, W):
    idx = np.arange(H * W).reshape(H, W)
    edges = [(idx[i, j], idx[i, j + 1]) for i in range(H) for j in range(W - 1)]
    edges += [(idx[i, j], idx[i + 1, j]) for i in range(H - 1) for j in range(W)]
    return edges


def adjacency(n, edges, w):
    Wm = np.zeros((n, n))
    for a, b in edges:
        Wm[a, b] = Wm[b, a] = w
    return Wm


def gibbs_ising(h, Wm, sweeps, rng, x0=None):
    n = len(h)
    x = rng.choice([-1, 1], n) if x0 is None else x0.copy()
    for _ in range(sweeps):
        for i in range(n):
            x[i] = 1 if rng.uniform() < expit(2 * (h[i] + Wm[i] @ x)) else -1
    return x


def icm(h, Wm, x0, max_sweeps=50):
    x = x0.copy()
    for _ in range(max_sweeps):
        changed = False
        for i in range(len(x)):
            new = 1 if h[i] + Wm[i] @ x > 0 else -1
            if new != x[i]:
                x[i], changed = new, True
        if not changed:
            break
    return x


def pseudo_likelihood_fit(X, edges, iters=500, lr=0.05):
    """Fit a shared field h and a shared coupling w by gradient ascent on the pseudo-likelihood."""
    n = X.shape[1]
    A = adjacency(n, edges, 1.0)
    h, w = 0.0, 0.0
    for _ in range(iters):
        field = h + w * X @ A                      # local field at every node, (samples, n)
        r = X - np.tanh(field)                     # d/dfield of log p(x_i | rest) = x_i - tanh(field)
        h += lr * r.mean()
        w += lr * (r * (X @ A)).mean()
    return h, w


def graphical_lasso(S, lam, rho=1.0, iters=500):
    """ADMM for max log det Theta - tr(S Theta) - lam sum_{i != j} |Theta_ij|."""
    d = len(S)
    Z = np.eye(d)
    U = np.zeros((d, d))
    off = ~np.eye(d, dtype=bool)
    for _ in range(iters):
        vals, vecs = np.linalg.eigh(rho * (Z - U) - S)
        theta_vals = (vals + np.sqrt(vals ** 2 + 4 * rho)) / (2 * rho)
        Theta = (vecs * theta_vals) @ vecs.T
        A = Theta + U
        Z = np.where(off, np.sign(A) * np.maximum(np.abs(A) - lam / rho, 0), A)
        U += Theta - Z
    return Z


def demo():
    rng = np.random.default_rng(0)
    H = W = 24
    yy, xx = np.mgrid[:H, :W]
    clean = np.where((yy - 12) ** 2 + (xx - 12) ** 2 < 64, 1, -1).ravel()
    flip = rng.uniform(size=H * W) < 0.2
    noisy = np.where(flip, -clean, clean)
    edges = grid_neighbours(H, W)
    beta, coupling = 0.5 * np.log(0.8 / 0.2), 0.8         # evidence strength from the 20% flip rate
    x_icm = icm(beta * noisy, adjacency(H * W, edges, coupling), noisy)
    xs = noisy.copy()
    marg = np.zeros(H * W)
    Wm = adjacency(H * W, edges, coupling)
    for t in range(60):
        xs = gibbs_ising(beta * noisy, Wm, 1, rng, xs)
        if t >= 10:
            marg += xs
    x_gibbs = np.where(marg > 0, 1, -1)
    print(f"Ising denoising of a 24x24 disc with 20% flipped pixels: pixel error noisy {np.mean(noisy != clean):.3f}, "
          f"ICM {np.mean(x_icm != clean):.3f}, Gibbs max-marginals {np.mean(x_gibbs != clean):.3f}")

    # pseudo-likelihood recovers Ising parameters from Gibbs samples
    Hs = Ws = 6
    e = grid_neighbours(Hs, Ws)
    h_true, w_true = 0.1, 0.3
    Wm = adjacency(Hs * Ws, e, w_true)
    x = rng.choice([-1, 1], Hs * Ws)
    data = []
    for t in range(3000):
        x = gibbs_ising(np.full(Hs * Ws, h_true), Wm, 1, rng, x)
        if t >= 200 and t % 2 == 0:
            data.append(x.copy())
    h_hat, w_hat = pseudo_likelihood_fit(np.array(data, float), e)
    print(f"Pseudo-likelihood on Gibbs samples of a 6x6 Ising model: h {h_hat:.3f} (true {h_true}), w {w_hat:.3f} (true {w_true})")

    # graphical lasso recovers a chain-structured precision matrix
    d = 8
    Theta = np.eye(d) * 2
    for i in range(d - 1):
        Theta[i, i + 1] = Theta[i + 1, i] = -0.8
    X = rng.multivariate_normal(np.zeros(d), np.linalg.inv(Theta), 2000)
    S = np.cov(X.T)
    Z = graphical_lasso(S, 0.05)
    est_edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(Z[i, j]) > 1e-3}
    true_edges = {(i, i + 1) for i in range(d - 1)}
    inv_edges = {(i, j) for i in range(d) for j in range(i + 1, d) if abs(np.linalg.inv(S)[i, j]) > 1e-3}
    print(f"Graphical lasso on a chain (7 true edges): recovered {len(est_edges & true_edges)}/7, "
          f"false edges {len(est_edges - true_edges)} (plain inverse covariance has {len(inv_edges)} nonzeros)")


if __name__ == "__main__":
    demo()
