"""Denoising diffusion models with classifier-free guidance, in JAX
(CPSC 440 lecture 15, CPSC 340 L36).

Forward process (fixed "encoder"): add a little Gaussian noise T times,
    q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I)
    =>  q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I),  abar_t = prod_{s<=t} (1 - beta_s).
With abar_T ~ 0, x_T is (almost) standard normal.

Reverse process (learned "decoder"): diffusion is a hierarchical VAE whose ELBO
reduces to matching the true denoising posterior
    q(x_{t-1} | x_t, x_0) = N(gamma_t x_t + delta_t x_0, sigma_t^2 I),
so we set p_theta(x_{t-1} | x_t) = N(gamma_t x_t + delta_t xhat_theta(x_t, t), sigma_t^2 I)
and, dropping the weights as the lecture does, train the denoiser with
    min E_{x_0, t, x_t ~ q(x_t | x_0)} || xhat_theta(x_t, t) - x_0 ||^2.
(Predicting x_0, the noise eps, or the score are equivalent reparameterizations.)

Sampling: x_T ~ N(0, I), then x_{t-1} ~ p_theta(x_{t-1} | x_t) for t = T..1.

Classifier-free guidance: train one network on (x, y) and, with probability 0.2,
on (x, null). At sampling time use
    (1 + w) xhat(x_t, t, y) - w xhat(x_t, t, null),
which sharpens samples towards class y (w = 0 is the plain conditional model).

Evaluation: the Fréchet distance between Gaussians fit to samples and to data
(FID without the Inception features, since the data are 2-D).

Run:  python 18-diffusion-and-image-generation/diffusion.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.linalg import sqrtm

T = 100
BETAS = np.linspace(1e-4, 0.2, T)
ALPHAS = 1 - BETAS
ABAR = np.cumprod(ALPHAS)
ABAR_PREV = np.concatenate([[1.0], ABAR[:-1]])
# q(x_{t-1} | x_t, x_0) = N(GAMMA x_t + DELTA x_0, SIG2)
GAMMA = np.sqrt(ALPHAS) * (1 - ABAR_PREV) / (1 - ABAR)
DELTA = np.sqrt(ABAR_PREV) * BETAS / (1 - ABAR)
SIG2 = BETAS * (1 - ABAR_PREV) / (1 - ABAR)


def time_features(t):
    f = jnp.arange(1, 9)
    tt = t[:, None] / T
    return jnp.concatenate([jnp.sin(np.pi * f * tt), jnp.cos(np.pi * f * tt)], -1)


def init(key, n_classes=2, h=128):
    ks = jax.random.split(key, 4)
    d_in = 2 + 16 + n_classes + 1          # x, time features, one-hot class incl. "null"
    he = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(2 / a)
    return [(he(ks[0], d_in, h), jnp.zeros(h)), (he(ks[1], h, h), jnp.zeros(h)),
            (he(ks[2], h, h), jnp.zeros(h)), (he(ks[3], h, 2) * 0.1, jnp.zeros(2))]


def denoise(params, x, t, y):
    """xhat_0(x_t, t, y); y = n_classes means unconditional."""
    n_classes = params[0][0].shape[0] - 2 - 16 - 1
    h = jnp.concatenate([x, time_features(t), jax.nn.one_hot(y, n_classes + 1)], -1)
    for W, b in params[:-1]:
        h = jax.nn.silu(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def train(X, y, steps=6000, batch=256, p_uncond=0.2, seed=0):
    params = init(jax.random.PRNGKey(seed), int(y.max()) + 1)
    sched = optax.cosine_decay_schedule(2e-3, steps)
    opt = optax.adam(sched)
    state = opt.init(params)
    sqrt_abar, sqrt_1m = jnp.asarray(np.sqrt(ABAR)), jnp.asarray(np.sqrt(1 - ABAR))
    null = int(y.max()) + 1

    def loss(p, x0, yb, key):
        k1, k2, k3 = jax.random.split(key, 3)
        t = jax.random.randint(k1, (len(x0),), 0, T)
        eps = jax.random.normal(k2, x0.shape)
        xt = sqrt_abar[t, None] * x0 + sqrt_1m[t, None] * eps
        yb = jnp.where(jax.random.uniform(k3, (len(x0),)) < p_uncond, null, yb)
        return jnp.mean(jnp.sum((denoise(p, xt, t, yb) - x0) ** 2, -1))

    @jax.jit
    def step(p, st, x0, yb, key):
        l, g = jax.value_and_grad(loss)(p, x0, yb, key)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        idx = rng.integers(0, len(X), batch)
        key, sub = jax.random.split(key)
        params, state, _ = step(params, state, jnp.asarray(X[idx]), jnp.asarray(y[idx]), sub)
    return params


def sample(params, n, y=None, w=0.0, seed=0):
    """Ancestral sampling through the learned reverse process; y=None is unconditional."""
    null = params[0][0].shape[0] - 2 - 16 - 1
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    x = jax.random.normal(sub, (n, 2))
    yc = jnp.full(n, null if y is None else y)
    yn = jnp.full(n, null)
    den = jax.jit(denoise)
    for t in range(T - 1, -1, -1):
        tt = jnp.full(n, t)
        x0 = den(params, x, tt, yc)
        if y is not None and w != 0:
            x0 = (1 + w) * x0 - w * den(params, x, tt, yn)
        mean = GAMMA[t] * x + DELTA[t] * x0
        if t > 0:
            key, sub = jax.random.split(key)
            x = mean + np.sqrt(SIG2[t]) * jax.random.normal(sub, x.shape)
        else:
            x = mean
    return np.asarray(x)


def frechet_distance(A, B):
    mu1, mu2 = A.mean(0), B.mean(0)
    S1, S2 = np.cov(A.T), np.cov(B.T)
    covmean = np.real(sqrtm(S1 @ S2))
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * covmean))


def demo():
    from sklearn.datasets import make_moons

    X, y = make_moons(4000, noise=0.05, random_state=0)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    Xref, yref = make_moons(4000, noise=0.05, random_state=1)
    Xref = (Xref - Xref.mean(0)) / Xref.std(0)
    print(f"Noise schedule: abar_T = {ABAR[-1]:.1e} (x_T is close to N(0, I))")
    params = train(X, y)

    def near_data(S):
        d = ((S[:, None] - Xref[None]) ** 2).sum(-1)
        return np.sqrt(d.min(1)), yref[d.argmin(1)]

    noise = np.random.default_rng(0).standard_normal((2000, 2))
    S = sample(params, 2000)
    dist, _ = near_data(S)
    print(f"Pure noise:             Frechet distance to data {frechet_distance(noise, Xref):.3f}, "
          f"median distance to nearest real point {np.median(near_data(noise)[0]):.3f}")
    print(f"Unconditional samples:  Frechet distance to data {frechet_distance(S, Xref):.3f}, "
          f"median distance to nearest real point {np.median(dist):.3f}")
    print(f"  (two fresh draws of real data: Frechet distance {frechet_distance(X[:2000], Xref):.3f}, "
          f"median distance {np.median(near_data(X[:2000])[0]):.3f})")
    print("Classifier-free guidance, asking for moon 0:")
    for w in [0.0, 1.0, 3.0]:
        S = sample(params, 2000, y=0, w=w, seed=1)
        dist, lab = near_data(S)
        print(f"  w={w}: fraction of samples on moon 0 {np.mean(lab == 0):.3f}, "
              f"median distance to data {np.median(dist):.3f}, spread (std) {S.std(0).round(3)}")


if __name__ == "__main__":
    demo()
