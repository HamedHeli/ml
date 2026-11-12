"""EDM: "Elucidating the Design Space of Diffusion-Based Generative Models" (Karras, Aittala, Aila & Laine, NeurIPS
2022), in JAX on the same 2-D two-moons data as `diffusion.py` (DDPM; Ho, Jain & Abbeel, NeurIPS 2020).

EDM writes every diffusion model as one probability-flow ODE in the noise level sigma, then picks each design choice
separately:

    noisy data      x = y + sigma * n,   n ~ N(0, I)            (no scaling of y: "variance exploding", s(t) = 1)
    ODE             dx / dsigma = (x - D(x; sigma)) / sigma      (sigma(t) = t, so time *is* the noise level)
    score           grad_x log p(x; sigma) = (D(x; sigma) - x) / sigma^2

Preconditioning: the network F is wrapped so that its input and target both have unit variance at every sigma,
    D(x; sigma) = c_skip x + c_out F(c_in x; c_noise),
    c_skip = sd^2 / (sigma^2 + sd^2),  c_out = sigma sd / sqrt(sigma^2 + sd^2),
    c_in = 1 / sqrt(sigma^2 + sd^2),   c_noise = log(sigma) / 4,           sd = std of the data.

Training: ln sigma ~ N(P_mean, P_std^2) and loss weight lambda(sigma) = 1 / c_out^2, which is the same as an
unweighted MSE between F and the effective target (y - c_skip x) / c_out.

Sampling: N noise levels sigma_i = (smax^(1/rho) + i/(N-1) (smin^(1/rho) - smax^(1/rho)))^rho with rho = 7, which
spends most steps at low noise, followed by sigma_N = 0. Heun's 2nd-order method (an Euler step, then a correction with
the slope at the end point) costs 2N - 1 network evaluations (NFE). The paper also has a stochastic sampler ("churn")
that adds back a little noise at each step; it is left out here.

Run:  python 18-diffusion-and-image-generation/edm.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from diffusion import frechet_distance

SIGMA_DATA = 1.0              # the data are standardized (images in [-1, 1] use 0.5)
P_MEAN, P_STD = -1.2, 1.2     # training noise distribution, as in the paper
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0


def precond(sigma, sd=SIGMA_DATA):
    s2 = sigma ** 2 + sd ** 2
    return sd ** 2 / s2, sigma * sd / jnp.sqrt(s2), 1 / jnp.sqrt(s2), jnp.log(sigma) / 4


def noise_features(c_noise):
    f = jnp.arange(1, 9)
    return jnp.concatenate([jnp.sin(f * c_noise[:, None]), jnp.cos(f * c_noise[:, None])], -1)


def init(key, h=128):
    ks = jax.random.split(key, 4)
    he = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(2 / a)
    return [(he(ks[0], 2 + 16, h), jnp.zeros(h)), (he(ks[1], h, h), jnp.zeros(h)),
            (he(ks[2], h, h), jnp.zeros(h)), (he(ks[3], h, 2) * 0.1, jnp.zeros(2))]


def F(params, x, c_noise):
    h = jnp.concatenate([x, noise_features(c_noise)], -1)
    for W, b in params[:-1]:
        h = jax.nn.silu(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def D(params, x, sigma, precondition=True):
    """The denoiser D(x; sigma) for a batch; sigma has shape (n,)."""
    if not precondition:                                  # ablation: the network predicts y directly from raw x
        return F(params, x, jnp.log(sigma) / 4)
    c_skip, c_out, c_in, c_noise = precond(sigma)
    return c_skip[:, None] * x + c_out[:, None] * F(params, c_in[:, None] * x, c_noise)


def train(X, steps=6000, batch=256, precondition=True, seed=0):
    params = init(jax.random.PRNGKey(seed))
    opt = optax.adam(optax.cosine_decay_schedule(2e-3, steps))
    state = opt.init(params)

    def loss(p, y, key):
        k1, k2 = jax.random.split(key)
        sigma = jnp.exp(P_MEAN + P_STD * jax.random.normal(k1, (len(y),)))
        x = y + sigma[:, None] * jax.random.normal(k2, y.shape)
        _, c_out, _, _ = precond(sigma)
        lam = 1 / c_out ** 2                               # lambda(sigma) = (sigma^2 + sd^2) / (sigma sd)^2
        if not precondition:
            lam = jnp.ones_like(sigma)                     # plain x0-prediction MSE, as in diffusion.py
        return jnp.mean(lam * jnp.sum((D(p, x, sigma, precondition) - y) ** 2, -1))

    @jax.jit
    def step(p, st, y, key):
        l, g = jax.value_and_grad(loss)(p, y, key)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        params, state, _ = step(params, state, jnp.asarray(X[rng.integers(0, len(X), batch)]), sub)
    return params


def sigma_steps(N, rho=RHO, smin=SIGMA_MIN, smax=SIGMA_MAX):
    i = np.arange(N)
    s = (smax ** (1 / rho) + i / (N - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return np.append(s, 0.0)


def ode_sample(den, x, N, method="heun", rho=RHO):
    """Integrate dx/dsigma = (x - D(x; sigma)) / sigma from sigma_max to 0. Returns samples and network calls."""
    sig = sigma_steps(N, rho)
    nfe = 0
    for i in range(N):
        s, s_next = sig[i], sig[i + 1]
        d = (x - den(x, s)) / s
        nfe += 1
        x_next = x + (s_next - s) * d                      # Euler step
        if method == "heun" and s_next > 0:                # 2nd-order correction (skipped on the final step to 0)
            d_next = (x_next - den(x_next, s_next)) / s_next
            nfe += 1
            x_next = x + (s_next - s) * 0.5 * (d + d_next)
        x = x_next
    return x, nfe


def sample(params, n, N=18, method="heun", rho=RHO, precondition=True, seed=0):
    """Deterministic sampling with the learned denoiser."""
    den = jax.jit(lambda x, s: D(params, x, jnp.full(n, s), precondition))
    x0 = SIGMA_MAX * jax.random.normal(jax.random.PRNGKey(seed), (n, 2))
    x, nfe = ode_sample(den, x0, N, method, rho)
    return np.asarray(x), nfe


def ideal_denoiser(Y):
    """The exact minimizer of the denoising loss for the empirical distribution of the points Y:
    D(x; sigma) = sum_i softmax_i(-||x - y_i||^2 / (2 sigma^2)) y_i. Its ODE can only reproduce training points."""
    def den(x, s):
        logits = -((x[:, None, :] - Y[None]) ** 2).sum(-1) / (2 * s * s)
        w = np.exp(logits - logits.max(1, keepdims=True))
        return (w / w.sum(1, keepdims=True)) @ Y
    return den


def discretization_error(Y, nfes=(9, 15, 21, 35), n=300, seed=0):
    """Mean distance between few-step and 400-step (reference) ODE solutions, with the exact denoiser so that
    only the solver contributes error."""
    den = ideal_denoiser(Y)
    x0 = SIGMA_MAX * np.random.default_rng(seed).standard_normal((n, 2))
    ref, _ = ode_sample(den, x0, 400)
    out = {}
    for nfe in nfes:
        e = ode_sample(den, x0, nfe, "euler")[0]
        h = ode_sample(den, x0, (nfe + 1) // 2, "heun")[0]
        out[nfe] = (float(np.linalg.norm(e - ref, axis=1).mean()), float(np.linalg.norm(h - ref, axis=1).mean()))
    memorized = float(np.median(np.sqrt(((ref[:, None] - Y[None]) ** 2).sum(-1).min(1))))
    return out, memorized


def demo():
    from sklearn.datasets import make_moons

    X, _ = make_moons(4000, noise=0.05, random_state=0)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    Xref, _ = make_moons(4000, noise=0.05, random_state=1)
    Xref = (Xref - Xref.mean(0)) / Xref.std(0)

    def score(S):
        d = np.sqrt(((S[:, None] - Xref[None]) ** 2).sum(-1).min(1))
        return frechet_distance(S, Xref), float(np.median(d))

    print("noise levels for N = 8, rho = 7:", np.round(sigma_steps(8), 3))
    print("\n1. Exact denoiser of 500 training points: solver error vs. a 400-step solution (mean distance)")
    err, memorized = discretization_error(X[:500].astype(np.float64))
    for nfe, (e, h) in err.items():
        print(f"   NFE {nfe:3d}: Euler {e:.4f}   Heun {h:.4f}")
    print(f"   median distance of the exact-ODE samples to the nearest training point: {memorized:.1e}")

    params = train(X)
    print("\n2. Learned denoiser (Frechet distance, median distance to nearest real point)")
    for nfe in (9, 15, 21, 35):
        e, h = sample(params, 2000, nfe, "euler")[0], sample(params, 2000, (nfe + 1) // 2, "heun")[0]
        print(f"   NFE {nfe:3d}: Euler {score(e)[0]:.4f} / {score(e)[1]:.4f}   Heun {score(h)[0]:.4f} / {score(h)[1]:.4f}")
    print("\n3. Step spacing, Heun with N = 18 (35 NFE)")
    for rho in (1.0, 3.0, 7.0):
        fd, nn = score(sample(params, 2000, 18, "heun", rho=rho)[0])
        print(f"   rho = {rho:g}: Frechet {fd:.4f}, median NN distance {nn:.4f}")
    raw = train(X, precondition=False)
    fd, nn = score(sample(raw, 2000, 18, "heun", precondition=False)[0])
    print(f"\n4. Same network and sampler without preconditioning or loss weighting: Frechet {fd:.4f}, "
          f"median NN distance {nn:.4f}")


if __name__ == "__main__":
    demo()
