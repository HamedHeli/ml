"""Autoencoders, variational autoencoders and conditional VAEs
(CPSC 340 L34; CPSC 440 lectures 13-14).

Autoencoder: encoder f_phi: x -> z, decoder g_theta: z -> x, trained to minimize
reconstruction error. Linear f, g with squared error = PCA.

VAE: a deep latent-variable model
    z ~ N(0, I),  x | z ~ Bernoulli(sigmoid(g_theta(z)))   (pixels in {0, 1})
p_theta(x) = int p(z) p_theta(x | z) dz is intractable, so learn an *amortized*
inference network q_phi(z | x) = N(mu_phi(x), diag sigma_phi(x)^2) and maximize

    ELBO(x) = E_{q(z|x)}[log p_theta(x | z)] - KL(q_phi(z | x) || N(0, I))  <=  log p_theta(x)

with the reparameterization z = mu + sigma * eps. The KL term has a closed form:
    KL = 1/2 sum_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2).
beta-VAE multiplies the KL term by beta (beta > 1 for "disentangling").

Conditional VAE (in-painting): condition everything on observed pixels c:
    p(z), p_theta(x_missing | z, c), q_phi(z | x_missing, c).
Sampling z ~ N(0, I) and decoding with c gives *different plausible completions*.

We also report an importance-sampled estimate of log p(x) (IWAE bound with K
samples), which is tighter than the ELBO and shows the ELBO's gap.

Run:  python 12-variational-inference-and-vaes/vae.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_mlp(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (a, b)) * jnp.sqrt(2 / a), jnp.zeros(b)))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def bernoulli_loglik(logits, x):
    return -jnp.sum(jnp.logaddexp(0, -logits) * x + jnp.logaddexp(0, logits) * (1 - x), -1)


# ----------------------------------------------------------------------------- models

def ae_loss(params, x, key):
    z = mlp(params["enc"], x)[..., : params["k"].shape[0]]
    return -jnp.mean(bernoulli_loglik(mlp(params["dec"], z), x))


def vae_terms(params, x, key, cond=None):
    k = params["k"].shape[0]
    inp = x if cond is None else jnp.concatenate([x, cond], -1)
    h = mlp(params["enc"], inp)
    mu, log_s2 = h[..., :k], h[..., k:]
    eps = jax.random.normal(key, mu.shape)
    z = mu + jnp.exp(0.5 * log_s2) * eps
    dec_in = z if cond is None else jnp.concatenate([z, cond], -1)
    rec = bernoulli_loglik(mlp(params["dec"], dec_in), x)
    kl = 0.5 * jnp.sum(mu ** 2 + jnp.exp(log_s2) - 1 - log_s2, -1)
    return rec, kl


def vae_loss(params, x, key, cond=None, beta=1.0):
    rec, kl = vae_terms(params, x, key, cond)
    return -jnp.mean(rec - beta * kl)


def iwae_bound(params, x, key, K=200):
    """log (1/K) sum_k p(x, z_k) / q(z_k | x), z_k ~ q: a tighter lower bound on log p(x)."""
    k = params["k"].shape[0]
    h = mlp(params["enc"], x)
    mu, log_s2 = h[..., :k], h[..., k:]
    s = jnp.exp(0.5 * log_s2)
    eps = jax.random.normal(key, (K,) + mu.shape)
    z = mu + s * eps
    log_q = jnp.sum(-0.5 * eps ** 2 - jnp.log(s) - 0.5 * jnp.log(2 * jnp.pi), -1)
    log_pz = jnp.sum(-0.5 * z ** 2 - 0.5 * jnp.log(2 * jnp.pi), -1)
    log_px = bernoulli_loglik(mlp(params["dec"], z), x)
    lw = log_px + log_pz - log_q
    return jnp.mean(jax.scipy.special.logsumexp(lw, 0) - jnp.log(K))


def train(loss_fn, params, X, steps=4000, batch=128, lr=2e-3, seed=0, **kw):
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, xb, key, *extra):
        l, g = jax.value_and_grad(loss_fn)(params, xb, key, *extra, **kw)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, l

    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(X[0]), batch)
        key, sub = jax.random.split(key)
        params, state, _ = step(params, state, *[a[idx] for a in X[:1]], sub, *[a[idx] for a in X[1:]])
    return params


def make_params(key, d_in, k, hidden=128, d_cond=0, vae=True):
    k1, k2 = jax.random.split(key)
    return {"enc": init_mlp(k1, [d_in + d_cond, hidden, hidden, 2 * k if vae else k]),
            "dec": init_mlp(k2, [k + d_cond, hidden, hidden, d_in]),
            "k": jnp.zeros(k)}


def load_binary_digits():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    X = (X > 7).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(X))
    return X[perm[:1400]], X[perm[1400:]]


def demo(steps=4000):
    Xtr, Xte = load_binary_digits()
    Xtr_j, Xte_j = jnp.asarray(Xtr), jnp.asarray(Xte)
    key = jax.random.PRNGKey(0)
    k = 4

    ae = train(ae_loss, make_params(key, 64, k, vae=False), (Xtr_j,), steps)
    print(f"Autoencoder (k={k}): reconstruction NLL train {float(ae_loss(ae, Xtr_j, key)):.2f}, "
          f"test {float(ae_loss(ae, Xte_j, key)):.2f} nats/image (a deterministic code overfits)")

    vae = train(vae_loss, make_params(key, 64, k), (Xtr_j,), steps)
    rec, kl = vae_terms(vae, Xte_j, key)
    elbo = float(jnp.mean(rec - kl))
    iw = float(iwae_bound(vae, Xte_j, key))
    print(f"VAE (k={k}): test ELBO {elbo:.2f} = reconstruction {float(jnp.mean(rec)):.2f} - KL {float(jnp.mean(kl)):.2f}; "
          f"IWAE(200) bound on log p(x) {iw:.2f}")
    print("  (compare: product of Bernoullis -25.1, mixture of 30 Bernoullis -19.4 nats per image)")
    for beta in [4.0]:
        vb = train(vae_loss, make_params(key, 64, k), (Xtr_j,), steps, beta=beta)
        rec_b, kl_b = vae_terms(vb, Xte_j, key)
        print(f"beta-VAE (beta={beta}): reconstruction {float(jnp.mean(rec_b)):.2f}, KL {float(jnp.mean(kl_b)):.2f} "
              f"(posterior collapse: z carries almost no information)")

    # conditional VAE: observe the top half (32 pixels), in-paint the bottom half
    top, bot = slice(0, 32), slice(32, 64)
    cv = train(vae_loss, make_params(key, 32, k, d_cond=32), (Xtr_j[:, bot], Xtr_j[:, top]), steps)
    zs = jax.random.normal(jax.random.PRNGKey(1), (5, k))
    c = jnp.repeat(Xte_j[:1, top], 5, 0)
    samples = np.asarray(jax.nn.sigmoid(mlp(cv["dec"], jnp.concatenate([zs, c], -1))) > 0.5)
    print("Conditional VAE in-painting: given the top half of a test digit, five sampled bottom halves")
    rows = [np.asarray(Xte[0, top]).reshape(4, 8)]
    for r in range(4):
        print("   " + "".join("#" if v else "." for v in rows[0][r]))
    print("   " + "-" * 8 + "   (true bottom | samples)")
    truth = Xte[0, bot].reshape(4, 8)
    for r in range(4):
        print("   " + "".join("#" if v else "." for v in truth[r]) + "   "
              + "  ".join("".join("#" if v else "." for v in s.reshape(4, 8)[r]) for s in samples))


if __name__ == "__main__":
    demo()
