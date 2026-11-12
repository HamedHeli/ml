"""From prompt to picture: every stage of a latent text-to-image diffusion model, on a toy world you can run.

Real Stable Diffusion cannot be downloaded or run here, so this script builds a miniature with the same stages and
computes each one for real:

1. Tokenize the prompt with CLIP's byte-pair encoding (the real token ids, computed with Stable Diffusion 1.x's
   tokenizer vocabulary and hard-coded below).
2. Condition on the text. Our "text encoder" only recognises the class word (dog, cat, car, house); Stable Diffusion
   uses CLIP's 12-layer text Transformer, which turns the 77 tokens into a 77 x 768 matrix.
3. Encode images into a small latent space. A linear autoencoder (PCA) maps 64 x 64 x 3 pictures to a 4 x 8 x 8
   latent (Stable Diffusion: a VAE maps 512 x 512 x 3 to 4 x 64 x 64).
4. Start from Gaussian noise in that latent space and run the probability-flow ODE (EDM's Euler sampler, 30 steps),
   with classifier-free guidance.
5. Decode the final latent into a picture.

The denoiser is the *exact* optimal denoiser for our dataset, D(z; sigma | prompt) = a softmax-weighted average of the
training latents whose caption matches the prompt. A trained U-Net approximates this function; because the exact one
has seen only a finite dataset, its samples land on (blurry decodings of) training pictures, whereas a trained network
generalizes and makes new ones.

Run:  python 18-diffusion-and-image-generation/prompt_to_image.py      # about 20 s; writes figures/ptoi_*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

S = 64
FUR = {"brown": (139, 90, 43), "black": (40, 40, 40), "golden": (218, 165, 32), "white": (235, 235, 230),
       "gray": (128, 128, 128), "orange": (230, 120, 40)}
CAR = {"red": (200, 30, 30), "blue": (40, 80, 200), "yellow": (240, 200, 30), "green": (40, 150, 60)}


def canvas(rng):
    sky = tuple(int(c) for c in rng.uniform([150, 190, 220], [200, 225, 250]))
    im = Image.new("RGB", (S, S), sky)
    d = ImageDraw.Draw(im)
    g = int(rng.uniform(44, 50))
    d.rectangle([0, g, S, S], fill=(90, 160, 70))
    return im, d, g

def dog(rng):
    im, d, g = canvas(rng); name = rng.choice(list(FUR)); c = FUR[name]
    face = rng.choice([-1, 1]); cx = 32 + rng.integers(-4, 5); bw = rng.integers(13, 17); bh = rng.integers(7, 9)
    by = g - 9 - bh // 2
    d.ellipse([cx - bw, by - bh, cx + bw, by + bh], fill=c)                       # body
    for lx in [cx - bw + 3, cx - bw + 7, cx + bw - 7, cx + bw - 3]:               # legs
        d.rectangle([lx - 1, by + bh - 2, lx + 1, g], fill=c)
    hx, hy = cx + face * (bw + 2), by - bh - 2
    d.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], fill=c)                           # head
    d.ellipse([hx + face * 4 - 4, hy - 1, hx + face * 4 + 5 if face > 0 else hx + face * 4 + 4, hy + 5], fill=c)  # snout
    d.ellipse([hx + face * 9 - 1, hy, hx + face * 9 + 1, hy + 2], fill=(0, 0, 0))  # nose
    ear = tuple(max(0, v - 50) for v in c)
    d.ellipse([hx - face * 4 - 3, hy - 5, hx - face * 4 + 3, hy + 6], fill=ear)    # floppy ear
    d.ellipse([hx + face * 1 - 1, hy - 3, hx + face * 1 + 1, hy - 1], fill=(0, 0, 0))  # eye
    tx = cx - face * bw
    d.line([tx, by - 2, tx - face * 6, by - 2 - rng.integers(5, 10)], fill=c, width=3)  # tail up
    return im, f"a {name} dog standing on grass"

def cat(rng):
    im, d, g = canvas(rng); name = rng.choice(list(FUR)); c = FUR[name]
    face = rng.choice([-1, 1]); cx = 32 + rng.integers(-4, 5); bw = rng.integers(10, 13); bh = rng.integers(5, 7)
    by = g - 8 - bh // 2
    d.ellipse([cx - bw, by - bh, cx + bw, by + bh], fill=c)
    for lx in [cx - bw + 3, cx - bw + 6, cx + bw - 6, cx + bw - 3]:
        d.rectangle([lx - 1, by + bh - 2, lx, g], fill=c)
    hx, hy = cx + face * (bw + 1), by - bh - 3
    d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=c)
    d.polygon([(hx - 5, hy - 2), (hx - 4, hy - 10), (hx - 1, hy - 4)], fill=c)     # pointy ears
    d.polygon([(hx + 5, hy - 2), (hx + 4, hy - 10), (hx + 1, hy - 4)], fill=c)
    d.ellipse([hx + face * 2 - 1, hy - 2, hx + face * 2 + 1, hy], fill=(30, 160, 60))
    tx = cx - face * bw
    d.line([tx, by, tx - face * 3, by - 14], fill=c, width=2)
    return im, f"a {name} cat on grass"

def car(rng):
    im, d, g = canvas(rng); name = rng.choice(list(CAR)); c = CAR[name]
    cx = 32 + rng.integers(-4, 5); w = rng.integers(18, 22)
    d.rectangle([cx - w, g - 16, cx + w, g - 6], fill=c)
    d.polygon([(cx - w + 6, g - 16), (cx - w + 11, g - 25), (cx + w - 9, g - 25), (cx + w - 3, g - 16)], fill=c)
    d.rectangle([cx - w + 11, g - 23, cx + w - 10, g - 17], fill=(200, 230, 250))
    for wx in [cx - w + 8, cx + w - 8]:
        d.ellipse([wx - 5, g - 11, wx + 5, g - 1], fill=(20, 20, 20))
    return im, f"a {name} car parked on grass"

def house(rng):
    im, d, g = canvas(rng); name = rng.choice(list(CAR)); c = CAR[name]
    cx = 32 + rng.integers(-4, 5); w = rng.integers(13, 17)
    d.rectangle([cx - w, g - 2 * w + 2, cx + w, g], fill=(230, 220, 190))
    d.polygon([(cx - w - 3, g - 2 * w + 3), (cx, g - 3 * w), (cx + w + 3, g - 2 * w + 3)], fill=c)
    d.rectangle([cx - 3, g - 11, cx + 3, g], fill=(110, 70, 40))
    d.rectangle([cx + 5, g - 2 * w + 7, cx + 10, g - 2 * w + 12], fill=(150, 200, 240))
    return im, f"a house with a {name} roof"


DRAWERS = {"dog": dog, "cat": cat, "car": car, "house": house}

# Real CLIP BPE tokenization (Stable Diffusion 1.x tokenizer, vocabulary bpe_simple_vocab_16e6) of the prompt.
PROMPT = "give me the picture of a dog"
TOKENS = ["<|startoftext|>", "give</w>", "me</w>", "the</w>", "picture</w>", "of</w>", "a</w>", "dog</w>",
          "<|endoftext|>"]
TOKEN_IDS = [49406, 1781, 614, 518, 1674, 539, 320, 1929, 49407]
CONTEXT = 77                          # CLIP pads every prompt to 77 tokens (with the end-of-text id)

LATENT_SHAPE = (4, 8, 8)              # channels x height x width
K = int(np.prod(LATENT_SHAPE))
SIGMA_MAX, SIGMA_MIN, RHO = 80.0, 0.002, 7.0
SEED = 4                              # the seed shown step by step


def make_dataset(n_per_class=300, seed=1):
    rng = np.random.default_rng(seed)
    imgs, labels, captions = [], [], []
    for name, f in DRAWERS.items():
        for _ in range(n_per_class):
            im, cap = f(rng)
            imgs.append(np.asarray(im, np.float64) / 255)
            labels.append(name)
            captions.append(cap)
    return np.array(imgs), np.array(labels), captions


class LinearAutoencoder:
    """PCA as the 'VAE': encode to K numbers, decode back. Latents are scaled to unit variance, which is what
    Stable Diffusion's latent scaling factor (0.18215) does for its VAE latents."""

    def __init__(self, imgs):
        X = imgs.reshape(len(imgs), -1)
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = Vt[:K]
        self.scale = ((X - self.mu) @ self.V.T).std()

    def encode(self, imgs):
        return (imgs.reshape(len(imgs), -1) - self.mu) @ self.V.T / self.scale

    def decode(self, z):
        return np.clip((z * self.scale) @ self.V + self.mu, 0, 1).reshape(-1, S, S, 3)


def text_condition(prompt):
    """The toy text encoder: which of our four classes does the prompt name? (None = unconditional.)"""
    words = prompt.lower().replace(",", " ").split()
    hits = [c for c in DRAWERS if c in words]
    return hits[0] if hits else None


def make_denoiser(Z, labels):
    def denoise(z, sigma, cond):
        pool = Z if cond is None else Z[labels == cond]
        logits = -((z[:, None, :] - pool[None]) ** 2).sum(-1) / (2 * sigma ** 2)
        w = np.exp(logits - logits.max(1, keepdims=True))
        return (w / w.sum(1, keepdims=True)) @ pool
    return denoise


def sigma_steps(n):
    i = np.arange(n)
    return np.append((SIGMA_MAX ** (1 / RHO) + i / (n - 1) * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO, 0.0)


def generate(denoise, prompt, seed, guidance=3.0, n_steps=30):
    """Returns the final latent and a list of (sigma, noisy latent, predicted clean latent) for every step."""
    cond = text_condition(prompt)
    z = SIGMA_MAX * np.random.default_rng(seed).standard_normal((1, K))
    sig, history = sigma_steps(n_steps), []
    for i in range(n_steps):
        s = sig[i]
        d_cond = denoise(z, s, cond)
        d = d_cond if cond is None or guidance == 0 else denoise(z, s, None) + (1 + guidance) * (d_cond - denoise(z, s, None))
        history.append((s, z.copy(), d.copy()))
        z = z + (sig[i + 1] - s) * (z - d) / s           # Euler step of dz/dsigma = (z - D(z; sigma)) / sigma
    return z, history


# ------------------------------------------------------------------ figures
def show(ax, img, title=None, fs=8):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)


def latent_tile(z):
    """Lay the 4 x 8 x 8 latent out as its four 8 x 8 channels side by side."""
    c = z.reshape(LATENT_SHAPE)
    return np.concatenate([np.pad(ch, ((0, 0), (0, 1)), constant_values=np.nan) for ch in c], 1)[:, :-1]


def fig_tokens():
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.axis("off")
    shown = TOKENS + ["<|endoftext|>", "…", "<|endoftext|>"]
    ids = TOKEN_IDS + [49407, None, 49407]
    pos = list(range(len(TOKENS))) + [len(TOKENS), None, CONTEXT - 1]
    for i, (t, idx, p) in enumerate(zip(shown, ids, pos)):
        x = i * 1.0
        color = "#dbe9f6" if 0 < i < len(TOKENS) - 1 else "#eeeeee"
        ax.add_patch(plt.Rectangle((x, 0.35), 0.92, 0.5, color=color, ec="#555"))
        ax.text(x + 0.46, 0.6, t.replace("</w>", "").replace("<|", "<").replace("|>", ">"), ha="center", va="center", fontsize=7.5)
        if idx is not None:
            ax.text(x + 0.46, 0.18, str(idx), ha="center", fontsize=7.5, family="monospace")
            ax.text(x + 0.46, 0.98, f"pos {p}", ha="center", fontsize=6.5, color="#666")
    ax.text(0, -0.15, f'"{PROMPT}"  →  CLIP byte-pair encoding  →  {len(TOKENS)} tokens, padded to {CONTEXT}. '
            "Grey: start/end/padding tokens. Bottom: vocabulary ids (of 49,408).", fontsize=8)
    ax.set_xlim(-0.1, len(shown)); ax.set_ylim(-0.3, 1.1)
    fig.savefig(os.path.join(FIG, "ptoi_1_tokens.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_dataset_and_latent(imgs, labels, ae):
    idx = [np.where(labels == c)[0][j] for c in DRAWERS for j in range(3)]
    fig, axes = plt.subplots(3, 12, figsize=(12, 3.3))
    z = ae.encode(imgs[idx]); rec = ae.decode(z)
    for j, i in enumerate(idx):
        show(axes[0, j], imgs[i], labels[i])
        axes[1, j].imshow(latent_tile(z[j]), cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        show(axes[2, j], rec[j])
    axes[0, 0].set_ylabel("image\n64×64×3", fontsize=8)
    axes[1, 0].set_ylabel("latent\n4×8×8", fontsize=8)
    axes[2, 0].set_ylabel("decoded", fontsize=8)
    fig.suptitle("The training world (1200 captioned pictures) and its latent space: encoder → 256 numbers (4×8×8) → decoder",
                 fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "ptoi_2_latent_space.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_denoising(ae, history, title, fname, steps=(0, 6, 10, 13, 16, 19, 22, 25, 29)):
    fig, axes = plt.subplots(3, len(steps), figsize=(1.25 * len(steps), 4.1))
    for j, k in enumerate(steps):
        s, z, d = history[k]
        axes[0, j].imshow(latent_tile(z[0] / np.sqrt(1 + s ** 2)), cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                          interpolation="nearest")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[0, j].set_title(f"step {k + 1}\nσ = {s:.3g}", fontsize=7)
        show(axes[1, j], ae.decode(z / np.sqrt(1 + s ** 2))[0])
        show(axes[2, j], ae.decode(d)[0])
    axes[0, 0].set_ylabel("noisy latent", fontsize=7)
    axes[1, 0].set_ylabel("decoded\nnoisy latent", fontsize=7)
    axes[2, 0].set_ylabel("predicted\nclean picture", fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_seeds_and_prompts(ae, denoise):
    prompts = ["give me the picture of a dog", "a cat", "a car", "a house", ""]
    fig, axes = plt.subplots(len(prompts), 8, figsize=(8.5, 1.15 * len(prompts) + 0.3))
    for r, p in enumerate(prompts):
        for seed in range(8):
            z, _ = generate(denoise, p, seed)
            show(axes[r, seed], ae.decode(z)[0], f"seed {seed}" if r == 0 else None, fs=7)
        axes[r, 0].set_ylabel({"": "(empty prompt)"}.get(p, p if len(p) < 12 else "…a dog"), fontsize=8)
    fig.suptitle("Rows: prompt (sets *what*) · Columns: starting noise (sets *which one*)", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "ptoi_5_seeds_and_prompts.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def demo():
    os.makedirs(FIG, exist_ok=True)
    imgs, labels, captions = make_dataset()
    ae = LinearAutoencoder(imgs)
    Z = ae.encode(imgs)
    denoise = make_denoiser(Z, labels)
    print("prompt:", PROMPT)
    print("tokens:", [t.replace("</w>", "") for t in TOKENS], "ids:", TOKEN_IDS)
    print("toy text encoder ->", text_condition(PROMPT))
    print("latent std after scaling:", round(float(Z.std()), 3), "| variance kept by the 256-number latent:",
          round(float(1 - ((ae.decode(Z) - imgs) ** 2).sum() / ((imgs - imgs.mean(0)) ** 2).sum()), 3))
    fig_tokens()
    fig_dataset_and_latent(imgs, labels, ae)
    z, hist = generate(denoise, PROMPT, seed=SEED)
    fig_denoising(ae, hist, f'"{PROMPT}", seed {SEED}, guidance w = 3: 30 denoising steps', "ptoi_3_denoising_dog.png")
    _, hist_u = generate(denoise, "", seed=SEED)
    fig_denoising(ae, hist_u, f"Same starting noise (seed {SEED}), empty prompt: the model drifts to whatever it is "
                  "closest to", "ptoi_4_denoising_no_prompt.png")
    fig_seeds_and_prompts(ae, denoise)
    # how early is the content decided?  the class of the training picture closest to the predicted clean latent
    for name, h in [("dog prompt", hist), ("empty prompt", hist_u)]:
        nearest = [labels[np.argmin(((Z - d[0]) ** 2).sum(1))] for _, _, d in h]
        print(f"{name}: nearest training class of the prediction at steps 1, 5, 10, 15, 20, 30:",
              [nearest[k] for k in (0, 4, 9, 14, 19, 29)])
    counts = {}
    for seed in range(40):
        zf, _ = generate(denoise, PROMPT, seed)
        c = labels[np.argmin(((Z - zf[0]) ** 2).sum(1))]
        counts[c] = counts.get(c, 0) + 1
    print("40 seeds with the dog prompt end on:", counts)
    counts = {}
    for seed in range(40):
        zf, _ = generate(denoise, "", seed)
        c = labels[np.argmin(((Z - zf[0]) ** 2).sum(1))]
        counts[c] = counts.get(c, 0) + 1
    print("40 seeds with an empty prompt end on:", counts)


if __name__ == "__main__":
    demo()
