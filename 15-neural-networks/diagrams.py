"""Block diagrams of the architectures in building-blocks.md, drawn with matplotlib.

Run:  python 15-neural-networks/diagrams.py      # writes figures/arch_*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# One colour per kind of block, used in every figure.
COL = dict(norm="#a9c8ec", attn="#fde3c8", mlp="#fff2c2", emb="#e6b4ab", pos="#a5cdd1", linear="#d9a9c4",
           softmax="#b7d8a6", conv="#c9e4c5", act="#eeeeee", rnn="#d7cdee", gate="#f6c1c1", io="#ffffff",
           ssm="#cfe0f7", router="#f4d1a6", pool="#dde6ef")
EDGE = "#555555"
FRAME = "#c62828"


class Diagram:
    def __init__(self, ax):
        self.ax = ax
        self.nodes = {}

    # ---- primitives
    def box(self, name, x, y, label, kind, w=2.2, h=0.5, fs=9):
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                         fc=COL[kind], ec=EDGE, lw=1))
        self.ax.text(x, y, label, ha="center", va="center", fontsize=fs)
        self.nodes[name] = (x, y, w / 2, h / 2)

    def add(self, name, x, y, r=0.17, symbol="+"):
        self.ax.add_patch(Circle((x, y), r, fc="white", ec=EDGE, lw=1))
        self.ax.text(x, y - 0.01, symbol, ha="center", va="center", fontsize=11)
        self.nodes[name] = (x, y, r, r)

    def text(self, x, y, s, fs=9, **kw):
        self.ax.text(x, y, s, fontsize=fs, **kw)

    def _arrow(self, pts, head=True):
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([x0, x1], [y0, y1], color=EDGE, lw=1.2, solid_capstyle="round")
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if head else "-", mutation_scale=10,
                                          color=EDGE, lw=1.2, shrinkA=0, shrinkB=0))

    def side(self, name, where):
        x, y, hw, hh = self.nodes[name]
        return {"top": (x, y + hh), "bottom": (x, y - hh), "left": (x - hw, y), "right": (x + hw, y)}[where]

    def up(self, a, b):
        """Straight arrow from the top of a to the bottom of b."""
        self._arrow([self.side(a, "top"), self.side(b, "bottom")])

    def link(self, a, b, a_side="top", b_side="bottom", via=None):
        pts = [self.side(a, a_side)] + (via or []) + [self.side(b, b_side)]
        self._arrow(pts)

    def residual(self, branch_from, to, dx=1.55, side="right", below=False):
        """A skip connection: leaves the main line just above `branch_from` (or just below it, with below=True)
        and enters `to` from the side."""
        x, y, hw, hh = self.nodes[branch_from]
        yb = y - hh - 0.15 if below else y + hh + 0.12
        xs = x + (dx if side == "right" else -dx)
        tx, ty, thw, _ = self.nodes[to]
        self.ax.plot([x], [yb], "o", color=EDGE, ms=3)
        self._arrow([(x, yb), (xs, yb), (xs, ty), (tx + (thw if side == "right" else -thw), ty)])

    def frame(self, x0, y0, x1, y1, label=None, label_side="left"):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                         fc="none", ec=FRAME, lw=1.6))
        if label:
            lx = x0 - 0.1 if label_side == "left" else x1 + 0.1
            self.ax.text(lx, (y0 + y1) / 2, label, ha="right" if label_side == "left" else "left", va="center",
                         fontsize=10, fontweight="bold")

    def title(self, x, y, s, fs=11):
        self.ax.text(x, y, s, ha="center", va="bottom", fontsize=fs, fontweight="bold")


def new(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    return fig, Diagram(ax)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prenorm_block(d, x, y0, mixer, mixer_kind, channel, channel_kind, norm="Norm", prefix="", h_mixer=0.5, w=2.2,
                  fs=9):
    """Norm → token mixer → (+) → Norm → channel mixer → (+), with both residuals. Returns the top y."""
    p = prefix
    d.box(p + "n1", x, y0, norm, "norm", w=w, fs=fs)
    d.box(p + "m1", x, y0 + 0.8 + (h_mixer - 0.5) / 2, mixer, mixer_kind, w=w, h=h_mixer, fs=fs)
    y = y0 + 0.8 + h_mixer + 0.25
    d.add(p + "a1", x, y)
    d.box(p + "n2", x, y + 0.7, norm, "norm", w=w, fs=fs)
    d.box(p + "m2", x, y + 1.5, channel, channel_kind, w=w, h=0.62 if "\n" in channel else 0.5, fs=fs)
    d.add(p + "a2", x, y + 2.25)
    for a, b in [("n1", "m1"), ("m1", "a1"), ("a1", "n2"), ("n2", "m2"), ("m2", "a2")]:
        d.up(p + a, p + b)
    d.residual(p + "n1", p + "a1", dx=w / 2 + 0.35, below=True)    # branches *before* the norm
    d.residual(p + "a1", p + "a2", dx=w / 2 + 0.35)
    return y + 2.25


# ------------------------------------------------------------------ 1. the shared skeleton and its variants
def fig_skeleton():
    fig, d = new(15, 5.9, (-1.2, 23.5), (-1.3, 6.6))
    variants = [
        ("The skeleton", "token mixer\n(between positions)", "attn", "channel mixer\n(each position)", "mlp", "Norm"),
        ("Transformer\n(GPT, BERT, ViT)", "Multi-head\nself-attention", "attn", "MLP / SwiGLU", "mlp", "LayerNorm (or RMS)"),
        ("ConvNeXt", "7×7 depthwise conv", "conv", "MLP (1×1 convs,\n4× expansion)", "mlp", "LayerNorm"),
        ("MLP-Mixer", "MLP across\npositions", "mlp", "MLP across\nchannels", "mlp", "LayerNorm"),
        ("Mamba", "selective\nstate-space scan", "ssm", "(gating inside\nthe same block)", "gate", "RMSNorm"),
        ("Mixture of experts", "Multi-head\nself-attention", "attn", "router + expert\nMLPs", "router", "RMSNorm"),
    ]
    for i, (name, mix, mk, ch, ck, norm) in enumerate(variants):
        x = 1.6 + i * 3.9
        d.title(x, 5.65, name, fs=10)
        d.box(f"in{i}", x, -0.6, "x (residual stream)", "io", w=2.6, fs=8)
        top = prenorm_block(d, x, 0.4, mix, mk, ch, ck, norm=norm, prefix=f"v{i}", h_mixer=0.7, w=2.6, fs=8)
        d.up(f"in{i}", f"v{i}n1")
        d.box(f"out{i}", x, top + 0.75, "x (to next block)", "io", w=2.6, fs=8)
        d.up(f"v{i}a2", f"out{i}")
        if i == 0:
            d.frame(x - 1.75, -0.2, x + 1.95, top + 0.3)
    save(fig, "arch_1_skeleton.png")


# ------------------------------------------------------------------ 2. Transformers
def fig_transformers():
    fig, d = new(15, 8.6, (-3.5, 20.5), (-1.7, 11.9))
    # original encoder–decoder (post-norm)
    ex, dx = 1.2, 6.2
    d.title((ex + dx) / 2, 10.9, "Original Transformer (Vaswani et al. 2017): post-norm, encoder–decoder")
    d.text(ex, -1.55, "input sequence", ha="center")
    d.box("e_emb", ex, -0.8, "Embeddings", "emb")
    d.add("e_pos+", ex, 0.2)
    d.box("e_pos", ex - 2.4, 0.2, "Positional\nencoding", "pos", w=1.6, h=0.6, fs=8)
    d.link("e_pos", "e_pos+", "right", "left")
    d.up("e_emb", "e_pos+")
    d.box("e_att", ex, 1.4, "Multi-head\nself-attention", "attn", h=0.7)
    d.box("e_an1", ex, 2.4, "Add & Norm", "norm")
    d.box("e_mlp", ex, 3.4, "MLP", "mlp")
    d.box("e_an2", ex, 4.4, "Add & Norm", "norm")
    for a, b in [("e_pos+", "e_att"), ("e_att", "e_an1"), ("e_an1", "e_mlp"), ("e_mlp", "e_an2")]:
        d.up(a, b)
    d.residual("e_pos+", "e_an1", dx=1.45, side="left")
    d.residual("e_an1", "e_an2", dx=1.45, side="left")
    d.frame(ex - 1.75, 0.85, ex + 1.35, 4.85, "× N", "left")
    d.text(dx, -1.55, "target sequence (shifted right)", ha="center")
    d.box("d_emb", dx, -0.8, "Embeddings", "emb")
    d.add("d_pos+", dx, 0.2)
    d.box("d_pos", dx + 2.4, 0.2, "Positional\nencoding", "pos", w=1.6, h=0.6, fs=8)
    d.link("d_pos", "d_pos+", "left", "right")
    d.up("d_emb", "d_pos+")
    d.box("d_att", dx, 1.4, "Masked multi-head\nself-attention", "attn", w=2.6, h=0.7)
    d.box("d_an1", dx, 2.4, "Add & Norm", "norm")
    d.box("d_x", dx, 3.5, "Multi-head\ncross-attention", "attn", h=0.7)
    d.box("d_an2", dx, 4.5, "Add & Norm", "norm")
    d.box("d_mlp", dx, 5.5, "MLP", "mlp")
    d.box("d_an3", dx, 6.5, "Add & Norm", "norm")
    d.box("lin", dx, 7.6, "Linear", "linear", h=0.4)
    d.box("sm", dx, 8.3, "Softmax", "softmax", h=0.4)
    d.text(dx, 8.75, "next-token probabilities", ha="center")
    for a, b in [("d_pos+", "d_att"), ("d_att", "d_an1"), ("d_an1", "d_x"), ("d_x", "d_an2"), ("d_an2", "d_mlp"),
                 ("d_mlp", "d_an3"), ("d_an3", "lin"), ("lin", "sm")]:
        d.up(a, b)
    d.residual("d_pos+", "d_an1", dx=1.45)
    d.residual("d_an1", "d_an2", dx=1.45)
    d.residual("d_an2", "d_an3", dx=1.45)
    d.frame(dx - 1.35, 0.85, dx + 1.75, 6.9, "× N", "right")
    d._arrow([d.side("e_an2", "top"), (ex, 5.3), (3.7, 5.3), (3.7, 3.5), d.side("d_x", "left")])
    d.text(2.45, 5.45, "keys & values", fs=8, ha="center")
    d.text((ex + dx) / 2, 10.45, "Each sub-layer:  x ← Norm(x + Sublayer(x))", fs=9, ha="center", style="italic")

    # GPT / LLaMA decoder-only (pre-norm)
    gx = 14.5
    d.title(gx, 10.9, "GPT / LLaMA-style decoder-only: pre-norm")
    d.text(gx, -1.55, "tokens so far", ha="center")
    d.box("g_emb", gx, -0.8, "Token embeddings", "emb", w=2.8)
    top = prenorm_block(d, gx, 0.6, "Masked multi-head\nself-attention\n(RoPE on Q, K)", "attn", "SwiGLU MLP", "mlp",
                        norm="RMSNorm", prefix="g", h_mixer=0.95, w=2.6)
    d.up("g_emb", "gn1")
    d.frame(gx - 1.55, 0.2, gx + 2.0, top + 0.35, "× N", "left")
    d.box("g_fn", gx, top + 0.95, "RMSNorm", "norm")
    d.box("g_lin", gx, top + 1.75, "Linear (tied to\nembeddings)", "linear", h=0.6)
    d.box("g_sm", gx, top + 2.6, "Softmax", "softmax", h=0.4)
    d.text(gx, top + 3.05, "next-token probabilities", ha="center")
    for a, b in [("ga2", "g_fn"), ("g_fn", "g_lin"), ("g_lin", "g_sm")]:
        d.up(a, b)
    d.text(gx + 0.25, 10.45, "Each sub-layer:  x ← x + Sublayer(Norm(x))", fs=9, ha="center", style="italic")
    d.text(gx + 0.25, 9.75, "no positional encoding added; RoPE rotates\nqueries and keys inside attention", fs=8,
           ha="center", color="#444")
    save(fig, "arch_2_transformers.png")


# ------------------------------------------------------------------ 3. multi-head attention
def fig_attention():
    fig, d = new(13, 6.3, (-1, 17.5), (-1.2, 7.4))
    # left: one head
    x = 3.2
    d.title(x, 6.85, "Scaled dot-product attention (one head)")
    for name, lx, lbl in [("q", x - 2.2, "Q"), ("k", x, "K"), ("v", x + 2.2, "V")]:
        d.box(name, lx, -0.5, lbl, "io", w=1.0, h=0.45)
    d.box("mm1", x - 1.1, 1.0, "MatMul  QKᵀ", "linear", w=2.2, h=0.45)
    d.box("scale", x - 1.1, 1.9, "Scale  ÷ √d_k", "act", w=2.2, h=0.45)
    d.box("mask", x - 1.1, 2.8, "Mask (decoder only)", "act", w=2.2, h=0.45, fs=8)
    d.box("soft", x - 1.1, 3.7, "Softmax (each row)", "softmax", w=2.2, h=0.45)
    d.box("mm2", x, 5.0, "MatMul  (weights · V)", "linear", w=2.6, h=0.45)
    d.box("out", x, 6.3, "output: one new vector per token", "io", w=3.6, h=0.45, fs=8)
    d.link("q", "mm1", via=[(x - 2.2, 0.5), (x - 1.4, 0.5)])
    d.link("k", "mm1", via=[(x, 0.5), (x - 0.8, 0.5)])
    for a, b in [("mm1", "scale"), ("scale", "mask"), ("mask", "soft")]:
        d.up(a, b)
    d.link("soft", "mm2", via=[(x - 1.1, 4.5), (x - 0.6, 4.5)])
    d.link("v", "mm2", via=[(x + 2.2, 4.5), (x + 0.6, 4.5)])
    d.up("mm2", "out")
    d.text(x - 3.4, 3.2, "n × n\nattention\nweights", fs=8, ha="center", color="#444")
    # right: multi-head
    m = 12.0
    d.title(m, 6.85, "Multi-head attention")
    d.box("x", m, -0.5, "X  (n tokens × d)", "io", w=2.8, h=0.45)
    for j, dxh in enumerate([-2.6, 0, 2.6]):
        hx = m + dxh
        d.box(f"proj{j}", hx, 1.0, "Linear → Q, K, V", "linear", w=2.2, h=0.5, fs=8)
        d.box(f"head{j}", hx, 2.4, f"attention\nhead {j + 1 if j < 2 else 'h'}", "attn", w=2.2, h=0.8, fs=8)
        d.link("x", f"proj{j}", via=[(m, 0.35), (hx, 0.35)])
        d.up(f"proj{j}", f"head{j}")
    d.text(m + 1.3, 2.35, "…", fs=14, ha="center")
    d.box("cat", m, 4.0, "Concatenate heads", "act", w=3.6, h=0.45)
    d.box("wo", m, 5.0, "Linear  W_O  (h·d_k → d)", "linear", w=3.6, h=0.45)
    d.box("mo", m, 6.3, "output  (n × d)", "io", w=2.8, h=0.45)
    for j, dxh in enumerate([-2.6, 0, 2.6]):
        d.link(f"head{j}", "cat", via=[(m + dxh, 3.45), (m + dxh * 0.4, 3.45)])
    d.up("cat", "wo"); d.up("wo", "mo")
    d.text(m, -1.15, "Self-attention: Q, K, V all from X.   Cross-attention: Q from X, K and V from another sequence.",
           fs=8, ha="center", color="#444")
    save(fig, "arch_3_attention.png")


# ------------------------------------------------------------------ 4. channel mixers
def fig_mlps():
    fig, d = new(13, 5.0, (-1, 18.5), (-1.3, 6.1))
    x = 1.6
    d.title(x, 5.6, "MLP")
    d.box("i1", x, -0.6, "x  (d)", "io", w=1.6, h=0.45)
    d.box("l1", x, 0.6, "Linear  d → 4d", "linear")
    d.box("g1", x, 1.7, "GELU / ReLU", "act")
    d.box("l2", x, 2.8, "Linear  4d → d", "linear")
    d.box("o1", x, 4.0, "output  (d)", "io", w=1.6, h=0.45)
    for a, b in [("i1", "l1"), ("l1", "g1"), ("g1", "l2"), ("l2", "o1")]:
        d.up(a, b)
    s = 7.0
    d.title(s, 5.6, "SwiGLU (gated) MLP")
    d.box("i2", s, -0.6, "x  (d)", "io", w=1.6, h=0.45)
    d.box("la", s - 1.3, 0.6, "Linear W₁", "linear", w=2.0)
    d.box("lb", s + 1.3, 0.6, "Linear W₃", "linear", w=2.0)
    d.box("sw", s - 1.3, 1.7, "SiLU", "act", w=2.0)
    d.add("mul", s, 2.6, symbol="×")
    d.box("l3", s, 3.5, "Linear W₂  → d", "linear")
    d.box("o2", s, 4.6, "output  (d)", "io", w=1.6, h=0.45)
    d.link("i2", "la", via=[(s, 0.0), (s - 1.3, 0.0)]); d.link("i2", "lb", via=[(s, 0.0), (s + 1.3, 0.0)])
    d.up("la", "sw")
    d.link("sw", "mul", "top", "left", via=[(s - 1.3, 2.6)])
    d.link("lb", "mul", "top", "right", via=[(s + 1.3, 2.6)])
    d.up("mul", "l3"); d.up("l3", "o2")
    d.text(s, 5.1, "one branch gates the other", fs=8, ha="center", color="#444")
    m = 13.8
    d.title(m, 5.6, "Mixture of experts (top-k routing)")
    d.box("i3", m, -0.6, "token x", "io", w=1.6, h=0.45)
    d.box("rt", m - 3.0, 0.9, "Router\nsoftmax(x W_r)", "router", w=2.2, h=0.7, fs=8)
    for j, ex in enumerate([-0.6, 1.2, 3.0]):
        d.box(f"e{j}", m + ex, 2.2, f"Expert {j + 1 if j < 2 else 'E'}\n(MLP)", "mlp", w=1.5, h=0.7, fs=8)
        d.link("i3", f"e{j}", via=[(m, 0.0), (m + ex, 0.0)])
    d.text(m + 2.1, 2.15, "…", fs=14, ha="center")
    d.link("i3", "rt", via=[(m, 0.0), (m - 3.0, 0.0)])
    d.box("sum", m + 1.2, 3.7, "Σ  gate_e · expert_e(x)\nover the top-k experts", "act", w=3.6, h=0.7, fs=8)
    for j, ex in enumerate([-0.6, 1.2, 3.0]):
        d.link(f"e{j}", "sum", via=[(m + ex, 3.15), (m + 1.2 + (ex - 1.2) * 0.4, 3.15)])
    d._arrow([d.side("rt", "top"), (m - 3.0, 3.7), d.side("sum", "left")])
    d.text(m - 2.9, 2.6, "which experts,\nwith what weight", fs=7.5, color="#444")
    d.box("o3", m + 1.2, 4.9, "output", "io", w=1.6, h=0.45)
    d.up("sum", "o3")
    save(fig, "arch_4_mlps.png")


# ------------------------------------------------------------------ 5. convolutional blocks
def fig_convs():
    fig, d = new(15, 6.6, (-1.4, 22.5), (-1.3, 8.4))

    def stack(prefix, x, items, title, res=True, res_label=None):
        d.title(x, 7.6, title, fs=10)
        d.box(prefix + "in", x, -0.6, "input", "io", w=1.6, h=0.45)
        prev, y = prefix + "in", 0.5
        for j, (lbl, kind) in enumerate(items):
            d.box(prefix + str(j), x, y, lbl, kind, w=2.5, h=0.5, fs=8)
            d.up(prev, prefix + str(j))
            prev, y = prefix + str(j), y + 0.85
        if res:
            d.add(prefix + "add", x, y)
            d.up(prev, prefix + "add")
            d.residual(prefix + "in", prefix + "add", dx=1.65)
            if res_label:
                d.text(x + 1.75, (y - 0.6) / 2, res_label, fs=7.5, color="#444", rotation=90, va="center")
            prev, y = prefix + "add", y + 0.7
        d.box(prefix + "out", x, y, "output", "io", w=1.6, h=0.45)
        d.up(prev, prefix + "out")

    stack("a", 1.3, [("3×3 conv", "conv"), ("BatchNorm", "norm"), ("ReLU", "act"), ("3×3 conv", "conv"),
                     ("BatchNorm", "norm")], "ResNet basic block", res_label="identity (or 1×1 conv)")
    stack("b", 6.8, [("1×1 conv  (reduce)", "conv"), ("BatchNorm, ReLU", "norm"), ("3×3 conv", "conv"),
                     ("BatchNorm, ReLU", "norm"), ("1×1 conv  (expand)", "conv"), ("BatchNorm", "norm")],
          "ResNet bottleneck\n(ResNet-50 and deeper)")
    stack("c", 12.3, [("1×1 conv  (expand 6×)", "conv"), ("BN, ReLU6", "norm"), ("3×3 depthwise conv", "conv"),
                      ("BN, ReLU6", "norm"), ("Squeeze-excite (opt.)", "gate"), ("1×1 conv  (project)", "conv"),
                      ("BatchNorm", "norm")], "MobileNetV2 / EfficientNet\ninverted bottleneck")
    stack("d", 17.8, [("7×7 depthwise conv", "conv"), ("LayerNorm", "norm"), ("1×1 conv  (d → 4d)", "mlp"),
                      ("GELU", "act"), ("1×1 conv  (4d → d)", "mlp")], "ConvNeXt block\n(2022)")
    d.text(10.5, -1.25, "depthwise conv = one filter per channel (token mixing);  1×1 conv = an MLP at every pixel "
           "(channel mixing)", fs=8.5, ha="center", color="#444")
    save(fig, "arch_5_conv_blocks.png")


# ------------------------------------------------------------------ 6. U-Net
def fig_unet():
    fig, d = new(13, 6.4, (-1, 19), (-1.5, 8.2))
    levels = [(0, "64×64×64"), (1, "32×32×128"), (2, "16×16×256"), (3, "8×8×512")]
    xs_down = [1.5, 3.5, 5.5]
    xs_up = [15.5, 13.5, 11.5]
    ys = [6.4, 4.6, 2.8]
    d.title(9.0, 7.6, "U-Net (Ronneberger et al. 2015; the denoiser in diffusion models)")
    d.box("inp", xs_down[0], 7.4 + 0.0 - 0.0, "image", "io", w=1.4, h=0.4)
    for i in range(3):
        d.box(f"dn{i}", xs_down[i], ys[i], f"conv blocks\n{levels[i][1]}", "conv", w=2.2, h=0.7, fs=8)
        d.box(f"up{i}", xs_up[i], ys[i], f"conv blocks\n{levels[i][1]}", "conv", w=2.2, h=0.7, fs=8)
        d._arrow([d.side(f"dn{i}", "right"), d.side(f"up{i}", "left")])
        d.text(9.0, ys[i] + 0.12, "skip: copy & concatenate", fs=7.5, ha="center", color="#444")
    d._arrow([d.side("inp", "bottom"), d.side("dn0", "top")])
    for i in range(2):
        d._arrow([d.side(f"dn{i}", "bottom"), (xs_down[i], ys[i + 1]), d.side(f"dn{i + 1}", "left")])
        d.text(xs_down[i] - 0.2, (ys[i] + ys[i + 1]) / 2 - 0.1, "downsample ↓", fs=7.5, ha="right")
        d._arrow([d.side(f"up{i + 1}", "right"), (xs_up[i], ys[i + 1]), d.side(f"up{i}", "bottom")])
        d.text(xs_up[i] + 0.2, (ys[i] + ys[i + 1]) / 2 - 0.1, "↑ upsample", fs=7.5)
    d.box("mid", 9.0, 0.8, f"bottleneck  {levels[3][1]}\nconv blocks + self-attention", "attn", w=4.2, h=0.8, fs=8)
    d._arrow([d.side("dn2", "bottom"), (xs_down[2], 0.8), d.side("mid", "left")])
    d._arrow([d.side("mid", "right"), (xs_up[2], 0.8), d.side("up2", "bottom")])
    d.box("outp", xs_up[0], 7.4, "output (same size)", "io", w=2.8, h=0.4)
    d._arrow([d.side("up0", "top"), d.side("outp", "bottom")])
    d.text(9.0, -0.6, "Coarse levels see the whole image; skips bring back pixel detail.  In diffusion: every conv block "
           "also receives the time embedding,\nand blocks at low resolution add cross-attention to the text.", fs=8,
           ha="center", color="#444")
    save(fig, "arch_6_unet.png")


# ------------------------------------------------------------------ 7. recurrent and state-space
def fig_recurrent():
    fig, d = new(15, 6.6, (-1, 22.5), (-1.6, 8.4))
    # RNN cell
    x = 2.0
    d.title(x, 7.7, "RNN cell")
    d.box("xt", x, -0.7, "x_t", "io", w=1.0, h=0.45)
    d.box("hprev", x - 2.0, 2.0, "h_{t−1}", "io", w=1.1, h=0.45)
    d.box("cell", x, 2.0, "tanh(W_h h_{t−1}\n+ W_x x_t + b)", "rnn", w=2.4, h=0.8, fs=8)
    d.box("ht", x + 2.4, 2.0, "h_t", "io", w=0.9, h=0.45)
    d.box("yt", x, 4.2, "ŷ_t = softmax(V h_t)", "softmax", w=2.4, h=0.5, fs=8)
    d.up("xt", "cell"); d.link("hprev", "cell", "right", "left"); d.link("cell", "ht", "right", "left")
    d.up("cell", "yt")
    d.text(x, -1.45, "same weights at every step;\ngradients pass through every step", fs=8, ha="center", color="#444")
    # LSTM
    L = 9.0
    d.title(L, 7.7, "LSTM cell")
    d.box("c_prev", L - 3.3, 6.2, "c_{t−1}", "io", w=1.1, h=0.45)
    d.add("fmul", L - 1.4, 6.2, symbol="×")
    d.add("cadd", L + 0.6, 6.2)
    d.box("c_t", L + 3.3, 6.2, "c_t  (cell state)", "io", w=1.7, h=0.45, fs=8)
    d.link("c_prev", "fmul", "right", "left"); d.link("fmul", "cadd", "right", "left"); d.link("cadd", "c_t", "right", "left")
    d.box("inp", L, 0.2, "[h_{t−1}, x_t]", "io", w=2.0, h=0.45)
    gates = [("f", L - 2.7, "forget gate\nσ", "gate"), ("i", L - 0.9, "input gate\nσ", "gate"),
             ("g", L + 0.9, "candidate\ntanh", "rnn"), ("o", L + 2.7, "output gate\nσ", "gate")]
    for nm, gx, lbl, kind in gates:
        d.box(nm, gx, 2.0, lbl, kind, w=1.55, h=0.75, fs=8)
        d.link("inp", nm, via=[(L, 0.9), (gx, 0.9)])
    d._arrow([d.side("f", "top"), (L - 2.7, 5.0), (L - 1.4, 5.0), d.side("fmul", "bottom")])
    d.add("imul", L - 0.1, 4.2, symbol="×")
    d.link("i", "imul", "top", "left", via=[(L - 0.9, 4.2)])
    d.link("g", "imul", "top", "right", via=[(L + 0.9, 4.2)])
    d._arrow([d.side("imul", "top"), (L - 0.1, 5.2), (L + 0.6, 5.2), d.side("cadd", "bottom")])
    d.box("tanhc", L + 3.3, 5.1, "tanh", "act", w=1.0, h=0.4)
    d._arrow([d.side("c_t", "bottom"), d.side("tanhc", "top")])
    d.add("omul", L + 3.3, 3.9, symbol="×")
    d.link("tanhc", "omul", "bottom", "top")
    d.link("o", "omul", "top", "left", via=[(L + 2.7, 3.9)])
    d.box("h_t", L + 4.9, 3.9, "h_t", "io", w=0.9, h=0.45)
    d.link("omul", "h_t", "right", "left")
    d.text(L, -1.45, "the cell state is copied forward (× forget gate ≈ 1),\nso information and gradients survive long gaps",
           fs=8, ha="center", color="#444")
    # Mamba
    M = 18.6
    d.title(M, 7.7, "Mamba block")
    d.box("m_in", M, -0.7, "x", "io", w=1.0, h=0.45)
    d.box("m_n", M, 0.3, "RMSNorm", "norm", w=2.0, h=0.45)
    d.box("m_la", M - 1.3, 1.4, "Linear", "linear", w=1.6, h=0.45)
    d.box("m_lb", M + 1.3, 1.4, "Linear", "linear", w=1.6, h=0.45)
    d.box("m_conv", M - 1.3, 2.4, "short causal\nconv", "conv", w=1.6, h=0.6, fs=8)
    d.box("m_act", M - 1.3, 3.4, "SiLU", "act", w=1.6, h=0.4)
    d.box("m_ssm", M - 1.3, 4.5, "selective SSM\nh_t = A_t h_{t−1} + B_t x_t", "ssm", w=2.75, h=0.8, fs=7.5)
    d.box("m_silu", M + 1.3, 2.9, "SiLU", "act", w=1.6, h=0.4)
    d.add("m_mul", M, 5.6, symbol="×")
    d.box("m_out", M, 6.4, "Linear", "linear", w=2.0, h=0.4)
    d.add("m_add", M, 7.1)
    d.up("m_in", "m_n")
    d.link("m_n", "m_la", via=[(M, 0.85), (M - 1.3, 0.85)]); d.link("m_n", "m_lb", via=[(M, 0.85), (M + 1.3, 0.85)])
    for a, b in [("m_la", "m_conv"), ("m_conv", "m_act"), ("m_act", "m_ssm"), ("m_lb", "m_silu")]:
        d.up(a, b)
    d.link("m_ssm", "m_mul", "top", "left", via=[(M - 1.3, 5.6)])
    d.link("m_silu", "m_mul", "top", "right", via=[(M + 1.3, 5.6)])
    d.up("m_mul", "m_out"); d.up("m_out", "m_add")
    d.residual("m_in", "m_add", dx=2.5)
    d.text(M, -1.45, "A_t, B_t depend on the input (\"selective\");\nO(n) in length, trained with a parallel scan",
           fs=8, ha="center", color="#444")
    save(fig, "arch_7_recurrent.png")


# ------------------------------------------------------------------ 8. Vision Transformer and DiT
def fig_vit():
    fig, d = new(13, 5.6, (-1, 19), (-1.3, 6.6))
    x = 4.5
    d.title(x, 6.0, "Vision Transformer (ViT)")
    d.box("img", x, -0.6, "image → 16×16 patches", "io", w=3.2, h=0.45)
    d.box("pe", x, 0.5, "Linear patch embedding", "emb", w=3.2)
    d.add("p+", x, 1.4)
    d.box("pos", x - 3.0, 1.4, "position\nembeddings", "pos", w=1.6, h=0.6, fs=8)
    d.box("cls", x + 3.0, 1.4, "[CLS] token", "emb", w=1.6, h=0.5, fs=8)
    d.link("pos", "p+", "right", "left"); d.link("cls", "p+", "left", "right")
    d.box("enc", x, 2.9, "Transformer encoder blocks\n(pre-norm attention + MLP) × L", "attn", w=4.0, h=0.9, fs=8.5)
    d.box("hn", x, 4.3, "LayerNorm", "norm", w=3.2)
    d.box("head", x, 5.3, "MLP head on [CLS] → class", "mlp", w=3.8)
    for a, b in [("img", "pe"), ("pe", "p+"), ("p+", "enc"), ("enc", "hn"), ("hn", "head")]:
        d.up(a, b)
    t = 15.0
    d.title(t - 1.2, 6.0, "DiT block (diffusion Transformer)")
    d.box("t_in", t, -0.6, "latent patches", "io", w=2.4, h=0.45)
    d.box("cond", t - 3.9, 0.9, "time step +\nclass / text", "pos", w=1.8, h=0.7, fs=8)
    d.box("ada", t - 3.9, 2.3, "MLP → γ, β, α\n(per channel)", "linear", w=1.8, h=0.7, fs=8)
    d.up("cond", "ada")
    top = prenorm_block(d, t, 0.4, "Multi-head\nself-attention", "attn", "MLP", "mlp", norm="adaLN: γ·LN(x)+β",
                        prefix="dt", h_mixer=0.7, w=2.6, fs=8)
    d.up("t_in", "dtn1")
    d._arrow([d.side("ada", "top"), (t - 3.9, d.nodes["dtn2"][1]), d.side("dtn2", "left")])
    d._arrow([d.side("ada", "right"), (t - 2.2, 2.3), (t - 2.2, 0.4), d.side("dtn1", "left")])
    d.text(t - 3.9, 4.25, "conditioning sets each\nnorm's scale and shift", fs=7.5, ha="center", color="#444")
    save(fig, "arch_8_vit_dit.png")


def main():
    for f in [fig_skeleton, fig_transformers, fig_attention, fig_mlps, fig_convs, fig_unet, fig_recurrent, fig_vit]:
        f()
    print("wrote", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
