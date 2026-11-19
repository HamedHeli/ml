"""The LLM architectures in Sebastian Raschka's LLM Architecture Gallery, as data, plus the trend analysis and figures
used in llm-architecture-design.md.

Source: https://sebastianraschka.com/llm-architecture-gallery/ (read 2026-11-19). The gallery lists 126 models; the
77 whose cards could be read are coded here, in the gallery's order. Fields the cards do not state are None, except
`norm`, where well-known configurations (Llama, Qwen, DeepSeek, ...) were filled in from the models' published configs;
those are listed in NORM_FROM_CONFIG so the analysis can separate them.

Run:  python 17-sequence-models-and-transformers/llm_architectures.py
      (prints the trend tables, writes llm_architectures.csv and figures/llm_*.png)
"""

from __future__ import annotations

import csv
import importlib.util
import os
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


@dataclass
class Model:
    name: str
    org: str
    date: str                      # YYYY-MM
    total_b: float                 # total parameters, billions
    active_b: float                # active parameters per token, billions (= total for dense)
    layers: int | None
    attn: str                      # global token mixer: MHA, GQA, MQA, MLA, CSA/HCA, none
    hybrid: str | None = None      # second token mixer: SWA, chunked, DeltaNet, KDA, Lightning, Mamba-2, conv, mLSTM
    n_local: int | None = None     # layers using the cheap mixer (sliding window, linear attention, SSM, conv)
    n_global: int | None = None    # full (global) attention layers
    qk_norm: bool = False
    norm: str | None = None        # pre, post-inside, sandwich, parallel (None = not stated)
    pos: str = "RoPE"              # learned, RoPE, partial RoPE, RoPE+NoPE, NoPE+recurrent, none
    ffn: str = "SwiGLU"            # SwiGLU, GELU, GeGLU, ReLU2, SiLU, (gated) MLP
    moe: bool = False
    experts: int | None = None     # routed experts
    top_k: int | None = None       # routed experts active per token
    shared: bool | None = None     # always-on shared expert
    dense_prefix: bool | None = None
    mtp: bool = False              # multi-token prediction head(s)
    context_k: float | None = None
    kv_kib: float | None = None    # KV cache per token, bf16, as given on the card
    notes: str = ""

    @property
    def active_frac(self):
        return None if self.active_b is None else self.active_b / self.total_b

    @property
    def year(self):
        return float(self.date[:4]) + (int(self.date[5:7]) - 1) / 12


def M(*a, **k):
    return Model(*a, **k)


MODELS = [
    M("GPT-2 XL", "OpenAI", "2019-11", 1.5, 1.5, 48, "MHA", norm="pre", pos="learned", ffn="GELU", context_k=1,
      kv_kib=300, notes="LayerNorm, dropout 0.1, biases everywhere"),
    M("Llama 3 8B", "Meta", "2024-04", 8, 8, 32, "GQA", norm="pre", context_k=8, kv_kib=128),
    M("Llama 3.2 1B", "Meta", "2024-09", 1, 1, 16, "GQA", norm="pre", context_k=128, kv_kib=32,
      notes="tied embeddings; pruned + distilled"),
    M("OLMo 2 7B", "Ai2", "2024-11", 7, 7, 32, "MHA", qk_norm=True, norm="post-inside", context_k=4, kv_kib=512),
    M("DeepSeek V3", "DeepSeek", "2024-12", 671, 37, 61, "MLA", moe=True, experts=256, top_k=8, shared=True,
      dense_prefix=True, mtp=True, context_k=128, kv_kib=68.6),
    M("DeepSeek R1", "DeepSeek", "2025-01", 671, 37, 61, "MLA", moe=True, experts=256, top_k=8, shared=True,
      dense_prefix=True, mtp=True, context_k=128, kv_kib=68.6, notes="V3 architecture, reasoning post-training"),
    M("Gemma 3 27B", "Google", "2025-03", 27, 27, 62, "GQA", "SWA", 52, 10, qk_norm=True, norm="sandwich",
      ffn="GeGLU", context_k=128, kv_kib=496, notes="vocab 262k"),
    M("Mistral Small 3.1 24B", "Mistral AI", "2025-03", 24, 24, 40, "GQA", norm="pre", context_k=128, kv_kib=160,
      notes="dropped the sliding window of earlier Mistrals"),
    M("Llama 4 Maverick", "Meta", "2025-04", 400, 17, 48, "GQA", "chunked", 36, 12, norm="pre", pos="RoPE+NoPE",
      moe=True, experts=128, top_k=1, shared=True, context_k=1000, kv_kib=192,
      notes="NoPE in the full-attention layers; dense and MoE blocks alternate"),
    M("Qwen3 0.6B", "Alibaba", "2025-04", 0.6, 0.6, 28, "GQA", qk_norm=True, norm="pre", context_k=32, kv_kib=112),
    M("Qwen3 235B-A22B", "Alibaba", "2025-04", 235, 22, 94, "GQA", qk_norm=True, norm="pre", moe=True, experts=128,
      top_k=8, shared=False, context_k=128, kv_kib=188),
    M("Qwen3 30B-A3B", "Alibaba", "2025-04", 30, 3, 48, "GQA", qk_norm=True, norm="pre", moe=True, experts=128,
      top_k=8, shared=False, context_k=128, kv_kib=96),
    M("Qwen3 32B", "Alibaba", "2025-04", 32, 32, 64, "GQA", qk_norm=True, norm="pre", context_k=128, kv_kib=256),
    M("Qwen3 4B", "Alibaba", "2025-04", 4, 4, 36, "GQA", qk_norm=True, norm="pre", context_k=32, kv_kib=144),
    M("Qwen3 8B", "Alibaba", "2025-04", 8, 8, 36, "GQA", qk_norm=True, norm="pre", context_k=128, kv_kib=144),
    M("SmolLM3 3B", "Hugging Face", "2025-06", 3, 3, 36, "GQA", norm="pre", pos="RoPE+NoPE", context_k=128,
      kv_kib=72, notes="every fourth layer drops RoPE"),
    M("Kimi K2", "Moonshot AI", "2025-07", 1000, 32, 61, "MLA", norm="pre", moe=True, experts=384, top_k=8,
      shared=True, dense_prefix=True, context_k=128, kv_kib=68.6, notes="DeepSeek V3 template, more experts"),
    M("GLM-4.5", "Z.ai", "2025-07", 355, 32, 92, "GQA", qk_norm=True, norm="pre", moe=True, experts=160, top_k=8,
      shared=True, dense_prefix=True, mtp=True, context_k=128, kv_kib=368),
    M("GPT-OSS 120B", "OpenAI", "2025-08", 117, 5.1, 36, "GQA", "SWA", 18, 18, norm="pre", moe=True, experts=128,
      top_k=4, shared=False, context_k=128, kv_kib=72, notes="attention bias and learned attention sinks"),
    M("GPT-OSS 20B", "OpenAI", "2025-08", 21, 3.6, 24, "GQA", "SWA", 12, 12, norm="pre", moe=True, experts=32,
      top_k=4, shared=False, context_k=128, kv_kib=48, notes="wider and shallower than Qwen3"),
    M("Gemma 3 270M", "Google", "2025-08", 0.27, 0.27, 18, "MQA", "SWA", 15, 3, qk_norm=True, norm="sandwich",
      ffn="GeGLU", context_k=32, kv_kib=18),
    M("Grok 2.5", "xAI", "2025-08", 270, None, 64, "GQA", moe=True, experts=8, top_k=2, shared=True,
      context_k=128, kv_kib=256, notes="active params not on the card; always-on SwiGLU path acts like a shared expert"),
    M("Qwen3 Next 80B-A3B", "Alibaba", "2025-09", 80, 3, 48, "GQA", "DeltaNet", 36, 12, qk_norm=True, norm="pre",
      moe=True, experts=512, top_k=10, shared=True, mtp=True, context_k=256, kv_kib=24, notes="gated attention"),
    M("MiniMax M2", "MiniMax", "2025-10", 230, 10, 62, "GQA", qk_norm=True, norm="pre", pos="partial RoPE", moe=True,
      experts=256, top_k=8, mtp=True, context_k=192, kv_kib=248),
    M("Kimi Linear 48B-A3B", "Moonshot AI", "2025-10", 48, 3, 27, "MLA", "KDA", 20, 7, norm="pre",
      pos="NoPE+recurrent", moe=True, experts=256, top_k=8, shared=True, context_k=1000, kv_kib=7.9,
      notes="short conv on Q/K/V in KDA layers"),
    M("OLMo 3 32B", "Ai2", "2025-11", 32, 32, 64, "GQA", "SWA", 48, 16, qk_norm=True, norm="post-inside",
      context_k=64, kv_kib=256, notes="YaRN on global layers only"),
    M("OLMo 3 7B", "Ai2", "2025-11", 7, 7, 32, "MHA", "SWA", 24, 8, qk_norm=True, norm="post-inside",
      context_k=64, kv_kib=512),
    M("DeepSeek V3.2", "DeepSeek", "2025-12", 671, 37, 61, "MLA", "DSA", norm="pre", moe=True, experts=256, top_k=8,
      shared=True, dense_prefix=True, mtp=True, context_k=128, kv_kib=68.6, notes="DeepSeek Sparse Attention"),
    M("Mistral Large 3", "Mistral AI", "2025-12", 673, 41, 61, "MLA", norm="pre", moe=True, shared=True,
      context_k=256, kv_kib=68.6, notes="near-clone of DeepSeek V3, fewer larger experts"),
    M("Nemotron 3 Nano 30B-A3B", "NVIDIA", "2025-12", 30, 3, 52, "GQA", "Mamba-2", 23, 6, norm="pre",
      pos="NoPE+recurrent", ffn="ReLU2", moe=True, context_k=1000, kv_kib=6, notes="52 blocks: 6 GQA, 23 Mamba-2, 23 MoE"),
    M("MiMo-V2-Flash", "Xiaomi", "2025-12", 309, 15, 48, "GQA", "SWA", 40, 8, moe=True, mtp=True, context_k=256,
      kv_kib=144, notes="128-token window"),
    M("GLM-4.7", "Z.ai", "2025-12", 355, 32, 92, "GQA", qk_norm=True, norm="pre", moe=True, experts=160, top_k=8,
      shared=True, dense_prefix=True, mtp=True, context_k=198, kv_kib=368),
    M("Arcee Trinity Large", "Arcee AI", "2026-01", 400, 13, 60, "GQA", "SWA", 45, 15, qk_norm=True, norm="sandwich",
      pos="RoPE+NoPE", moe=True, context_k=512, kv_kib=240, notes="gated attention, coarse-grained MoE"),
    M("GLM-5", "Z.ai", "2026-02", 744, 40, 78, "MLA", "DSA", moe=True, mtp=True, context_k=198, kv_kib=87.8),
    M("Nemotron 3 Super", "NVIDIA", "2026-03", 120, 12, 88, "GQA", "Mamba-2", 40, 8, norm="pre", pos="NoPE+recurrent",
      ffn="ReLU2", moe=True, mtp=True, context_k=1000, kv_kib=8, notes="latent-space MoE"),
    M("Gemma 4 31B", "Google", "2026-04", 30.7, 30.7, 60, "GQA", "SWA", 50, 10, qk_norm=True, norm="sandwich",
      pos="partial RoPE", ffn="GeGLU", context_k=256, kv_kib=840, notes="p-RoPE and unified K/V on global layers"),
    M("Gemma 4 26B-A4B", "Google", "2026-04", 25.2, 3.8, 30, "GQA", "SWA", 25, 5, qk_norm=True, norm="sandwich",
      pos="partial RoPE", ffn="GeGLU", moe=True, experts=128, top_k=8, shared=True, context_k=256, kv_kib=210),
    M("Gemma 4 12B", "Google", "2026-06", 12, 12, 48, "GQA", "SWA", 40, 8, qk_norm=True, norm="sandwich",
      pos="partial RoPE", ffn="GeGLU", context_k=256, kv_kib=328),
    M("Llama 3.2 3B", "Meta", "2024-09", 3, 3, 28, "GQA", norm="pre", context_k=128, kv_kib=112,
      notes="tied embeddings"),
    M("Qwen3 Coder Flash 30B-A3B", "Alibaba", "2025-07", 30, 3.3, 48, "GQA", qk_norm=True, norm="pre", moe=True,
      experts=128, top_k=8, shared=False, context_k=256, kv_kib=96),
    M("Kimi K2.5", "Moonshot AI", "2026-01", 1000, 32, 61, "MLA", norm="pre", moe=True, experts=384, top_k=8,
      shared=True, dense_prefix=True, context_k=256, kv_kib=68.6),
    M("Step 3.5 Flash", "StepFun", "2026-02", 196, 11, 48, "GQA", "SWA", 36, 12, moe=True, mtp=True, context_k=256,
      kv_kib=192),
    M("Nanbeige 4.1 3B", "BOSS Zhipin", "2026-02", 3, 3, 32, "GQA", norm="pre", context_k=256, kv_kib=64,
      notes="untied embeddings"),
    M("MiniMax M2.5", "MiniMax", "2026-02", 230, 10, 62, "GQA", qk_norm=True, norm="pre", pos="partial RoPE",
      moe=True, mtp=True, context_k=192, kv_kib=248, notes="deliberately avoids SWA and linear hybrids"),
    M("Tiny Aya 3.35B", "Cohere", "2026-02", 3.35, 3.35, 36, "GQA", "SWA", 27, 9, norm="parallel", pos="RoPE+NoPE",
      context_k=8, kv_kib=72, notes="attention and MLP in parallel"),
    M("Ling 2.5 1T", "InclusionAI", "2026-02", 1000, 63, 80, "MLA", "Lightning", 70, 10, moe=True, context_k=256,
      kv_kib=11.2),
    M("Qwen3.5 397B", "Alibaba", "2026-02", 397, 17, 60, "GQA", "DeltaNet", 45, 15, qk_norm=True, norm="pre",
      moe=True, experts=512, context_k=256, kv_kib=30, notes="gated attention"),
    M("Sarvam 30B", "Sarvam AI", "2026-03", 30, 2.4, 19, "GQA", qk_norm=True, moe=True, context_k=128, kv_kib=19),
    M("Sarvam 105B", "Sarvam AI", "2026-03", 105, 10.3, 32, "MLA", pos="RoPE+NoPE", moe=True, context_k=128,
      kv_kib=36, notes="KV LayerNorm"),
    M("Phi-4 14B", "Microsoft", "2024-12", 14, 14, 40, "GQA", norm="pre", context_k=16, kv_kib=200),
    M("xLSTM 7B", "NXAI", "2025-03", 7, 7, 32, "none", "mLSTM", 32, 0, norm="pre", pos="none", ffn="gated MLP",
      kv_kib=0, notes="no attention; matrix-memory recurrence"),
    M("GLM-4.5-Air", "Z.ai", "2025-07", 106, 12, 46, "GQA", qk_norm=True, norm="pre", moe=True, shared=True,
      dense_prefix=True, mtp=True, context_k=128, kv_kib=184),
    M("Ouro-Thinking 2.6B", "ByteDance", "2025-10", 2.6, 2.6, 48, "MHA", norm="sandwich", ffn="SiLU", context_k=64,
      kv_kib=1536, notes="looped: 48 shared layers x 4 passes, learned exit gate"),
    M("INTELLECT-3", "Prime Intellect", "2025-11", 106, 12, 46, "GQA", qk_norm=True, norm="pre", moe=True,
      shared=True, dense_prefix=True, mtp=True, context_k=128, kv_kib=184, notes="GLM-4.5-Air architecture"),
    M("LongCat-Flash-Lite", "Meituan", "2026-01", 68.5, 3, 28, "MLA", pos="RoPE+NoPE", ffn="SiLU", moe=True,
      experts=256, top_k=12, context_k=256, kv_kib=31.5, notes="128 zero/identity experts; n-gram embeddings"),
    M("Mistral Small 4", "Mistral AI", "2026-03", 119, 6.63, 36, "MLA", moe=True, experts=128, top_k=4, shared=True,
      context_k=256, kv_kib=22.5),
    M("Nemotron 3 Nano 4B", "NVIDIA", "2026-03", 4, 4, 42, "GQA", "Mamba-2", 21, 4, norm="pre", pos="NoPE+recurrent",
      ffn="ReLU2", context_k=256, kv_kib=16),
    M("MiniMax M2.7", "MiniMax", "2026-03", 230, 10, 62, "GQA", qk_norm=True, norm="pre", pos="partial RoPE",
      moe=True, mtp=True, context_k=200, kv_kib=248),
    M("Gemma 4 E2B", "Google", "2026-04", 5.1, 2.3, 35, "MQA", "SWA", 28, 7, qk_norm=True, norm="sandwich",
      pos="partial RoPE", ffn="GeGLU", context_k=128, kv_kib=35, notes="per-layer embeddings, KV sharing"),
    M("Gemma 4 E4B", "Google", "2026-04", 8, 4.5, 42, "GQA", "SWA", 35, 7, qk_norm=True, norm="sandwich",
      pos="partial RoPE", ffn="GeGLU", context_k=128, kv_kib=84, notes="per-layer embeddings, KV sharing"),
    M("DeepSeek V4-Flash", "DeepSeek", "2026-04", 284, 13, 43, "CSA/HCA", moe=True, experts=256, top_k=6,
      shared=True, mtp=True, context_k=1024, kv_kib=5.4, notes="compressed attention; mHC residual; hash routing"),
    M("DeepSeek V4-Pro", "DeepSeek", "2026-04", 1600, 49, 61, "CSA/HCA", moe=True, experts=384, top_k=6, shared=True,
      mtp=True, context_k=1024, kv_kib=7.7, notes="compressed attention; mHC residual; hash routing"),
    M("Laguna XS.2", "Poolside", "2026-04", 33, 3, 40, "GQA", "SWA", 30, 10, qk_norm=True, moe=True, experts=None,
      top_k=8, shared=True, context_k=256, kv_kib=160, notes="gated attention, 512-token window, sigmoid routing"),
    M("ZAYA1-8B", "Zyphra", "2026-05", 8.4, 0.76, 40, "GQA", qk_norm=True, ffn="SiLU", moe=True, top_k=1,
      context_k=128, kv_kib=40, notes="compressed convolutional attention (CCA)"),
    M("GLM-5.1", "Z.ai", "2026-04", 744, 40, 78, "MLA", "DSA", moe=True, mtp=True, context_k=198, kv_kib=87.8),
    M("Qwen3.6 35B-A3B", "Alibaba", "2026-04", 35, 3, 40, "GQA", "DeltaNet", 30, 10, qk_norm=True, norm="pre",
      moe=True, experts=256, top_k=8, shared=True, context_k=256, kv_kib=20),
    M("Kimi K2.6", "Moonshot AI", "2026-04", 1000, 32, 61, "MLA", norm="pre", moe=True, experts=384, top_k=8,
      shared=True, dense_prefix=True, context_k=256, kv_kib=68.6),
    M("Qwen3.6 27B", "Alibaba", "2026-04", 27, 27, 64, "GQA", "DeltaNet", 48, 16, qk_norm=True, norm="pre",
      context_k=256, kv_kib=64),
    M("MiMo-V2.5", "Xiaomi", "2026-04", 310, 15, 48, "GQA", "SWA", 39, 9, moe=True, mtp=True, context_k=1024,
      kv_kib=144),
    M("MiMo-V2.5-Pro", "Xiaomi", "2026-04", 1020, 42, 70, "GQA", "SWA", 60, 10, moe=True, experts=384, top_k=8,
      context_k=1024, kv_kib=350),
    M("Ling 2.6 1T", "InclusionAI", "2026-04", 1000, 63, 80, "MLA", "Lightning", 70, 10, moe=True, mtp=True,
      context_k=256, kv_kib=11.2),
    M("Tencent Hy3-preview", "Tencent", "2026-04", 295, 21, 80, "GQA", qk_norm=True, moe=True, experts=192, top_k=8,
      shared=True, dense_prefix=True, mtp=True, context_k=256, kv_kib=320, notes="sigmoid routing with expert bias"),
    M("Granite 4.1 30B", "IBM", "2026-04", 30, 30, 64, "GQA", norm="pre", context_k=128, kv_kib=256,
      notes="tied embeddings, muP scaling"),
    M("Command A+", "Cohere", "2026-05", 218, 25, 32, "GQA", "SWA", 24, 8, norm="parallel", pos="RoPE+NoPE",
      moe=True, context_k=128, kv_kib=128, notes="attention and MoE in parallel"),
    M("LFM2.5 1.2B", "Liquid AI", "2026-05", 1.2, 1.2, 16, "GQA", "conv", 10, 6, qk_norm=True, ffn="SiLU",
      context_k=32, kv_kib=12),
    M("LFM2.5 350M", "Liquid AI", "2026-05", 0.35, 0.35, 16, "GQA", "conv", 10, 6, qk_norm=True, ffn="SiLU",
      context_k=32, kv_kib=12),
    M("LFM2.5 8B-A1B", "Liquid AI", "2026-05", 8.3, 1.5, None, "GQA", "conv", qk_norm=True, ffn="SiLU", moe=True,
      experts=32, top_k=4, context_k=128),
]

# Norm placement for these came from the models' published configs, not from the gallery card.
NORM_FROM_CONFIG = {m.name for m in MODELS if m.norm == "pre" and m.name not in {"Phi-4 14B"}}

# The 49 models the gallery lists whose cards could not be read (the page's text ends at LFM2.5 8B-A1B).
UNREAD = [
    "JetBrains Mellum2 Thinking", "Nemotron 3 Ultra", "North Mini Code", "Kimi K2.7 Code", "MiniMax M3",
    "VibeThinker-3B", "GLM-5.2", "Laguna XS 2.1", "Soofi S", "Inkling", "BTL-3", "Antares", "Laguna S 2.1",
    "Motif 3 Beta", "Nanbeige 4.2", "Solar Open 2", "Kimi K3", "Ling 3.0 Flash", "Muse Glimmer",
    "Nemotron 3.5 Lightning", "Qwen3.8 27B", "Ornith 1.5", "GLM-5.3-Flash", "Qwen3.8-Flash-Next",
    "Tencent Hy4-preview", "DeepSeek V4.1-Flash", "Step-5 Preview", "MiMo-V2.6-Flash-RL", "MiMo-V2.6-Pro-RL",
    "Ember-1", "Naive-N0.5-Flash", "Kolibri-1", "Ling 3.1 Flash", "Muse Spark Open", "Nemotron 3.5 Super VL",
    "Reflection Beam", "Dory1.0", "Recursive Flint", "MiniMax M3.1-Flash", "Nanbeige 4.3", "LFM3",
    "Mistral Large 4", "LongCat-2.5", "Kimi K3.1 Pro", "ERNIE 5.5 Flash Base", "Nous Research Arete",
    "Inkling 2 Small", "Datology XL", "Marin Mochi",
]


# ---------------------------------------------------------------- classification
FAMILIES = OrderedDict([
    ("A", "Dense, full attention (Llama template)"),
    ("B", "Dense, local:global attention"),
    ("C", "MoE, full GQA attention"),
    ("D", "MoE + MLA (DeepSeek template)"),
    ("E", "MoE, local:global attention"),
    ("F", "Linear-attention hybrid (DeltaNet, KDA, Lightning)"),
    ("G", "SSM / conv hybrid (Mamba-2, LIV conv)"),
    ("H", "Outliers: no attention, looped, learned positions"),
])


def family(m: Model) -> str:
    if m.attn == "none" or "looped" in m.notes or m.pos == "learned":
        return "H"
    if m.hybrid in ("DeltaNet", "KDA", "Lightning"):
        return "F"
    if m.hybrid in ("Mamba-2", "conv"):
        return "G"
    local = m.hybrid in ("SWA", "chunked")
    if not m.moe:
        return "B" if local else "A"
    if local:
        return "E"
    if m.attn in ("MLA", "CSA/HCA"):
        return "D"
    return "C"


def period(m: Model) -> str:
    if m.date < "2025-01":
        return "≤2024"
    if m.date < "2025-07":
        return "2025 H1"
    if m.date < "2026-01":
        return "2025 H2"
    return "2026"


PERIODS = ["≤2024", "2025 H1", "2025 H2", "2026"]

FEATURES = OrderedDict([
    ("KV-sharing attention (GQA/MQA/MLA/compressed)", lambda m: m.attn in ("GQA", "MQA", "MLA", "CSA/HCA")),
    ("QK-norm", lambda m: m.qk_norm),
    ("Sparse MoE", lambda m: m.moe),
    ("Cheap token mixer in most layers (SWA/linear/SSM/conv)", lambda m: m.n_local is not None and m.n_local > 0),
    ("Some layers without RoPE (NoPE)", lambda m: m.pos in ("RoPE+NoPE", "NoPE+recurrent")),
    ("Multi-token prediction", lambda m: m.mtp),
    ("Gated FFN (SwiGLU/GeGLU)", lambda m: m.ffn in ("SwiGLU", "GeGLU", "SiLU", "gated MLP")),
])


def adoption_table(models=MODELS):
    """feature -> {period: (count, n)}"""
    by = {p: [m for m in models if period(m) == p] for p in PERIODS}
    return OrderedDict((f, {p: (sum(fn(m) for m in by[p]), len(by[p])) for p in PERIODS}) for f, fn in FEATURES.items())


def local_global_ratio(m: Model):
    if not m.n_local or not m.n_global:
        return None
    return m.n_local / m.n_global


def ratio_summary(models=MODELS):
    """cheap-mixer type -> sorted list of (ratio, model name)"""
    out = {}
    for m in models:
        r = local_global_ratio(m)
        if r is not None:
            out.setdefault(m.hybrid, []).append((round(r, 2), m.name))
    return {k: sorted(v) for k, v in out.items()}


def kv_by_attention(models=MODELS):
    """median KV cache per token (KiB) by attention class"""
    import statistics
    groups = {}
    for m in models:
        if m.kv_kib is None or m.attn == "none":
            continue
        cls = m.attn if m.hybrid not in ("DeltaNet", "KDA", "Lightning", "Mamba-2", "conv") else "hybrid"
        if m.hybrid == "SWA":
            cls += "+SWA"
        groups.setdefault(cls, []).append(m.kv_kib)
    return {k: (statistics.median(v), len(v)) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_adoption(path):
    plt = _plt()
    tab = adoption_table()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = range(len(PERIODS))
    for f, row in tab.items():
        ys = [100 * c / n if n else 0 for c, n in row.values()]
        ax.plot(xs, ys, marker="o", lw=2, label=f)
    ax.set_xticks(list(xs), [f"{p}\n(n={row[1]})" for p, row in zip(PERIODS, next(iter(tab.values())).values())])
    ax.set_ylabel("% of gallery models in that period")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Adoption of design choices across the 77 readable gallery models", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_kv(path):
    plt = _plt()
    colors = {"MHA": "#c62828", "MQA": "#8e24aa", "GQA": "#1e88e5", "MLA": "#43a047", "CSA/HCA": "#00897b"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for m in MODELS:
        if m.kv_kib is None or m.kv_kib == 0 or m.date < "2024":
            continue
        hyb = m.hybrid in ("DeltaNet", "KDA", "Lightning", "Mamba-2", "conv")
        marker = "^" if m.hybrid == "SWA" or m.hybrid == "chunked" else ("s" if hyb else "o")
        ax.scatter(m.year, m.kv_kib, c=colors[m.attn], marker=marker, s=40 + 6 * (m.active_b or 0) ** 0.5,
                   edgecolor="k", lw=0.4, alpha=0.85)
        if m.name in {"GPT-2 XL", "OLMo 2 7B", "Gemma 4 31B", "DeepSeek V3", "Qwen3 Next 80B-A3B", "Ouro-Thinking 2.6B",
                      "Nemotron 3 Nano 30B-A3B", "DeepSeek V4-Flash", "MiniMax M2", "Kimi Linear 48B-A3B"}:
            ax.annotate(m.name, (m.year, m.kv_kib), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.text(0.01, 0.97, "for scale: GPT-2 XL (2019, MHA) = 300 KiB", transform=ax.transAxes, fontsize=8, va="top",
            color="#c62828")
    ax.set_ylabel("KV cache per token, KiB (bf16, log scale)")
    ax.set_xlabel("release date")
    for k, c in colors.items():
        ax.scatter([], [], c=c, label=k, edgecolor="k", lw=0.4)
    ax.scatter([], [], marker="o", c="w", edgecolor="k", label="full attention only")
    ax.scatter([], [], marker="^", c="w", edgecolor="k", label="+ sliding-window / chunked layers")
    ax.scatter([], [], marker="s", c="w", edgecolor="k", label="+ linear-attention / SSM / conv layers")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("What the attention choices buy: memory per token of context", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _diagrams():
    path = os.path.join(HERE, "..", "15-neural-networks", "diagrams.py")
    spec = importlib.util.spec_from_file_location("nn_diagrams", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig_block_evolution(path):
    """Five decoder blocks, 2019 → 2026, drawn in the style of 15-neural-networks/building-blocks.md."""
    D = _diagrams()
    fig, d = D.new(17, 6.4, (-1.4, 26.2), (-1.9, 9.0))
    W = 3.0

    def column(i, title, sub, rows, residual_pairs, notes):
        x = 2.0 + i * 5.3
        d.title(x, 8.45, title, fs=10)
        d.text(x, 8.1, sub, fs=8, ha="center", color="#444")
        d.box(f"in{i}", x, -0.9, "x (residual stream)", "io", w=W, fs=8)
        prev, y = f"in{i}", -0.1
        for j, (label, kind, h) in enumerate(rows):
            name = f"c{i}_{j}"
            if kind == "+":
                d.add(name, x, y + 0.1)
                y += 0.55
            else:
                d.box(name, x, y + h / 2, label, kind, w=W, h=h, fs=8)
                y += h + 0.3
            d.up(prev, name)
            prev = name
        for a, b in residual_pairs:
            d.residual(f"c{i}_{a}", f"c{i}_{b}", dx=W / 2 + 0.3, below=(a == 0 or rows[a][1] != "+"))
        d.frame(x - W / 2 - 0.2, -0.3, x + W / 2 + 0.55, y - 0.1)
        d.text(x - W / 2 - 0.1, y + 0.05, "× N", fs=9, fontweight="bold")
        d.text(x, -1.6, notes, fs=7.5, ha="center", color="#333")

    column(0, "GPT-2 (2019)", "pre-norm baseline",
           [("LayerNorm", "norm", 0.45), ("MHA\n+ dropout", "attn", 0.75), ("", "+", 0),
            ("LayerNorm", "norm", 0.45), ("GELU MLP (4×)\n+ dropout", "mlp", 0.75), ("", "+", 0)],
           [(0, 2), (2, 5)], "learned absolute positions, biases on")
    column(1, "Llama 3 (2024)", "the dense template",
           [("RMSNorm", "norm", 0.45), ("GQA\n(RoPE on Q, K)", "attn", 0.75), ("", "+", 0),
            ("RMSNorm", "norm", 0.45), ("SwiGLU MLP", "mlp", 0.6), ("", "+", 0)],
           [(0, 2), (2, 5)], "no dropout, no biases")
    column(2, "Gemma 3 / OLMo 2 (2025)", "stability fixes",
           [("RMSNorm (pre)", "norm", 0.45), ("attention, QK-norm\nlocal 5 : 1 global", "attn", 0.8),
            ("RMSNorm (post)", "norm", 0.45), ("", "+", 0),
            ("RMSNorm (pre)", "norm", 0.45), ("GeGLU MLP", "mlp", 0.6), ("RMSNorm (post)", "norm", 0.45),
            ("", "+", 0)],
           [(0, 3), (3, 7)], "OLMo 2 drops the pre-norms (norm inside the residual)")
    column(3, "DeepSeek V3 template (2025)", "capacity without compute",
           [("RMSNorm", "norm", 0.45), ("MLA (latent K/V)", "attn", 0.6), ("", "+", 0),
            ("RMSNorm", "norm", 0.45), ("router → top-8 of 256\nexperts + 1 shared", "router", 0.8), ("", "+", 0)],
           [(0, 2), (2, 5)], "first 3 layers dense; MTP head on top")
    column(4, "Qwen3-Next / 3.5 (2025–26)", "linear-attention hybrid",
           [("RMSNorm", "norm", 0.45), ("3× Gated DeltaNet,\nthen 1× gated attn\n(QK-norm)", "ssm", 1.05),
            ("", "+", 0), ("RMSNorm", "norm", 0.45), ("MoE: 512 experts\n+ shared", "router", 0.7),
            ("", "+", 0)],
           [(0, 2), (2, 5)], "only 1 in 4 layers keeps a KV cache")
    _save_to(fig, path)


def _save_to(fig, path):
    plt = _plt()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(models=MODELS):
    """The comparison table in llm-architecture-design.md, grouped by family, oldest first inside a family."""
    def p(x):
        return f"{x:g}"
    rows = ["| Model | Date | Params (active) | Layers | Global attn | Cheap mixer : global | QK-norm | Norm | "
            "Positions | Channel mixer | MTP | KV/token |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for fk, fname in FAMILIES.items():
        rows.append(f"| **{fk}. {fname}** | | | | | | | | | | | |")
        for m in sorted((m for m in models if family(m) == fk), key=lambda m: m.date):
            params = p(m.total_b) + "B" + ("" if not m.moe else f" ({p(m.active_b)}B)" if m.active_b else " (n/s)")
            mix = "–" if not m.hybrid else (f"{m.hybrid} {m.n_local}:{m.n_global}" if m.n_local is not None
                                            else m.hybrid)
            norm = (m.norm or "n/s") + ("†" if m.name in NORM_FROM_CONFIG else "")
            if m.moe:
                ch = "MoE"
                if m.experts:
                    ch += f" {m.top_k}/{m.experts}"
                elif m.top_k:
                    ch += f" top-{m.top_k}"
                ch += {True: " +shared", False: " no shared", None: ""}[m.shared]
                ch += f" ({m.ffn})"
            else:
                ch = m.ffn
            kv = "n/s" if m.kv_kib is None else f"{m.kv_kib:g} KiB"
            rows.append(f"| {m.name} | {m.date} | {params} | {m.layers or 'n/s'} | {m.attn} | {mix} | "
                        f"{'yes' if m.qk_norm else '–'} | {norm} | {m.pos} | {ch} | {'yes' if m.mtp else '–'} | {kv} |")
    return "\n".join(rows)


# ---------------------------------------------------------------- main
def write_csv(path):
    keys = list(asdict(MODELS[0]).keys()) + ["family"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in MODELS:
            w.writerow({**asdict(m), "family": family(m)})


def main():
    os.makedirs(FIG, exist_ok=True)
    print(f"{len(MODELS)} models coded, {len(UNREAD)} listed but unread\n")
    fams = Counter(family(m) for m in MODELS)
    print("Families:")
    for k, v in FAMILIES.items():
        print(f"  {k} {v:<52} {fams[k]:>3}  " + ", ".join(m.name for m in MODELS if family(m) == k)[:110])
    print("\nAdoption by period (count / models in period):")
    print(f"  {'':<56}" + "".join(f"{p:>10}" for p in PERIODS))
    for f, row in adoption_table().items():
        print(f"  {f:<56}" + "".join(f"{c:>5}/{n:<4}" for c, n in row.values()))
    print("\nNorm placement:", Counter(m.norm for m in MODELS))
    print("Positional:", Counter(m.pos for m in MODELS))
    print("FFN:", Counter(m.ffn for m in MODELS))
    print("\nCheap-mixer : global-attention ratios:")
    for k, v in ratio_summary().items():
        print(f"  {k:<10}", ", ".join(f"{r:g} ({n})" for r, n in v))
    moe = [m for m in MODELS if m.moe]
    fr = sorted(m.active_frac for m in moe if m.active_frac is not None)
    print(f"\nMoE: {len(moe)} models, active fraction median {fr[len(fr) // 2]:.3f}, range {fr[0]:.3f}–{fr[-1]:.3f}")
    sh = Counter(m.shared for m in moe)
    print("  shared expert:", dict(sh))
    print("\nMedian KV cache per token (KiB) by attention:", kv_by_attention())
    write_csv(os.path.join(HERE, "llm_architectures.csv"))
    fig_adoption(os.path.join(FIG, "llm_adoption.png"))
    fig_kv(os.path.join(FIG, "llm_kv_cache.png"))
    fig_block_evolution(os.path.join(FIG, "llm_block_evolution.png"))
    print("\nwrote llm_architectures.csv and figures/llm_*.png")


if __name__ == "__main__":
    main()
