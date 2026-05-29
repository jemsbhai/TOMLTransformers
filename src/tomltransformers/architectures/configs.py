"""Model zoo: architecture configurations for the three transformer classes.

A single TransformerConfig covers decoder-only, encoder-only, and
encoder-decoder models, because they differ in ways the TO accounting must
respect (KV cache vs none, causal vs bidirectional attention, presence of
cross-attention, gated vs standard FFN, RMSNorm vs LayerNorm). The architecture
front-ends (decoder.py, encoder.py, encoder_decoder.py) consume these.

Parameter values are the standard published configurations. param_count is a
sanity/reporting feature (validated to known totals within a tolerance), not an
input to the energy model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

ARCHITECTURES = ("decoder_only", "encoder_only", "encoder_decoder")


@dataclass(frozen=True)
class TransformerConfig:
    name: str
    arch: str
    d_model: int
    d_ff: int
    n_heads: int
    vocab_size: int

    # Stack depth: n_layers for single-stack models; encoder/decoder split for enc-dec.
    n_layers: int = 0
    n_encoder_layers: int = 0
    n_decoder_layers: int = 0

    n_kv_heads: int = 0     # 0 -> MHA (= n_heads); < n_heads -> GQA; 1 -> MQA
    head_dim: int = 0       # 0 -> d_model // n_heads

    activation: str = "gelu"        # gelu | silu | relu
    norm_type: str = "layernorm"    # layernorm | rmsnorm
    ffn_type: str = "standard"      # standard (2 matmuls) | gated (3 matmuls, SwiGLU)
    tie_embeddings: bool = False
    max_position: int = 0           # learned positional-embedding rows (0 = RoPE/relative)

    # Encoder-only vision specifics.
    is_vision: bool = False
    num_patches: int = 0
    num_classes: int = 0

    # Encoder-decoder specifics.
    relative_position_bias: bool = False

    def __post_init__(self) -> None:
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}, got {self.arch!r}")
        if self.head_dim == 0 and self.d_model % self.n_heads != 0:
            raise ValueError(
                f"{self.name}: d_model {self.d_model} not divisible by n_heads "
                f"{self.n_heads} and no explicit head_dim"
            )
        if self.arch == "encoder_decoder":
            if self.n_encoder_layers <= 0 or self.n_decoder_layers <= 0:
                raise ValueError(f"{self.name}: enc-dec needs n_encoder_layers and n_decoder_layers")
        else:
            if self.n_layers <= 0:
                raise ValueError(f"{self.name}: {self.arch} needs n_layers > 0")

    # --- Resolved properties ---
    @property
    def d_head(self) -> int:
        return self.head_dim if self.head_dim > 0 else self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads > 0 else self.n_heads

    @property
    def is_gated(self) -> bool:
        return self.ffn_type == "gated"

    @property
    def is_causal(self) -> bool:
        return self.arch == "decoder_only"

    @property
    def has_kv_cache(self) -> bool:
        # Autoregressive decode (decoder-only and the decoder stack of enc-dec).
        return self.arch in ("decoder_only", "encoder_decoder")

    # --- Parameter count (sanity/reporting) ---
    def _attn_params(self) -> int:
        d, hd = self.d_model, self.d_head
        nh, kv = self.n_heads, self.kv_heads
        q = d * (nh * hd)
        k = d * (kv * hd)
        v = d * (kv * hd)
        o = (nh * hd) * d
        return q + k + v + o

    def _ffn_params(self) -> int:
        mult = 3 if self.is_gated else 2
        return mult * self.d_model * self.d_ff

    @property
    def param_count(self) -> int:
        d = self.d_model
        per_norm = 2 * d
        attn, ffn = self._attn_params(), self._ffn_params()

        if self.arch == "encoder_decoder":
            enc = self.n_encoder_layers * (attn + ffn + per_norm)
            # decoder layers add cross-attention and a third norm.
            dec = self.n_decoder_layers * (attn + self._attn_params() + ffn + 3 * d)
            layers = enc + dec
        else:
            layers = self.n_layers * (attn + ffn + per_norm)

        if self.is_vision:
            patch_dim = 3 * 16 * 16  # 16x16 RGB patches
            embed = patch_dim * d + self.num_classes * d
            pos = (self.num_patches + 1) * d
            head = 0
        else:
            embed = self.vocab_size * d
            pos = self.max_position * d
            head = 0 if (self.tie_embeddings or self.arch == "encoder_only") else self.vocab_size * d

        return layers + embed + pos + head


# ------------------------------------------------------------------------------
# Decoder-only
# ------------------------------------------------------------------------------
LLAMA_7B = TransformerConfig(
    name="LLaMA-7B", arch="decoder_only", n_layers=32, d_model=4096, d_ff=11008,
    n_heads=32, n_kv_heads=32, head_dim=128, vocab_size=32000,
    activation="silu", norm_type="rmsnorm", ffn_type="gated",
)
LLAMA3_8B = TransformerConfig(
    name="LLaMA-3-8B", arch="decoder_only", n_layers=32, d_model=4096, d_ff=14336,
    n_heads=32, n_kv_heads=8, head_dim=128, vocab_size=128256,
    activation="silu", norm_type="rmsnorm", ffn_type="gated",
)
MISTRAL_7B = TransformerConfig(
    name="Mistral-7B", arch="decoder_only", n_layers=32, d_model=4096, d_ff=14336,
    n_heads=32, n_kv_heads=8, head_dim=128, vocab_size=32000,
    activation="silu", norm_type="rmsnorm", ffn_type="gated",
)
GPT2 = TransformerConfig(
    name="GPT-2", arch="decoder_only", n_layers=12, d_model=768, d_ff=3072,
    n_heads=12, n_kv_heads=12, head_dim=64, vocab_size=50257,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)
DISTILGPT2 = TransformerConfig(
    name="DistilGPT2", arch="decoder_only", n_layers=6, d_model=768, d_ff=3072,
    n_heads=12, n_kv_heads=12, head_dim=64, vocab_size=50257,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)
GPT2_MEDIUM = TransformerConfig(
    name="GPT-2-medium", arch="decoder_only", n_layers=24, d_model=1024, d_ff=4096,
    n_heads=16, n_kv_heads=16, head_dim=64, vocab_size=50257,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)
GPT2_LARGE = TransformerConfig(
    name="GPT-2-large", arch="decoder_only", n_layers=36, d_model=1280, d_ff=5120,
    n_heads=20, n_kv_heads=20, head_dim=64, vocab_size=50257,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)
GPT2_XL = TransformerConfig(
    name="GPT-2-XL", arch="decoder_only", n_layers=48, d_model=1600, d_ff=6400,
    n_heads=25, n_kv_heads=25, head_dim=64, vocab_size=50257,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)

# ------------------------------------------------------------------------------
# Encoder-only
# ------------------------------------------------------------------------------
BERT_BASE = TransformerConfig(
    name="BERT-base", arch="encoder_only", n_layers=12, d_model=768, d_ff=3072,
    n_heads=12, head_dim=64, vocab_size=30522,
    activation="gelu", norm_type="layernorm", ffn_type="standard", max_position=512,
)
BERT_LARGE = TransformerConfig(
    name="BERT-large", arch="encoder_only", n_layers=24, d_model=1024, d_ff=4096,
    n_heads=16, head_dim=64, vocab_size=30522,
    activation="gelu", norm_type="layernorm", ffn_type="standard", max_position=512,
)
DISTILBERT = TransformerConfig(
    name="DistilBERT", arch="encoder_only", n_layers=6, d_model=768, d_ff=3072,
    n_heads=12, head_dim=64, vocab_size=30522,
    activation="gelu", norm_type="layernorm", ffn_type="standard", max_position=512,
)
VIT_B16 = TransformerConfig(
    name="ViT-B/16", arch="encoder_only", n_layers=12, d_model=768, d_ff=3072,
    n_heads=12, head_dim=64, vocab_size=0,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    is_vision=True, num_patches=196, num_classes=1000,
)
VIT_L16 = TransformerConfig(
    name="ViT-L/16", arch="encoder_only", n_layers=24, d_model=1024, d_ff=4096,
    n_heads=16, head_dim=64, vocab_size=0,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    is_vision=True, num_patches=196, num_classes=1000,
)

# ------------------------------------------------------------------------------
# Encoder-decoder
# ------------------------------------------------------------------------------
T5_BASE = TransformerConfig(
    name="T5-base", arch="encoder_decoder", n_encoder_layers=12, n_decoder_layers=12,
    d_model=768, d_ff=3072, n_heads=12, head_dim=64, vocab_size=32128,
    activation="relu", norm_type="rmsnorm", ffn_type="standard",
    tie_embeddings=True, relative_position_bias=True,
)
T5_SMALL = TransformerConfig(
    name="T5-small", arch="encoder_decoder", n_encoder_layers=6, n_decoder_layers=6,
    d_model=512, d_ff=2048, n_heads=8, head_dim=64, vocab_size=32128,
    activation="relu", norm_type="rmsnorm", ffn_type="standard",
    tie_embeddings=True, relative_position_bias=True,
)
BART_BASE = TransformerConfig(
    name="BART-base", arch="encoder_decoder", n_encoder_layers=6, n_decoder_layers=6,
    d_model=768, d_ff=3072, n_heads=12, head_dim=64, vocab_size=50265,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)
BART_LARGE = TransformerConfig(
    name="BART-large", arch="encoder_decoder", n_encoder_layers=12, n_decoder_layers=12,
    d_model=1024, d_ff=4096, n_heads=16, head_dim=64, vocab_size=50265,
    activation="gelu", norm_type="layernorm", ffn_type="standard",
    tie_embeddings=True, max_position=1024,
)


MODELS: Dict[str, TransformerConfig] = {
    m.name: m for m in (
        LLAMA_7B, LLAMA3_8B, MISTRAL_7B,
        GPT2, DISTILGPT2, GPT2_MEDIUM, GPT2_LARGE, GPT2_XL,
        BERT_BASE, BERT_LARGE, DISTILBERT, VIT_B16, VIT_L16,
        T5_BASE, T5_SMALL, BART_BASE, BART_LARGE,
    )
}


def get(name: str) -> TransformerConfig:
    try:
        return MODELS[name]
    except KeyError as exc:
        raise KeyError(f"unknown model '{name}'; known: {sorted(MODELS)}") from exc
