"""Encoder-only runnable workloads (shape-faithful to architectures/encoder.py).

Builds a real torch module whose per-layer structure matches what the encoder
TO-counting front-end counts, so measured energy and predicted TO counts describe
the same computation:

  pre-norm -> QKV projection -> BIDIRECTIONAL attention core (full s x s scores,
  no causal mask) -> output projection -> post-attn norm -> FFN, x n_layers,
  then an optional final norm + classifier on a single pooled token.

Two differences from the decoder workload, both following encoder.py:
  - Attention is bidirectional: is_causal=False always, and 'eager' forces the
    math kernel that materializes the full s x s score matrix (the standard-
    attention accounting), vs 'flash' which tiles in SRAM.
  - There is no KV cache and no prefill/decode split: a single forward pass over
    all s tokens (phase="encode").

Embedding depends on modality (encoder.py._embedding):
  - text (BERT): token-embedding gather via nn.Embedding (memory, no MACs);
    sequence length is the given seq_len.
  - vision (ViT): patch projection PATCH_DIM -> d_model over num_patches via a
    GEMM (nn.Linear); sequence length is fixed at num_patches + 1 (class token),
    so seq_len is ignored for vision.

Weights are random-init by default (energy depends on op shapes / data movement,
not values). A pretrained path is provided for spot checks (BERT/ViT via HF).

PARITY NOTE 2026-07-24 (pre-representativeness-run): the pretrained path loads
with attn_implementation="sdpa" requested (graceful fallback), matching the
random-init SDPA path's kernel family. AutoModel (bare encoder, no MLM head) is
already near structural parity with the random path: BERT's pooler (dense d->d
+ tanh on [CLS]) is the same magnitude as our pooled head (d->d on one token),
both negligible against the layer stack; the small embedding differences
(position/type embeddings, embedding LayerNorm) are adds, not GEMMs. Recorded
in the representativeness operationalization note.
"""

from __future__ import annotations

from typing import Optional

from .protocol import CallableWorkload, WorkloadSpec
from ..architectures.configs import TransformerConfig, get as get_config

PATCH_DIM = 3 * 16 * 16  # 16x16 RGB patch flattened (matches encoder.py)

_DTYPE = {"fp16": "float16", "fp32": "float32"}


def _torch():
    import torch
    return torch


def sequence_length(cfg: TransformerConfig, seq_len: Optional[int]) -> int:
    """Resolve the sequence length: fixed (num_patches + 1) for vision, else given.

    Mirrors encoder.py.sequence_length so the runnable model and the TO count
    agree on s.
    """
    if cfg.is_vision:
        return cfg.num_patches + 1   # patches + class token
    if seq_len is None:
        raise ValueError(f"{cfg.name}: seq_len is required for a text encoder")
    return seq_len


class _EncoderModel:
    """Lazily-built torch.nn module mirroring the counted encoder structure.

    Like _DecoderModel, kept as a thin wrapper so the package imports without
    torch on machines that lack it; torch is imported only at construction.
    """

    def __init__(self, cfg: TransformerConfig, dtype_str: str, device: str):
        torch = _torch()
        import torch.nn as nn

        self.cfg = cfg
        torch_dtype_name = _DTYPE.get(dtype_str, dtype_str)
        if not hasattr(torch, torch_dtype_name):
            raise ValueError(
                f"dtype '{dtype_str}' is neither a precision key {list(_DTYPE)} "
                f"nor a torch dtype name"
            )
        self.dtype = getattr(torch, torch_dtype_name)
        self.device = device
        d = cfg.d_model
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.kv_heads
        self.head_dim = cfg.d_head
        self.is_gated = cfg.is_gated
        self.is_vision = cfg.is_vision
        self.num_classes = cfg.num_classes

        Norm = nn.LayerNorm  # RMSNorm vs LayerNorm differ negligibly in energy.

        def lin(i, o):
            return nn.Linear(i, o, bias=False)

        # --- embedding (modality-dependent) ---
        if cfg.is_vision:
            # Patch projection: PATCH_DIM -> d_model (a GEMM over num_patches).
            self.patch_proj = lin(PATCH_DIM, d)
            self.embed = None
        else:
            # Token-embedding gather (no MACs).
            self.patch_proj = None
            self.embed = nn.Embedding(cfg.vocab_size, d)

        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            layer = nn.ModuleDict()
            layer["norm1"] = Norm(d)
            q_dim = self.n_heads * self.head_dim
            kv_dim = self.n_kv_heads * self.head_dim
            layer["qkv"] = lin(d, q_dim + 2 * kv_dim)
            layer["out"] = lin(q_dim, d)
            layer["norm2"] = Norm(d)
            if cfg.is_gated:
                layer["gate"] = lin(d, cfg.d_ff)
                layer["up"] = lin(d, cfg.d_ff)
                layer["down"] = lin(cfg.d_ff, d)
            else:
                layer["up"] = lin(d, cfg.d_ff)
                layer["down"] = lin(cfg.d_ff, d)
            self.layers.append(layer)

        # Optional classification head on the pooled (class/[CLS]) token.
        self.head_norm = Norm(d)
        head_out = cfg.num_classes if cfg.num_classes > 0 else d
        self.head = lin(d, head_out)

        # Register submodules so .to(device, dtype) moves everything.
        self.module = nn.Module()
        if self.embed is not None:
            self.module.embed = self.embed
        if self.patch_proj is not None:
            self.module.patch_proj = self.patch_proj
        self.module.layers = self.layers
        self.module.head_norm = self.head_norm
        self.module.head = self.head
        self.module = self.module.to(device=device, dtype=self.dtype).eval()

        self._act = self._activation_fn(cfg.activation)

    @staticmethod
    def _activation_fn(kind: str):
        torch = _torch()
        import torch.nn.functional as F
        return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}.get(kind, F.gelu)

    def _layer(self, layer, x, *, attn_kind: str):
        """One bidirectional encoder layer (non-causal attention)."""
        torch = _torch()
        import torch.nn.functional as F
        B, S, _ = x.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q_dim = nh * hd
        kv_dim = nkv * hd

        h = layer["norm1"](x)
        qkv = layer["qkv"](h)
        q = qkv[..., :q_dim].view(B, S, nh, hd).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].view(B, S, nkv, hd).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].view(B, S, nkv, hd).transpose(1, 2)
        if nkv != nh:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Bidirectional: no causal mask. 'eager' -> MATH kernel materializes the
        # full s x s scores (standard-attention accounting); 'flash' tiles.
        if attn_kind == "eager":
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        with torch.nn.attention.sdpa_kernel(backends):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        a = a.transpose(1, 2).reshape(B, S, q_dim)
        x = x + layer["out"](a)
        h2 = layer["norm2"](x)
        if self.is_gated:
            ff = layer["down"](self._act(layer["gate"](h2)) * layer["up"](h2))
        else:
            ff = layer["down"](self._act(layer["up"](h2)))
        x = x + ff
        return x

    def forward(self, inp, *, attn_kind: str, include_head: bool = True):
        """Full encoder forward.

        inp is token ids (B, S) for text, or patch features (B, num_patches,
        PATCH_DIM) for vision. Returns the head logits on the pooled token if
        include_head, else the full hidden states.
        """
        if self.is_vision:
            x = self.patch_proj(inp)            # (B, num_patches, d)
        else:
            x = self.embed(inp)                 # (B, S, d)
        for layer in self.layers:
            x = self._layer(layer, x, attn_kind=attn_kind)
        if not include_head:
            return x
        pooled = self.head_norm(x[:, :1, :])    # single pooled/class token
        return self.head(pooled)


def build_encoder_workload(
    model: str | TransformerConfig,
    *,
    seq_len: Optional[int] = None,
    precision: str = "fp16",
    weights: str = "random",          # "random" | "pretrained"
    attn_kind: str = "flash",
    inner_iters: int = 1,
    batch_size: int = 1,
    device_index: int = 0,
    include_head: bool = True,
    pretrained_id: Optional[str] = None,
) -> CallableWorkload:
    """Construct a runnable encoder-only workload (single bidirectional pass).

    encode: one forward over s tokens (no generation, no KV cache), optional
            classifier on the pooled token. Looped `inner_iters` times per
            measured execution to clear the window-length floor (use
            measure_until_floor to size it).

    For vision configs (is_vision), seq_len is ignored: s = num_patches + 1.
    For text encoders, seq_len is required.
    """
    if precision not in _DTYPE:
        raise ValueError(f"precision must be one of {list(_DTYPE)}, got {precision}")

    cfg = model if isinstance(model, TransformerConfig) else get_config(model)
    if cfg.arch != "encoder_only":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not encoder_only")

    s = sequence_length(cfg, seq_len)

    torch = _torch()
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    spec = WorkloadSpec(
        model_name=cfg.name, arch=cfg.arch, phase="encode", seq_len=s,
        precision=precision, weights=weights, attn_kind=attn_kind,
        inner_iters=inner_iters, batch_size=batch_size,
        extra={"is_vision": cfg.is_vision, "include_head": include_head},
    )

    if weights == "pretrained":
        return _build_pretrained_encode(
            cfg, spec, s, precision, device, batch_size, inner_iters, pretrained_id)

    # random-init, shape-faithful
    em = _EncoderModel(cfg, precision, device)
    if cfg.is_vision:
        inp = torch.randn(batch_size, s, PATCH_DIM, device=device, dtype=em.dtype)
    else:
        inp = torch.randint(0, max(cfg.vocab_size, 1), (batch_size, s), device=device)

    @torch.no_grad()
    def run():
        for _ in range(inner_iters):
            em.forward(inp, attn_kind=attn_kind, include_head=include_head)

    def free():
        nonlocal em
        del em
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _load_pretrained_encoder(hf_id: str, dtype, device: str):
    """Load a pretrained bare encoder with SDPA requested (parity with the
    random-init SDPA path); graceful fallback on older transformers. The
    implementation that actually loaded is readable at
    model.config._attn_implementation."""
    from transformers import AutoModel
    try:
        m = AutoModel.from_pretrained(hf_id, torch_dtype=dtype,
                                      attn_implementation="sdpa")
    except (TypeError, ValueError):
        m = AutoModel.from_pretrained(hf_id, torch_dtype=dtype)
    return m.to(device).eval()


def _build_pretrained_encode(cfg, spec, s, precision, device, batch_size,
                             inner_iters, pretrained_id):
    """Load a real pretrained encoder for spot checks (downloads weights).

    Text encoders (BERT/DistilBERT) take token ids; vision encoders (ViT) take
    pixel values (B, 3, 224, 224).
    """
    torch = _torch()
    hf_id = pretrained_id or _default_hf_id(cfg.name)
    dtype = getattr(torch, _DTYPE[precision])
    model_obj = _load_pretrained_encoder(hf_id, dtype, device)

    if cfg.is_vision:
        inp = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                model_obj(pixel_values=inp)
    else:
        vocab = getattr(model_obj.config, "vocab_size", max(cfg.vocab_size, 1))
        inp = torch.randint(0, vocab, (batch_size, s), device=device)

        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                model_obj(inp)

    def free():
        nonlocal model_obj
        del model_obj
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _default_hf_id(name: str) -> str:
    return {
        "BERT-base": "bert-base-uncased",
        "BERT-large": "bert-large-uncased",
        "DistilBERT": "distilbert-base-uncased",
        "ViT-B/16": "google/vit-base-patch16-224",
        "ViT-L/16": "google/vit-large-patch16-224",
    }.get(name, name)


def _empty_cache() -> None:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
