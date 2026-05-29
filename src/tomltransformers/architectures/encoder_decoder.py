"""Encoder-decoder front-end: T5 / BART TO accounting.

The most structurally involved class: it combines all three attention patterns.

  - Encoder: bidirectional self-attention over the source, runs ONCE, no cache
    (reuses the encoder-only layer).
  - Decoder self-attention: causal, over the target, with its own KV cache that
    grows as generation proceeds (the decoder-only prefill/decode logic).
  - Cross-attention: queries come from the decoder, but keys and values are
    projected ONCE from the final encoder output and cached for the whole
    generation. So during decode, cross-attention performs no K/V projection and
    no cache write -- only a cached read whose size scales with the SOURCE length
    (constant across decode steps), unlike self-attention whose cache grows with
    the TARGET length.

Decoder layers therefore have three norms (self-attn, cross-attn, FFN). T5
(RMSNorm, relative-position bias, ReLU, 12+12 layers) vs BART (LayerNorm, learned
positions, GELU, 6+6) fall out of the config.

Phases exposed separately so the cost attribution across the three attention
types is explicit: encode (once), decoder_prefill (once; projects and caches the
cross-attention K/V), and decode_step (per generated token). All return a
TOBreakdown (use .mcer, .as_record()).
"""

from __future__ import annotations

from .attention import (attention_core, kv_cache_read, kv_cache_write,
                        output_projection, qkv_projection)
from .common import FP16, Precision, TOBreakdown, embedding_lookup, ffn, linear, norm, total
from .configs import TransformerConfig
from .encoder import encoder_layer


def _require_enc_dec(cfg: TransformerConfig) -> None:
    if cfg.arch != "encoder_decoder":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not encoder_decoder")


# --- cross-attention projections (Q from decoder; K,V from encoder output) ---
def _q_proj(n_q: int, cfg: TransformerConfig, *, device: str, prec: Precision) -> TOBreakdown:
    return linear(cfg.d_model, cfg.n_heads * cfg.d_head, n_q, device=device, prec=prec)


def _kv_proj(n_kv: int, cfg: TransformerConfig, *, device: str, prec: Precision) -> TOBreakdown:
    k = linear(cfg.d_model, cfg.kv_heads * cfg.d_head, n_kv, device=device, prec=prec)
    v = linear(cfg.d_model, cfg.kv_heads * cfg.d_head, n_kv, device=device, prec=prec)
    return total((k, v))


# --- decoder layers -----------------------------------------------------------
def _decoder_layer_prefill(cfg, src_len, tgt_len, *, device, prec, attn_kind) -> TOBreakdown:
    """One decoder layer processing tgt_len target tokens; projects+caches cross K/V."""
    d = cfg.d_model
    self_block = total((
        norm(tgt_len, d, norm_type=cfg.norm_type, device=device, prec=prec),
        qkv_projection(tgt_len, tgt_len, cfg, device=device, prec=prec),
        attention_core(tgt_len, tgt_len, cfg, device=device, prec=prec, causal=True, kind=attn_kind),
        kv_cache_write(tgt_len, cfg, device=device, prec=prec),              # self-KV cache
        output_projection(tgt_len, cfg, device=device, prec=prec),
    ))
    cross_block = total((
        norm(tgt_len, d, norm_type=cfg.norm_type, device=device, prec=prec),
        _q_proj(tgt_len, cfg, device=device, prec=prec),                     # Q from decoder
        _kv_proj(src_len, cfg, device=device, prec=prec),                    # K,V from encoder (once)
        attention_core(tgt_len, src_len, cfg, device=device, prec=prec, causal=False, kind=attn_kind),
        kv_cache_write(src_len, cfg, device=device, prec=prec),              # cache cross-K,V (once)
        output_projection(tgt_len, cfg, device=device, prec=prec),
    ))
    ffn_block = total((
        norm(tgt_len, d, norm_type=cfg.norm_type, device=device, prec=prec),
        ffn(tgt_len, d, cfg.d_ff, ffn_type=cfg.ffn_type, activation_kind=cfg.activation,
            device=device, prec=prec),
    ))
    return total((self_block, cross_block, ffn_block))


def _decoder_layer_decode(cfg, src_len, self_ctx, *, device, prec, attn_kind) -> TOBreakdown:
    """One decoder layer for a single decode step (self-KV grows, cross-KV cached)."""
    d = cfg.d_model
    self_block = total((
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        qkv_projection(1, 1, cfg, device=device, prec=prec),
        kv_cache_read(max(self_ctx - 1, 0), cfg, device=device, prec=prec),  # self-KV (grows)
        attention_core(1, self_ctx, cfg, device=device, prec=prec, causal=True, kind=attn_kind),
        kv_cache_write(1, cfg, device=device, prec=prec),
        output_projection(1, cfg, device=device, prec=prec),
    ))
    cross_block = total((
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        _q_proj(1, cfg, device=device, prec=prec),                           # Q only; no K/V projection
        kv_cache_read(src_len, cfg, device=device, prec=prec),               # cross-KV cache (constant)
        attention_core(1, src_len, cfg, device=device, prec=prec, causal=False, kind=attn_kind),
        output_projection(1, cfg, device=device, prec=prec),
    ))
    ffn_block = total((
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        ffn(1, d, cfg.d_ff, ffn_type=cfg.ffn_type, activation_kind=cfg.activation,
            device=device, prec=prec),
    ))
    return total((self_block, cross_block, ffn_block))


# --- phases -------------------------------------------------------------------
def encode(cfg: TransformerConfig, src_len: int, *, device: str = "rtx4090",
           prec: Precision = FP16, attn_kind: str = "flash") -> TOBreakdown:
    """Encoder pass over the source (bidirectional, once, no cache, no head)."""
    _require_enc_dec(cfg)
    d = cfg.d_model
    return total((
        embedding_lookup(src_len, d, device=device, prec=prec),
        encoder_layer(cfg, src_len, device=device, prec=prec, attn_kind=attn_kind)
        .scaled(cfg.n_encoder_layers),
        norm(src_len, d, norm_type=cfg.norm_type, device=device, prec=prec),  # encoder final norm
    ))


def decoder_prefill(cfg: TransformerConfig, src_len: int, tgt_len: int = 1, *,
                    device: str = "rtx4090", prec: Precision = FP16,
                    attn_kind: str = "flash") -> TOBreakdown:
    """Decoder over tgt_len target tokens; projects and caches the cross-attention K/V."""
    _require_enc_dec(cfg)
    d = cfg.d_model
    return total((
        embedding_lookup(tgt_len, d, device=device, prec=prec),
        _decoder_layer_prefill(cfg, src_len, tgt_len, device=device, prec=prec, attn_kind=attn_kind)
        .scaled(cfg.n_decoder_layers),
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),   # final norm (last token)
        linear(d, cfg.vocab_size, 1, device=device, prec=prec),          # lm_head
    ))


def decode_step(cfg: TransformerConfig, src_len: int, tgt_ctx: int, *,
                device: str = "rtx4090", prec: Precision = FP16,
                attn_kind: str = "flash") -> TOBreakdown:
    """A single decode step: self-attn over tgt_ctx target tokens, cross-attn over src_len."""
    _require_enc_dec(cfg)
    if tgt_ctx < 1:
        raise ValueError("tgt_ctx must be >= 1")
    d = cfg.d_model
    return total((
        embedding_lookup(1, d, device=device, prec=prec),
        _decoder_layer_decode(cfg, src_len, tgt_ctx, device=device, prec=prec, attn_kind=attn_kind)
        .scaled(cfg.n_decoder_layers),
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        linear(d, cfg.vocab_size, 1, device=device, prec=prec),
    ))


def decode_total(cfg: TransformerConfig, src_len: int, tgt_prefill_len: int, n_generate: int, *,
                 device: str = "rtx4090", prec: Precision = FP16,
                 attn_kind: str = "flash") -> TOBreakdown:
    """Total cost of generating n_generate target tokens (self-KV grows; cross-KV constant)."""
    _require_enc_dec(cfg)
    acc = TOBreakdown()
    for t in range(n_generate):
        tgt_ctx = tgt_prefill_len + t + 1
        acc = acc + decode_step(cfg, src_len, tgt_ctx, device=device, prec=prec, attn_kind=attn_kind)
    return acc


# ------------------------------------------------------------------------------
# Illustrative report (prior-weighted TO ratios, pre-calibration; dev tool)
# ------------------------------------------------------------------------------
def _report() -> None:
    from . import configs as cf

    print("\nEncoder-decoder MCER by phase (prior-weighted TO ratios; rtx4090, flash)")
    print("=" * 78)
    print(f"{'model':10s} {'src':>5s} {'tgt_ctx':>8s} {'encode_MCER':>12s} {'decode_MCER':>12s}")
    for cfg in (cf.T5_BASE, cf.BART_BASE):
        for src in (256, 1024):
            enc = encode(cfg, src).mcer
            ds = decode_step(cfg, src, 128).mcer
            print(f"{cfg.name:10s} {src:>5d} {128:>8d} {enc:>12.4f} {ds:>12.2f}")

    print("\nCross-attention is cached: decode-step off-chip traffic vs src and tgt (T5-base)")
    print("-" * 78)
    base = decode_step(cf.T5_BASE, 512, 128).to_hbm
    for src, tgt in ((512, 128), (4096, 128), (512, 4096)):
        hbm = decode_step(cf.T5_BASE, src, tgt).to_hbm
        print(f"  src={src:>5d} tgt_ctx={tgt:>5d}   decode to_hbm (rel) = {hbm / base:>6.2f}")


if __name__ == "__main__":
    _report()
