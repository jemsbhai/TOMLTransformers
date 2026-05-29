"""Encoder-only front-end: BERT / ViT TO accounting.

A single bidirectional forward pass. There is no KV cache and no prefill/decode
split: all s tokens are processed in parallel, and every weight is loaded once
and amortized over the sequence (so, like decoder prefill, this is compute-bound
at batch 1 for typical sequence lengths).

Two differences from a causal decoder layer:
  - Attention is bidirectional, so the full s x s score matrix is computed (no
    causal lower-triangle halving) -- roughly twice the attention work of a
    causal prefill of the same length.
  - No KV cache is written (nothing is reused across steps; there are no steps).

Embedding depends on modality: BERT gathers token embeddings (memory, no MACs);
ViT projects image patches (a GEMM, PATCH_DIM -> d_model over num_patches), and
its sequence length is fixed by the architecture (num_patches + 1 for the class
token).
"""

from __future__ import annotations

from .attention import attention_core, output_projection, qkv_projection
from .common import (FP16, Precision, TOBreakdown, embedding_lookup, ffn, linear,
                     norm, total)
from .configs import TransformerConfig

PATCH_DIM = 3 * 16 * 16  # 16x16 RGB patch flattened


def _require_encoder(cfg: TransformerConfig) -> None:
    if cfg.arch != "encoder_only":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not encoder_only")


def encoder_layer(cfg: TransformerConfig, s: int, *, device: str,
                  prec: Precision = FP16, attn_kind: str = "flash") -> TOBreakdown:
    """One bidirectional encoder layer over s tokens (two norms, no KV cache)."""
    d = cfg.d_model
    return total((
        norm(s, d, norm_type=cfg.norm_type, device=device, prec=prec),
        qkv_projection(s, s, cfg, device=device, prec=prec),
        attention_core(s, s, cfg, device=device, prec=prec, causal=False, kind=attn_kind),
        output_projection(s, cfg, device=device, prec=prec),
        norm(s, d, norm_type=cfg.norm_type, device=device, prec=prec),
        ffn(s, d, cfg.d_ff, ffn_type=cfg.ffn_type, activation_kind=cfg.activation,
            device=device, prec=prec),
    ))


def _embedding(cfg: TransformerConfig, seq_len: int, *, device: str,
               prec: Precision = FP16) -> TOBreakdown:
    if cfg.is_vision:
        # Patch projection: PATCH_DIM -> d_model over num_patches patches (a GEMM).
        return linear(PATCH_DIM, cfg.d_model, cfg.num_patches, device=device, prec=prec)
    # Token-embedding gather: no MACs.
    return embedding_lookup(seq_len, cfg.d_model, device=device, prec=prec)


def sequence_length(cfg: TransformerConfig, seq_len: int | None) -> int:
    """Resolve the sequence length: fixed (num_patches + 1) for vision, else given."""
    if cfg.is_vision:
        return cfg.num_patches + 1   # patches + class token
    if seq_len is None:
        raise ValueError(f"{cfg.name}: seq_len is required for a text encoder")
    return seq_len


def encode(cfg: TransformerConfig, seq_len: int | None = None, *, device: str = "rtx4090",
           prec: Precision = FP16, attn_kind: str = "flash",
           include_head: bool = True) -> TOBreakdown:
    """Full encoder forward pass; optional pooled-token classification head.

    Returns a TOBreakdown (use .mcer, .as_record(), .compute, .memory).
    """
    _require_encoder(cfg)
    s = sequence_length(cfg, seq_len)
    d = cfg.d_model
    parts = [
        _embedding(cfg, s, device=device, prec=prec),
        encoder_layer(cfg, s, device=device, prec=prec, attn_kind=attn_kind)
        .scaled(cfg.n_layers),
    ]
    if include_head:
        # Final norm and a classifier/pooler on the single pooled (class) token.
        parts.append(norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec))
        head_out = cfg.num_classes if cfg.num_classes > 0 else d
        parts.append(linear(d, head_out, 1, device=device, prec=prec))
    return total(parts)


# ------------------------------------------------------------------------------
# Illustrative report (prior-weighted TO ratios, pre-calibration; dev tool)
# ------------------------------------------------------------------------------
def _report() -> None:
    from . import configs as cf

    print("\nEncoder-only single-pass MCER (prior-weighted TO ratios; rtx4090, flash)")
    print("=" * 78)
    print(f"{'model':12s} {'seq':>6s} {'MCER':>8s} {'compute_frac':>13s}")
    cases = [(cf.BERT_BASE, 128), (cf.BERT_BASE, 512), (cf.BERT_LARGE, 512), (cf.VIT_B16, None)]
    for cfg, s in cases:
        b = encode(cfg, s)
        s_eff = sequence_length(cfg, s)
        print(f"{cfg.name:12s} {s_eff:>6d} {b.mcer:>8.4f} {b.compute / b.total:>13.3f}")

    print("\nBidirectional attention: flash vs standard, BERT-base (illustrative > 512)")
    print("-" * 78)
    for s in (512, 2048, 8192):
        f = encode(cf.BERT_BASE, s, attn_kind="flash").mcer
        st = encode(cf.BERT_BASE, s, attn_kind="standard").mcer
        print(f"  s={s:>6d}   flash MCER={f:>8.4f}   standard MCER={st:>8.4f}   ratio={st / f:>6.2f}")


if __name__ == "__main__":
    _report()
