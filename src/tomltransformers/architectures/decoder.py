"""Decoder-only front-end: prefill and decode TO accounting.

Assembles the common blocks and attention pieces into a full causal model and
emits per-phase feature records (energy_model.FEATURES) plus the memory-compute
energy ratio (MCER).

Layer structure (pre-norm, as in GPT-2 / LLaMA): input norm -> self-attention
(QKV, scores+softmax+AV, output projection) -> post-attention norm -> FFN. Two
norms per layer. Residual adds are elementwise and negligible (< 1% of energy);
they are omitted and absorbed by calibration.

Prefill vs decode (the phase transition)
----------------------------------------
Prefill processes all s prompt tokens in one pass: each weight matrix is read
from off-chip once and reused across s tokens of compute, and the KV cache is
written. Arithmetic intensity is high, so prefill is compute-bound (MCER < 1).

Decode generates one token per step: every weight matrix is re-read from
off-chip for a single token of compute, and the cached K and V for the whole
context are read back. Arithmetic intensity is ~1/s of prefill, so decode is
memory-bound (MCER >> 1). The decode/prefill MCER ratio is therefore ~s.

The numbers this module prints are prior-weighted TO ratios, an architectural
sanity check, not calibrated energy. Calibrated MCER follows from fitting the
energy model to measured GPU energy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attention import (attention_core, kv_cache_read, kv_cache_write,
                        output_projection, qkv_projection)
from .common import (FP16, Precision, TOBreakdown, embedding_lookup, ffn, linear,
                     norm, total)
from .configs import TransformerConfig


@dataclass(frozen=True)
class PhaseResult:
    phase: str            # "prefill" | "decode_step" | "decode_total"
    seq_len: int
    n_tokens: int         # tokens produced/processed in this phase
    breakdown: TOBreakdown

    @property
    def mcer(self) -> float:
        return self.breakdown.mcer

    def record(self) -> dict[str, float]:
        return self.breakdown.as_record()


def _require_decoder(cfg: TransformerConfig) -> None:
    if cfg.arch != "decoder_only":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not decoder_only")


def _prefill_layer(cfg, s, *, device, prec, attn_kind) -> TOBreakdown:
    d = cfg.d_model
    return total((
        norm(s, d, norm_type=cfg.norm_type, device=device, prec=prec),
        qkv_projection(s, s, cfg, device=device, prec=prec),
        attention_core(s, s, cfg, device=device, prec=prec, causal=True, kind=attn_kind),
        kv_cache_write(s, cfg, device=device, prec=prec),
        output_projection(s, cfg, device=device, prec=prec),
        norm(s, d, norm_type=cfg.norm_type, device=device, prec=prec),
        ffn(s, d, cfg.d_ff, ffn_type=cfg.ffn_type, activation_kind=cfg.activation,
            device=device, prec=prec),
    ))


def _decode_layer(cfg, context_len, *, device, prec, attn_kind) -> TOBreakdown:
    d = cfg.d_model
    cache_prev = max(context_len - 1, 0)   # cached K,V before this step
    return total((
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        qkv_projection(1, 1, cfg, device=device, prec=prec),
        kv_cache_read(cache_prev, cfg, device=device, prec=prec),
        attention_core(1, context_len, cfg, device=device, prec=prec, causal=True, kind=attn_kind),
        kv_cache_write(1, cfg, device=device, prec=prec),
        output_projection(1, cfg, device=device, prec=prec),
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        ffn(1, d, cfg.d_ff, ffn_type=cfg.ffn_type, activation_kind=cfg.activation,
            device=device, prec=prec),
    ))


def prefill(cfg: TransformerConfig, seq_len: int, *, device: str = "rtx4090",
            prec: Precision = FP16, attn_kind: str = "flash") -> PhaseResult:
    """One prefill pass over seq_len prompt tokens; lm_head on the last token."""
    _require_decoder(cfg)
    d = cfg.d_model
    parts = (
        embedding_lookup(seq_len, d, device=device, prec=prec),
        _prefill_layer(cfg, seq_len, device=device, prec=prec, attn_kind=attn_kind)
        .scaled(cfg.n_layers),
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),   # final norm (last token)
        linear(d, cfg.vocab_size, 1, device=device, prec=prec),          # lm_head (last token)
    )
    return PhaseResult("prefill", seq_len, seq_len, total(parts))


def decode_step(cfg: TransformerConfig, context_len: int, *, device: str = "rtx4090",
                prec: Precision = FP16, attn_kind: str = "flash") -> PhaseResult:
    """A single decode step generating one token at the given context length."""
    _require_decoder(cfg)
    if context_len < 1:
        raise ValueError("context_len must be >= 1")
    d = cfg.d_model
    parts = (
        embedding_lookup(1, d, device=device, prec=prec),
        _decode_layer(cfg, context_len, device=device, prec=prec, attn_kind=attn_kind)
        .scaled(cfg.n_layers),
        norm(1, d, norm_type=cfg.norm_type, device=device, prec=prec),
        linear(d, cfg.vocab_size, 1, device=device, prec=prec),          # lm_head
    )
    return PhaseResult("decode_step", context_len, 1, total(parts))


def decode_total(cfg: TransformerConfig, prefill_len: int, n_generate: int, *,
                 device: str = "rtx4090", prec: Precision = FP16,
                 attn_kind: str = "flash") -> PhaseResult:
    """Total cost of generating n_generate tokens after a prefill_len prompt.

    Context grows by one each step; KV-cache reads grow accordingly.
    """
    _require_decoder(cfg)
    acc = TOBreakdown()
    for t in range(n_generate):
        ctx = prefill_len + t + 1
        acc = acc + decode_step(cfg, ctx, device=device, prec=prec,
                                attn_kind=attn_kind).breakdown
    return PhaseResult("decode_total", prefill_len + n_generate, n_generate, acc)


# ------------------------------------------------------------------------------
# Illustrative report (prior-weighted TO ratios, pre-calibration)
# ------------------------------------------------------------------------------
def _report() -> None:
    from . import configs as cf

    models = [cf.GPT2, cf.LLAMA_7B, cf.MISTRAL_7B]
    print("\nPrefill vs decode MCER (prior-weighted TO ratios; device=rtx4090, flash)")
    print("=" * 78)
    print(f"{'model':12s} {'seq':>6s} {'prefill_MCER':>13s} {'decode_MCER':>12s} {'decode/prefill':>15s}")
    for s in (512, 2048):
        for cfg in models:
            pf = prefill(cfg, s).mcer
            ds = decode_step(cfg, s).mcer
            print(f"{cfg.name:12s} {s:>6d} {pf:>13.4f} {ds:>12.2f} {ds / pf:>15.0f}")

    print("\nStandard vs FlashAttention, prefill MCER at long context (LLaMA-7B)")
    print("-" * 78)
    for s in (2048, 8192, 16384):
        f = prefill(cf.LLAMA_7B, s, attn_kind="flash").mcer
        st = prefill(cf.LLAMA_7B, s, attn_kind="standard").mcer
        print(f"  s={s:>6d}   flash MCER={f:>8.4f}   standard MCER={st:>8.4f}   ratio={st / f:>6.2f}")


if __name__ == "__main__":
    _report()
