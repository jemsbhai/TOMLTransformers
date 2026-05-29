"""Attention TO accounting: standard vs FlashAttention, MHA / GQA / MQA.

The components here are phase-agnostic pieces; the front-ends (decoder, encoder,
encoder-decoder) assemble them into prefill, decode, bidirectional, and
cross-attention passes.

Standard vs FlashAttention
--------------------------
Both compute the same forward arithmetic: QK^T (s_q*s_kv*head_dim MACs per head),
a per-element softmax over the valid scores, and the score-by-value matmul
(another s_q*s_kv*head_dim MACs per head). They differ in two places only, and
those are the headline:

  - Memory: standard attention materializes the s_q x s_kv score matrix in
    off-chip memory (written after QK^T, read back for softmax and AV), which is
    O(s^2) traffic. FlashAttention tiles the computation and keeps scores in
    on-chip SRAM, so its score traffic to off-chip memory is zero (it pays
    O(s*d) for Q/K/V/O instead, accounted in the projections).
  - Dispatch: standard runs three kernels (QK^T, softmax, AV); Flash fuses them
    into one.

The number of score passes through off-chip memory for standard attention
(score_passes, default 2 = one write, one read) and the per-element softmax cost
are calibration targets.

Causal masking
--------------
A causal query block attends only to keys at or before it, so the valid
query-key pair count is the lower triangle: for an aligned block,
    pairs = s_q * s_kv - s_q*(s_q-1)/2
which gives s*(s+1)/2 for prefill (s_q = s_kv = s) and s_kv for a single decode
step (s_q = 1). Bidirectional (encoder) and cross-attention use the full
s_q * s_kv.
"""

from __future__ import annotations

from tomltransformers import to_costs as tc

from .common import FP16, Precision, TOBreakdown, linear, total
from .configs import TransformerConfig


def attention_pairs(s_q: int, s_kv: int, causal: bool) -> int:
    """Number of valid (query, key) score entries."""
    if not causal:
        return s_q * s_kv
    return s_q * s_kv - (s_q * (s_q - 1)) // 2


def qkv_projection(n_q: int, n_kv: int, cfg: TransformerConfig, *,
                   device: str, prec: Precision = FP16) -> TOBreakdown:
    """Project queries over n_q tokens and keys/values over n_kv tokens.

    Q has n_heads*head_dim width; K and V have kv_heads*head_dim (smaller under
    GQA/MQA). Three GEMMs.
    """
    d = cfg.d_model
    q = linear(d, cfg.n_heads * cfg.d_head, n_q, device=device, prec=prec)
    k = linear(d, cfg.kv_heads * cfg.d_head, n_kv, device=device, prec=prec)
    v = linear(d, cfg.kv_heads * cfg.d_head, n_kv, device=device, prec=prec)
    return total((q, k, v))


def output_projection(n_q: int, cfg: TransformerConfig, *,
                      device: str, prec: Precision = FP16) -> TOBreakdown:
    """Project the attention output (n_heads*head_dim) back to d_model."""
    return linear(cfg.n_heads * cfg.d_head, cfg.d_model, n_q, device=device, prec=prec)


def attention_core(s_q: int, s_kv: int, cfg: TransformerConfig, *,
                   device: str, prec: Precision = FP16, causal: bool,
                   kind: str = "flash", score_passes: int = 2) -> TOBreakdown:
    """Scores (QK^T), softmax, and AV. Standard vs Flash differ in memory/dispatch.

    Compute is identical for both kinds; only off-chip score traffic and the
    kernel-launch count change.
    """
    nh, hd = cfg.n_heads, cfg.d_head
    pairs = attention_pairs(s_q, s_kv, causal)

    to_mac = 2 * nh * pairs * hd * tc.mac(prec.compute)   # QK^T + AV
    to_nonlinear = nh * pairs * tc.op("softmax")
    score_elems = nh * pairs

    if kind == "standard":
        to_hbm = tc.mem_cost(score_passes * score_elems, tc.offchip_tier(device), prec.act)
        to_sram = tc.mem_cost(score_elems, "sram", prec.act)
        launches = 3   # QK^T, softmax, AV
    elif kind == "flash":
        to_hbm = 0.0                                       # scores never leave SRAM
        to_sram = tc.mem_cost(score_elems, "sram", prec.act)
        launches = 1   # fused kernel
    else:
        raise ValueError(f"unknown attention kind '{kind}' (use 'standard' or 'flash')")

    return TOBreakdown(to_mac=to_mac, to_nonlinear=to_nonlinear,
                       to_sram=to_sram, to_hbm=to_hbm, n_launches=launches)


def kv_cache_read(kv_len: int, cfg: TransformerConfig, *,
                  device: str, prec: Precision = FP16) -> TOBreakdown:
    """Read the cached K and V for a context of kv_len tokens from off-chip.

    2 * kv_len * kv_heads * head_dim words. This (with per-step weight reload) is
    what makes autoregressive decode bandwidth-bound. Folded into the attention
    kernel, so no extra launch.
    """
    words = 2 * kv_len * cfg.kv_heads * cfg.d_head
    return TOBreakdown(to_hbm=tc.mem_cost(words, tc.offchip_tier(device), prec.act),
                       n_launches=0)


def kv_cache_write(n_new: int, cfg: TransformerConfig, *,
                   device: str, prec: Precision = FP16) -> TOBreakdown:
    """Write n_new tokens' K and V into the off-chip cache (prefill writes s; decode 1)."""
    words = 2 * n_new * cfg.kv_heads * cfg.d_head
    return TOBreakdown(to_hbm=tc.mem_cost(words, tc.offchip_tier(device), prec.act),
                       n_launches=0)
