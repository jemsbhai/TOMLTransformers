"""Shared TO-counting building blocks for the architecture front-ends.

Every component returns a TOBreakdown: the granular transistor-operation counts,
weighted by the to_costs priors, split into the categories the energy model
fits (to_mac, to_nonlinear, to_sram, to_hbm) plus dispatch counts (n_launches,
n_fused_steps). Breakdowns add and scale, so a front-end builds a layer from
blocks, scales by the layer count, and reads off the feature record.

Memory-tiering convention (a modeling choice, and a calibration target):
  - Weights are streamed from off-chip memory (HBM/GDDR, resolved per device).
  - Activations and tiles are reused on-chip (SRAM).
  - The attention score matrix is off-chip for standard attention and on-chip
    for FlashAttention; that distinction lives in attention.py.

Precision convention:
  - compute  : MAC arithmetic precision (FP16 by default on GPU tensor cores).
  - weight   : storage precision of weights (drives off-chip word count).
  - act      : storage precision of activations (drives on-chip word count).
  W4A16 is Precision(compute='fp16', weight='int4', act='fp16');
  W8A8 is Precision(compute='int8', weight='int8', act='int8').

Kernel-launch convention (drives the dispatch term, M3+; a calibration target):
  - Each GEMM is one launch. Elementwise activations and gated multiplies are
    treated as fused (zero extra launches) by default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from tomltransformers import to_costs as tc


# ------------------------------------------------------------------------------
# Precision bundle
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Precision:
    compute: str = "fp16"
    weight: str = "fp16"
    act: str = "fp16"


FP16 = Precision()
W4A16 = Precision(compute="fp16", weight="int4", act="fp16")
W8A8 = Precision(compute="int8", weight="int8", act="int8")


# ------------------------------------------------------------------------------
# TO breakdown (the front-end's natural output; maps to energy-model features)
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class TOBreakdown:
    to_mac: float = 0.0
    to_nonlinear: float = 0.0
    to_sram: float = 0.0
    to_hbm: float = 0.0
    n_launches: float = 0.0
    n_fused_steps: float = 0.0

    def __add__(self, other: "TOBreakdown") -> "TOBreakdown":
        return TOBreakdown(*(getattr(self, f.name) + getattr(other, f.name)
                             for f in fields(self)))

    def scaled(self, k: float) -> "TOBreakdown":
        """Multiply every field by k (e.g. replicate a layer across the stack)."""
        return TOBreakdown(*(getattr(self, f.name) * k for f in fields(self)))

    def as_record(self) -> dict[str, float]:
        """Feature dict consumed by energy_model (keys match energy_model.FEATURES)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def compute(self) -> float:
        return self.to_mac + self.to_nonlinear

    @property
    def memory(self) -> float:
        return self.to_sram + self.to_hbm

    @property
    def total(self) -> float:
        return self.compute + self.memory

    @property
    def mcer(self) -> float:
        """Memory-compute energy ratio in TO units."""
        return self.memory / self.compute if self.compute > 0 else float("inf")


_ZERO = TOBreakdown()


def total(breakdowns) -> TOBreakdown:
    """Sum an iterable of TOBreakdown."""
    acc = _ZERO
    for b in breakdowns:
        acc = acc + b
    return acc


# ------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------
def linear(in_dim: int, out_dim: int, n_tokens: int, *, device: str,
           prec: Precision = FP16, launches: int = 1) -> TOBreakdown:
    """A dense projection y = W x over n_tokens.

    Compute: n_tokens * in_dim * out_dim MACs.
    Memory:  weights (in_dim*out_dim) from off-chip; activations (in+out per
             token) on-chip.
    """
    macs = n_tokens * in_dim * out_dim
    weight_words = in_dim * out_dim
    act_elems = n_tokens * (in_dim + out_dim)
    return TOBreakdown(
        to_mac=macs * tc.mac(prec.compute),
        to_hbm=tc.mem_cost(weight_words, tc.offchip_tier(device), prec.weight),
        to_sram=tc.mem_cost(act_elems, "sram", prec.act),
        n_launches=launches,
    )


def activation(n_elements: int, *, kind: str, prec: Precision = FP16,
               launches: int = 0) -> TOBreakdown:
    """Elementwise nonlinear activation (gelu/silu/relu), fused by default."""
    return TOBreakdown(
        to_nonlinear=n_elements * tc.op(kind),
        to_sram=tc.mem_cost(2 * n_elements, "sram", prec.act),
        n_launches=launches,
    )


def norm(n_tokens: int, d: int, *, norm_type: str, device: str,
         prec: Precision = FP16, launches: int = 1) -> TOBreakdown:
    """RMSNorm or LayerNorm over n_tokens rows of width d."""
    elems = n_tokens * d
    return TOBreakdown(
        to_nonlinear=elems * tc.op(norm_type),
        to_sram=tc.mem_cost(2 * elems, "sram", prec.act),
        to_hbm=tc.mem_cost(d, tc.offchip_tier(device), prec.weight),  # gamma (and beta), small
        n_launches=launches,
    )


def ffn(n_tokens: int, d: int, d_ff: int, *, ffn_type: str, activation_kind: str,
        device: str, prec: Precision = FP16) -> TOBreakdown:
    """Feed-forward block. 'gated' is SwiGLU-style (gate, up, down); 'standard'
    is up then down. Elementwise activation and the gated multiply are fused.
    """
    if ffn_type == "gated":
        gate = linear(d, d_ff, n_tokens, device=device, prec=prec)
        up = linear(d, d_ff, n_tokens, device=device, prec=prec)
        act = activation(n_tokens * d_ff, kind=activation_kind, prec=prec)
        mul = TOBreakdown(to_mac=n_tokens * d_ff * tc.mac(prec.compute))  # fused gating multiply
        down = linear(d_ff, d, n_tokens, device=device, prec=prec)
        return total((gate, up, act, mul, down))
    elif ffn_type == "standard":
        up = linear(d, d_ff, n_tokens, device=device, prec=prec)
        act = activation(n_tokens * d_ff, kind=activation_kind, prec=prec)
        down = linear(d_ff, d, n_tokens, device=device, prec=prec)
        return total((up, act, down))
    raise ValueError(f"unknown ffn_type '{ffn_type}'")


def embedding_lookup(n_tokens: int, d: int, *, device: str,
                     prec: Precision = FP16) -> TOBreakdown:
    """Token-embedding gather: reads n_tokens rows of width d from off-chip; no MACs."""
    return TOBreakdown(
        to_hbm=tc.mem_cost(n_tokens * d, tc.offchip_tier(device), prec.weight),
        n_launches=1,
    )
