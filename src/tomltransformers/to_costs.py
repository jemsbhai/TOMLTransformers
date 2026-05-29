"""Transistor-operation (TO) cost model: the single source of truth.

This module centralizes every TO constant used across the framework, replacing
the duplicated cost tables that previously lived in the prototype scripts and
the design-doc markdown. Nothing else in the package should hard-code a TO
value; everything imports from here.

--------------------------------------------------------------------------------
Reference node and unit convention
--------------------------------------------------------------------------------
TO values are anchored to Horowitz's 45 nm CMOS energy measurements with the
convention 1 TO is approximately 1 fJ. Under this convention an FP32
multiply-accumulate (one FP32 multiply at 3.7 pJ plus one FP32 add at 0.9 pJ,
or equivalently a single fused multiply-add at ~4.6 pJ) corresponds to ~5000
TOs. We adopt:

    MAC_FP32 = 5000 TOs   (the published TOML/signals value)

IMPORTANT: TO values are RELATIVE weights, not direct energy predictions. The
fitted energy-model coefficients (see energy_model.py) absorb absolute scale,
process-node scaling, and architecture/hardware specifics. What the framework
relies on is the relative structure across operation types, memory tiers, and
precisions, not the absolute fJ figure.

--------------------------------------------------------------------------------
Resolved inconsistency (FP16 vs FP32 anchor)
--------------------------------------------------------------------------------
The signals paper anchors MAC = 5000 to FP32. The quantization design doc
treated 5000 as the FP16 baseline. These disagree by the FP32/FP16 ratio. We
standardize on FP32 as the anchor (MAC_FP32 = 5000) and express every other
precision as a multiplier relative to FP32 (PRECISION_MAC_MULT). Because the
fitted coefficients absorb absolute scale, the choice of anchor does not change
predictions as long as it is applied consistently; we pick FP32 to match the
value printed in the published work.

--------------------------------------------------------------------------------
Calibration targets
--------------------------------------------------------------------------------
Several costs are estimates rather than directly measured Horowitz values, in
particular the nonlinear-activation costs (softmax, GELU, SiLU, norms), the
iterative-function costs (exp, div, sqrt), the HBM word cost, and the
precision MAC multipliers. These are flagged via Provenance.NEW_ESTIMATE and
collected in CALIBRATION_TARGETS. They are the parameters the measurement
campaign fits against observed energy; they must never be reported as
established constants without calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Provenance(str, Enum):
    """Where a TO constant comes from, for honest reporting and auditing."""

    MEASURED = "measured"            # directly from Horowitz (2014) 45 nm data
    PUBLISHED_TOML = "published"     # appears in a published/submitted TOML paper
    NEW_ESTIMATE = "new_estimate"    # estimated here; a calibration target


@dataclass(frozen=True)
class TOCost:
    """A single TO cost with its value, provenance, and a short note."""

    value: float
    provenance: Provenance
    note: str = ""


# ------------------------------------------------------------------------------
# Per-operation TO costs (counts of switching events per single operation).
# ------------------------------------------------------------------------------
# Arithmetic and logic are anchored to Horowitz; nonlinear and iterative costs
# are estimates pending calibration. Activation/softmax/norm values match the
# signals-paper TO table for continuity across the paper series.

OP_COST: Dict[str, TOCost] = {
    # --- Anchored arithmetic / logic ---
    "mac": TOCost(5_000, Provenance.PUBLISHED_TOML,
                  "FP32 fused multiply-add (~4.6 pJ); the precision anchor"),
    "comparison": TOCost(50, Provenance.PUBLISHED_TOML,
                         "simple comparator (e.g. ReLU/argmax branch)"),
    "relu": TOCost(100, Provenance.PUBLISHED_TOML,
                   "compare + conditional pass-through"),

    # --- Iterative / transcendental (estimates: poly approx or Newton-Raphson) ---
    "exp": TOCost(18_000, Provenance.NEW_ESTIMATE,
                  "polynomial approximation; same circuit family as sigmoid"),
    "div": TOCost(15_000, Provenance.NEW_ESTIMATE,
                  "iterative divider (Newton-Raphson)"),
    "sqrt": TOCost(15_000, Provenance.NEW_ESTIMATE,
                   "iterative; similar to division"),

    # --- Nonlinear activations ---
    "sigmoid": TOCost(18_000, Provenance.PUBLISHED_TOML,
                      "LUT + interpolation (FLAIRS)"),
    "tanh": TOCost(15_000, Provenance.PUBLISHED_TOML,
                   "LUT + interpolation (FLAIRS)"),
    "gelu": TOCost(20_000, Provenance.NEW_ESTIMATE,
                   "tanh-approx GELU: tanh + MACs (signals table)"),
    "silu": TOCost(23_100, Provenance.NEW_ESTIMATE,
                   "x * sigmoid(x): sigmoid + MAC + multiply"),

    # --- Attention / normalization (per-element, amortized where noted) ---
    "softmax": TOCost(25_000, Provenance.NEW_ESTIMATE,
                      "per element: exp + accumulate + divide; modeled as a "
                      "single per-element cost, NOT exp+div composed"),
    "layernorm": TOCost(12_000, Provenance.NEW_ESTIMATE,
                        "per element, amortized: mean + var + rsqrt + scale"),
    "rmsnorm": TOCost(10_000, Provenance.NEW_ESTIMATE,
                      "per element, amortized: square + mean + rsqrt + scale"),
}


# ------------------------------------------------------------------------------
# Memory hierarchy: TO cost per 32-bit word, by tier.
# ------------------------------------------------------------------------------
# The SRAM-vs-HBM gap is what produces the prefill/decode MCER phase transition.
# HBM is the single most important calibration target in the whole framework.

MEM_TIER: Dict[str, TOCost] = {
    "register": TOCost(50, Provenance.NEW_ESTIMATE,
                       "register file / intermediate values"),
    "sram": TOCost(192, Provenance.PUBLISHED_TOML,
                   "on-chip L1/L2/shared memory (FLAIRS/signals)"),
    "hbm": TOCost(10_000, Provenance.PUBLISHED_TOML,
                  "off-chip HBM2e; weights and KV cache; KEY calibration target"),
    "dram": TOCost(64_000, Provenance.NEW_ESTIMATE,
                   "CPU-side DRAM; not used in GPU inference; for completeness"),
}

# Tier ordering from cheapest to most expensive (used for sanity checks).
MEM_TIER_ORDER = ("register", "sram", "hbm", "dram")


# ------------------------------------------------------------------------------
# Precision: bytes per element and MAC cost multiplier relative to FP32.
# ------------------------------------------------------------------------------
# Memory word counts scale with bytes-per-element (one 32-bit word = 4 bytes).
# Compute cost scales with PRECISION_MAC_MULT. The multipliers are estimates
# (Horowitz-informed) and are calibration targets, fit by the W4A16 / W8A8
# experiments.

PRECISION_BYTES: Dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,   # stored as 32-bit; reduced-precision mantissa
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

PRECISION_MAC_MULT: Dict[str, TOCost] = {
    "fp32": TOCost(1.00, Provenance.PUBLISHED_TOML, "anchor"),
    "tf32": TOCost(0.50, Provenance.NEW_ESTIMATE, "reduced-mantissa estimate"),
    "fp16": TOCost(0.33, Provenance.NEW_ESTIMATE, "Horowitz-informed estimate"),
    "bf16": TOCost(0.33, Provenance.NEW_ESTIMATE, "Horowitz-informed estimate"),
    "int8": TOCost(0.05, Provenance.NEW_ESTIMATE, "INT8 MAC ~0.23 pJ vs FP32 4.6"),
    "int4": TOCost(0.025, Provenance.NEW_ESTIMATE, "INT4 MAC estimate"),
}

WORD_BITS = 32  # one "word" = 32 bits = 4 bytes, the memory-cost accounting unit


# ------------------------------------------------------------------------------
# Calibration targets: the estimated values fit against measured energy.
# ------------------------------------------------------------------------------
CALIBRATION_TARGETS: frozenset[str] = frozenset(
    [k for k, c in OP_COST.items() if c.provenance is Provenance.NEW_ESTIMATE]
    + ["mem:hbm"]  # HBM is the headline calibration parameter even though listed PUBLISHED
    + [f"prec:{p}" for p, c in PRECISION_MAC_MULT.items()
       if c.provenance is Provenance.NEW_ESTIMATE]
)


# ------------------------------------------------------------------------------
# Accessors. Use these rather than reaching into the dicts directly.
# ------------------------------------------------------------------------------
def op(name: str) -> float:
    """TO cost of a single operation (e.g. op('softmax'))."""
    try:
        return OP_COST[name].value
    except KeyError as exc:
        raise KeyError(
            f"unknown operation '{name}'; known: {sorted(OP_COST)}"
        ) from exc


def mem_word(tier: str) -> float:
    """TO cost of one 32-bit word access at the given memory tier."""
    try:
        return MEM_TIER[tier].value
    except KeyError as exc:
        raise KeyError(
            f"unknown memory tier '{tier}'; known: {sorted(MEM_TIER)}"
        ) from exc


def _check_precision(precision: str) -> None:
    if precision not in PRECISION_BYTES:
        raise KeyError(
            f"unknown precision '{precision}'; known: {sorted(PRECISION_BYTES)}"
        )


def mac(precision: str = "fp32") -> float:
    """TO cost of one MAC at the given precision."""
    _check_precision(precision)
    return OP_COST["mac"].value * PRECISION_MAC_MULT[precision].value


def words_per_element(precision: str = "fp32") -> float:
    """Number of 32-bit words occupied by one element of the given precision.

    fp32 -> 1.0, fp16/bf16 -> 0.5, int8 -> 0.25, int4 -> 0.125.
    """
    _check_precision(precision)
    return PRECISION_BYTES[precision] / (WORD_BITS / 8.0)


def mem_cost(n_elements: float, tier: str, precision: str = "fp32") -> float:
    """TO cost to move ``n_elements`` of ``precision`` data through ``tier``.

    Accounts for the fact that lower-precision data occupies fewer words, so a
    given parameter count costs proportionally less HBM traffic at INT8/INT4.
    """
    return n_elements * words_per_element(precision) * mem_word(tier)


# ------------------------------------------------------------------------------
# Self-test / human-readable dump.
# ------------------------------------------------------------------------------
def _selftest() -> None:
    # Hierarchy ordering must be strictly increasing in cost.
    tiers = [mem_word(t) for t in MEM_TIER_ORDER]
    assert tiers == sorted(tiers) and len(set(tiers)) == len(tiers), tiers

    # Precision MAC cost must be monotone non-increasing fp32 >= fp16 >= int8 >= int4.
    assert mac("fp32") >= mac("fp16") >= mac("int8") >= mac("int4")

    # Word occupancy sanity.
    assert words_per_element("fp32") == 1.0
    assert words_per_element("fp16") == 0.5
    assert words_per_element("int4") == 0.125

    # HBM dominates SRAM by the expected order of magnitude (~52x).
    assert mem_word("hbm") / mem_word("sram") > 40.0

    # Headline framing: softmax costs 5x a MAC in TOs (5000x vs a FLOP, which
    # counts softmax as ~5 ops). Guard the ratio that the paper leans on.
    assert op("softmax") / op("mac") == 5.0

    # Every calibration target references a real key.
    for key in CALIBRATION_TARGETS:
        if key.startswith("mem:"):
            assert key.split(":", 1)[1] in MEM_TIER
        elif key.startswith("prec:"):
            assert key.split(":", 1)[1] in PRECISION_MAC_MULT
        else:
            assert key in OP_COST

    print("to_costs self-test: OK")


def _dump() -> None:
    print(f"\nTO cost model (1 TO ~= 1 fJ @ 45 nm; MAC_FP32 = {OP_COST['mac'].value})")
    print("=" * 72)
    for label, table in (("Operations", OP_COST), ("Memory tiers (per 32b word)", MEM_TIER)):
        print(f"\n{label}:")
        for name, c in table.items():
            star = " *" if (name in CALIBRATION_TARGETS or f"mem:{name}" in CALIBRATION_TARGETS) else "  "
            print(f"  {star} {name:12s} {c.value:>10,.0f}  [{c.provenance.value:13s}] {c.note}")
    print("\nPrecision (MAC multiplier rel. FP32 | bytes/elem):")
    for p in PRECISION_BYTES:
        c = PRECISION_MAC_MULT[p]
        star = " *" if f"prec:{p}" in CALIBRATION_TARGETS else "  "
        print(f"  {star} {p:6s} mult={c.value:<6} bytes={PRECISION_BYTES[p]:<4}  [{c.provenance.value}]")
    print(f"\n* = calibration target ({len(CALIBRATION_TARGETS)} total)")


if __name__ == "__main__":
    _dump()
    print()
    _selftest()
