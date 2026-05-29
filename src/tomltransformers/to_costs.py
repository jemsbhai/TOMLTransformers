"""Transistor-operation (TO) cost model: the single source of truth.

This module centralizes every TO constant used across the framework, replacing
the duplicated cost tables that previously lived in the prototype scripts and
the design-doc markdown. Nothing else in the package should hard-code a TO
value; everything imports from here.

================================================================================
Unit convention
================================================================================
TO values are energies at a reference operating point, with the convention

    1 TO  ==  1 fJ  (femtojoule)

so a TO value can be read directly as an energy in fJ. This lets us seed the
priors from published per-operation and per-bit energy figures and keep them
physically interpretable. The fitted energy-model coefficients (energy_model.py)
still absorb the residual node/voltage/overhead scaling; what the framework
relies on is the *relative* structure across operation types and memory tiers,
which is exactly what we ground in the literature here.

================================================================================
Provenance of the numbers (see SOURCES for full citations)
================================================================================
Compute. Per-operation logic energies trace to Horowitz's 45 nm data, still the
reference table used in current hardware talks (e.g. Dally, Hot Chips 2023). A
32 nm-and-below re-tabulation of *logic* energy per primitive op is not
available in the literature as a clean table, so we keep the 45 nm relative
structure as the prior and fit absolute scale. A modern 5 nm measured anchor
(Keller et al., JSSC 2023: 95.6 TOPS/W INT4) corroborates that low precision is
dramatically cheaper.

Memory. Unlike logic, off-chip DRAM-class memory has current, vendor-grade
energy-per-bit figures, and they are the numbers that dominate inference energy.
We seed each memory technology from its published pJ/bit and convert to fJ per
32-bit word (pJ/bit * 32 bits * 1000 fJ/pJ):

    on-chip SRAM   ~0.16 pJ/bit   (Horowitz 8 KB SRAM)            ->   5,000 fJ/word
    HBM4           ~2.40 pJ/bit   (~40% below HBM3E)              ->  76,800 fJ/word
    HBM3E          ~4.05 pJ/bit   (Samsung/SK hynix)              -> 129,600 fJ/word
    HBM2E          ~6.00 pJ/bit   (HBM2 6.25; A100 memory)        -> 192,000 fJ/word
    GDDR6X         ~7.25 pJ/bit   (Micron; RTX 4090 memory)       -> 232,000 fJ/word
    GDDR6          ~7.50 pJ/bit   (Micron)                        -> 240,000 fJ/word

These vendor pJ/bit values are the *ideal data-transfer* energy. The effective
per-word cost on a real GPU also includes memory-controller, activation, refresh,
and row-buffer effects, so every memory tier remains a calibration target.

This corrects a physical error in the prototype, which used 10,000 TOs/word for
HBM (~0.3 pJ/bit), roughly 20x too low. Because the memory-compute energy ratio
(MCER) is the ratio of memory TOs to compute TOs in the same unit, an under-scaled
memory cost suppressed MCER; with physically grounded costs, decode is far more
memory-bound than the prototype's MCER ~= 2.0 suggested. The true value is a
calibration outcome.

================================================================================
Device registry
================================================================================
Off-chip memory cost is device-specific: the RTX 4090 uses GDDR6X (~7.25 pJ/bit),
the A100 uses HBM2E (~6.0 pJ/bit). The 4090's memory is therefore *less* efficient
per bit than the A100's. DEVICES maps each GPU to its off-chip technology and
process node; the architecture front-ends resolve the off-chip tier via
offchip_tier(device).

================================================================================
Calibration targets
================================================================================
CALIBRATION_TARGETS collects the costs that must be fit against measured energy
rather than reported as established: the nonlinear/iterative op costs
(softmax, GELU, SiLU, norms, exp, div, sqrt), the precision MAC multipliers, and
every memory tier (effective on-GPU per-word cost). They must never be presented
as ground truth without calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


# ------------------------------------------------------------------------------
# Citations
# ------------------------------------------------------------------------------
SOURCES: Dict[str, str] = {
    "horowitz2014": "M. Horowitz, 'Computing's Energy Problem (and what we can do "
                    "about it)', IEEE ISSCC, 2014.",
    "dally2023": "W. J. Dally, 'Hardware for Deep Learning', Hot Chips 2023 keynote "
                 "(per-op energy table footnoted as 45 nm).",
    "keller2023": "B. Keller, R. Venkatesan et al., 'A 95.6-TOPS/W Deep Learning "
                  "Inference Accelerator with Per-Vector Scaled 4-bit Quantization "
                  "in 5 nm', IEEE JSSC, 2023.",
    "micron_gddr6x": "Micron, 'Doubling I/O Performance with PAM4: GDDR6X' technical "
                     "brief, 2020 (~7.25 pJ/bit GDDR6X, ~7.5 pJ/bit GDDR6).",
    "hbm_roadmap": "SK hynix / Samsung HBM generational efficiency data: HBM2 "
                   "6.25 pJ/bit -> HBM3E 4.05 pJ/bit; HBM4 ~40% below HBM3E "
                   "(2024-2025).",
}


class Provenance(str, Enum):
    """Where a TO constant comes from, for honest reporting and auditing."""

    MEASURED = "measured"            # measured silicon energy (Horowitz 45 nm or vendor device data)
    PUBLISHED_TOML = "published"     # appears in a published/submitted TOML paper
    NEW_ESTIMATE = "new_estimate"    # estimated here; a calibration target


@dataclass(frozen=True)
class TOCost:
    """A single TO cost: value (in TOs == fJ), provenance, citation, and note."""

    value: float
    provenance: Provenance
    source: str = ""   # key into SOURCES, or "" for pure estimates
    note: str = ""


# ------------------------------------------------------------------------------
# Per-operation TO costs (switching events per single operation, in fJ).
# ------------------------------------------------------------------------------
OP_COST: Dict[str, TOCost] = {
    # --- Anchored arithmetic / logic (Horowitz 45 nm relative structure) ---
    "mac": TOCost(5_000, Provenance.PUBLISHED_TOML, "horowitz2014",
                  "FP32 fused multiply-add (~4.6 pJ); precision anchor"),
    "comparison": TOCost(50, Provenance.PUBLISHED_TOML, "horowitz2014",
                         "simple comparator"),
    "relu": TOCost(100, Provenance.PUBLISHED_TOML, "horowitz2014",
                   "compare + conditional pass-through"),

    # --- Iterative / transcendental (estimates: poly approx / Newton-Raphson) ---
    "exp": TOCost(18_000, Provenance.NEW_ESTIMATE, "",
                  "polynomial approximation; same circuit family as sigmoid"),
    "div": TOCost(15_000, Provenance.NEW_ESTIMATE, "",
                  "iterative divider (Newton-Raphson)"),
    "sqrt": TOCost(15_000, Provenance.NEW_ESTIMATE, "",
                   "iterative; similar to division"),

    # --- Nonlinear activations ---
    "sigmoid": TOCost(18_000, Provenance.PUBLISHED_TOML, "horowitz2014",
                      "LUT + interpolation (FLAIRS)"),
    "tanh": TOCost(15_000, Provenance.PUBLISHED_TOML, "horowitz2014",
                   "LUT + interpolation (FLAIRS)"),
    "gelu": TOCost(20_000, Provenance.NEW_ESTIMATE, "",
                   "tanh-approx GELU: tanh + MACs"),
    "silu": TOCost(23_100, Provenance.NEW_ESTIMATE, "",
                   "x * sigmoid(x): sigmoid + MAC + multiply"),

    # --- Attention / normalization (per element, amortized where noted) ---
    "softmax": TOCost(25_000, Provenance.NEW_ESTIMATE, "",
                      "per element: exp + accumulate + divide; single per-element "
                      "cost, NOT exp+div composed"),
    "layernorm": TOCost(12_000, Provenance.NEW_ESTIMATE, "",
                        "per element, amortized: mean + var + rsqrt + scale"),
    "rmsnorm": TOCost(10_000, Provenance.NEW_ESTIMATE, "",
                      "per element, amortized: square + mean + rsqrt + scale"),
}


# ------------------------------------------------------------------------------
# Memory hierarchy: TO cost per 32-bit word (in fJ), grounded in pJ/bit data.
# ------------------------------------------------------------------------------
# On-chip tiers, then off-chip DRAM-class technologies (device-selected). The
# SRAM-vs-off-chip gap drives the prefill/decode MCER transition.

MEM_TIER: Dict[str, TOCost] = {
    # --- On-chip ---
    "register": TOCost(1_600, Provenance.NEW_ESTIMATE, "",
                       "~0.05 pJ/bit; register file / local reuse (estimate)"),
    "sram": TOCost(5_000, Provenance.MEASURED, "horowitz2014",
                   "~0.16 pJ/bit; on-chip 8 KB SRAM (L1/L2/shared)"),
    # --- Off-chip DRAM-class (vendor pJ/bit * 32 bits * 1000 fJ/pJ) ---
    "hbm4": TOCost(76_800, Provenance.MEASURED, "hbm_roadmap",
                   "~2.40 pJ/bit; HBM4 (~40% below HBM3E)"),
    "hbm3e": TOCost(129_600, Provenance.MEASURED, "hbm_roadmap",
                    "~4.05 pJ/bit; HBM3E (Samsung/SK hynix)"),
    "hbm2e": TOCost(192_000, Provenance.MEASURED, "hbm_roadmap",
                    "~6.00 pJ/bit; HBM2/2E; A100 memory"),
    "gddr6x": TOCost(232_000, Provenance.MEASURED, "micron_gddr6x",
                     "~7.25 pJ/bit; GDDR6X; RTX 4090 memory"),
    "gddr6": TOCost(240_000, Provenance.MEASURED, "micron_gddr6x",
                    "~7.50 pJ/bit; GDDR6"),
}

# Cheapest -> most expensive (used for sanity checks).
MEM_TIER_ORDER = ("register", "sram", "hbm4", "hbm3e", "hbm2e", "gddr6x", "gddr6")

ONCHIP_TIERS = ("register", "sram")
OFFCHIP_TIERS = ("hbm4", "hbm3e", "hbm2e", "gddr6x", "gddr6")


# ------------------------------------------------------------------------------
# Device registry: which off-chip memory each GPU uses, and its process node.
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Device:
    name: str
    offchip_tier: str   # key into MEM_TIER
    process_node: str
    note: str = ""


DEVICES: Dict[str, Device] = {
    "rtx4090": Device("rtx4090", "gddr6x", "TSMC 4N",
                      "Ada Lovelace; local calibration primary"),
    "a100": Device("a100", "hbm2e", "TSMC N7",
                   "Ampere A100 40GB SXM4; cross-platform validation"),
    "h100": Device("h100", "hbm3e", "TSMC 4N",
                   "Hopper; HBM3/3E-class (approximate)"),
}


# ------------------------------------------------------------------------------
# Precision: bytes per element and MAC cost multiplier relative to FP32.
# ------------------------------------------------------------------------------
PRECISION_BYTES: Dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,   # stored 32-bit; reduced-precision mantissa
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

PRECISION_MAC_MULT: Dict[str, TOCost] = {
    "fp32": TOCost(1.00, Provenance.PUBLISHED_TOML, "horowitz2014", "anchor"),
    "tf32": TOCost(0.50, Provenance.NEW_ESTIMATE, "", "reduced-mantissa estimate"),
    "fp16": TOCost(0.33, Provenance.NEW_ESTIMATE, "horowitz2014",
                   "Horowitz-informed estimate"),
    "bf16": TOCost(0.33, Provenance.NEW_ESTIMATE, "horowitz2014",
                   "Horowitz-informed estimate"),
    "int8": TOCost(0.05, Provenance.NEW_ESTIMATE, "horowitz2014",
                   "INT8 MAC ~0.23 pJ vs FP32 4.6 pJ"),
    "int4": TOCost(0.025, Provenance.NEW_ESTIMATE, "keller2023",
                   "INT4; 5 nm 95.6 TOPS/W corroborates large saving"),
}

WORD_BITS = 32  # the memory-cost accounting unit: one word = 32 bits = 4 bytes


# ------------------------------------------------------------------------------
# Calibration targets: estimated/effective values fit against measured energy.
# ------------------------------------------------------------------------------
CALIBRATION_TARGETS: frozenset[str] = frozenset(
    [k for k, c in OP_COST.items() if c.provenance is Provenance.NEW_ESTIMATE]
    + [f"mem:{t}" for t in MEM_TIER]  # effective on-GPU per-word cost is fit for every tier
    + [f"prec:{p}" for p, c in PRECISION_MAC_MULT.items()
       if c.provenance is Provenance.NEW_ESTIMATE]
)


# ------------------------------------------------------------------------------
# Accessors. Use these rather than reaching into the dicts directly.
# ------------------------------------------------------------------------------
def op(name: str) -> float:
    """TO cost (fJ) of a single operation, e.g. op('softmax')."""
    try:
        return OP_COST[name].value
    except KeyError as exc:
        raise KeyError(f"unknown operation '{name}'; known: {sorted(OP_COST)}") from exc


def mem_word(tier: str) -> float:
    """TO cost (fJ) of one 32-bit word access at the given memory tier."""
    try:
        return MEM_TIER[tier].value
    except KeyError as exc:
        raise KeyError(f"unknown memory tier '{tier}'; known: {sorted(MEM_TIER)}") from exc


def pj_per_bit(tier: str) -> float:
    """Physical energy of a memory tier in pJ/bit (for cross-checking priors)."""
    return mem_word(tier) / WORD_BITS / 1000.0


def offchip_tier(device: str) -> str:
    """The off-chip memory tier used by a given GPU, e.g. offchip_tier('rtx4090')."""
    try:
        return DEVICES[device].offchip_tier
    except KeyError as exc:
        raise KeyError(f"unknown device '{device}'; known: {sorted(DEVICES)}") from exc


def _check_precision(precision: str) -> None:
    if precision not in PRECISION_BYTES:
        raise KeyError(f"unknown precision '{precision}'; known: {sorted(PRECISION_BYTES)}")


def mac(precision: str = "fp32") -> float:
    """TO cost (fJ) of one MAC at the given precision."""
    _check_precision(precision)
    return OP_COST["mac"].value * PRECISION_MAC_MULT[precision].value


def words_per_element(precision: str = "fp32") -> float:
    """32-bit words occupied by one element: fp32->1.0, fp16->0.5, int8->0.25, int4->0.125."""
    _check_precision(precision)
    return PRECISION_BYTES[precision] / (WORD_BITS / 8.0)


def mem_cost(n_elements: float, tier: str, precision: str = "fp32") -> float:
    """TO cost to move ``n_elements`` of ``precision`` data through ``tier``.

    Lower-precision data occupies fewer words, so a given parameter count costs
    proportionally less traffic at INT8/INT4.
    """
    return n_elements * words_per_element(precision) * mem_word(tier)


# ------------------------------------------------------------------------------
# Self-test / human-readable dump.
# ------------------------------------------------------------------------------
def _selftest() -> None:
    # Memory ordering strictly increasing.
    costs = [mem_word(t) for t in MEM_TIER_ORDER]
    assert costs == sorted(costs) and len(set(costs)) == len(costs), costs

    # On-chip strictly cheaper than every off-chip technology.
    assert max(mem_word(t) for t in ONCHIP_TIERS) < min(mem_word(t) for t in OFFCHIP_TIERS)

    # Priors reproduce the cited pJ/bit figures.
    assert abs(pj_per_bit("hbm2e") - 6.00) < 1e-6
    assert abs(pj_per_bit("hbm3e") - 4.05) < 1e-6
    assert abs(pj_per_bit("gddr6x") - 7.25) < 1e-6
    assert abs(pj_per_bit("sram") - 0.15625) < 1e-6

    # Device mapping.
    assert offchip_tier("rtx4090") == "gddr6x"
    assert offchip_tier("a100") == "hbm2e"

    # Precision monotone, word occupancy.
    assert mac("fp32") >= mac("fp16") >= mac("int8") >= mac("int4")
    assert words_per_element("int4") == 0.125

    # Off-chip dominates on-chip SRAM by ~30-50x (A100/4090).
    assert 30 < mem_word("hbm2e") / mem_word("sram") < 60
    assert 30 < mem_word("gddr6x") / mem_word("sram") < 60

    # Headline framing: softmax is 5 MACs in TOs (5000x vs a FLOP).
    assert op("softmax") / op("mac") == 5.0

    # Every calibration target references a real key.
    for key in CALIBRATION_TARGETS:
        head, _, name = key.partition(":")
        if head == "mem":
            assert name in MEM_TIER
        elif head == "prec":
            assert name in PRECISION_MAC_MULT
        else:
            assert key in OP_COST

    # Cited (measured/published) costs carry a source.
    for table in (OP_COST, MEM_TIER):
        for k, c in table.items():
            if c.provenance in (Provenance.MEASURED, Provenance.PUBLISHED_TOML):
                assert c.source in SOURCES, (k, c.source)

    print("to_costs self-test: OK")


def _dump() -> None:
    print(f"\nTO cost model (1 TO == 1 fJ at reference; MAC_FP32 = {OP_COST['mac'].value})")
    print("=" * 78)
    print("\nOperations:")
    for name, c in OP_COST.items():
        star = " *" if name in CALIBRATION_TARGETS else "  "
        print(f"  {star} {name:12s} {c.value:>10,.0f}  [{c.provenance.value:12s}] {c.note}")
    print("\nMemory tiers (per 32-bit word | pJ/bit):")
    for name in MEM_TIER_ORDER:
        c = MEM_TIER[name]
        star = " *" if f"mem:{name}" in CALIBRATION_TARGETS else "  "
        print(f"  {star} {name:8s} {c.value:>10,.0f} fJ  {pj_per_bit(name):>6.3f} pJ/bit  "
              f"[{c.provenance.value:9s}] {c.note}")
    print("\nDevices:")
    for d in DEVICES.values():
        print(f"     {d.name:9s} off-chip={d.offchip_tier:7s} ({pj_per_bit(d.offchip_tier):.2f} pJ/bit)  "
              f"node={d.process_node}")
    print(f"\n* = calibration target ({len(CALIBRATION_TARGETS)} total)")


if __name__ == "__main__":
    _dump()
    print()
    _selftest()
