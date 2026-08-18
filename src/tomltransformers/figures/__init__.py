"""Figure data layer for the EXP-002 paper figures (Step 8a).

Plotting lives in scripts/make_figures.py; everything that touches frozen
data or committed artifacts lives here so it can be unit tested.
"""

from .data import (
    A100_ENERGY,
    A100_EXPLORE,
    A100_FIT,
    A100_VALIDATION,
    R4090_ENERGY,
    R4090_FIT,
    R4090_PREDICTIONS,
    R4090_VALIDATION,
    A100_PREDICTIONS,
    GateFailure,
    GateResult,
    ab_percentages,
    load_a100_explore,
    load_a100_fit,
    load_a100_predictions,
    load_4090_fit,
    load_4090_predictions,
    load_records,
    precision_pairs,
    run_lineage_gate,
)

__all__ = [
    "A100_ENERGY",
    "A100_EXPLORE",
    "A100_FIT",
    "A100_PREDICTIONS",
    "A100_VALIDATION",
    "R4090_ENERGY",
    "R4090_FIT",
    "R4090_PREDICTIONS",
    "R4090_VALIDATION",
    "GateFailure",
    "GateResult",
    "ab_percentages",
    "load_4090_fit",
    "load_4090_predictions",
    "load_a100_explore",
    "load_a100_fit",
    "load_a100_predictions",
    "load_records",
    "precision_pairs",
    "run_lineage_gate",
]
