"""Fit-time analysis package: feature bridge, splits, baselines (EXP-002+).

The bridge maps frozen sweep records to energy_model feature records per the
approved fit plan (experiments/exp_002_size_sweep/fit_plan.md). Splits,
baselines, and derived quantities land alongside as the fit script grows.
"""

from .bridge import (BridgeError, features_for_record, features_for_spec,
                     load_latest_records)

__all__ = [
    "BridgeError",
    "features_for_record",
    "features_for_spec",
    "load_latest_records",
]
