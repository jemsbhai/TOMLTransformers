"""Sweep package: pre-flight provenance, single-point measurement, and the grid
driver for EXP-002.

Built incrementally:
  1. provenance -- environment snapshot, config freeze, clean-git gate.
  2. point      -- single-point measurement: spec -> workload -> record.
  3. grid       -- frozen config -> list of PointSpecs (the enumeration).
"""

from __future__ import annotations

from .provenance import PreflightError, preflight
from .point import PointSpec, measure_single_point
from .grid import expand_grid, load_config

__all__ = [
    "preflight", "PreflightError",
    "PointSpec", "measure_single_point",
    "expand_grid", "load_config",
]
