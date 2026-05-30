"""Sweep package: pre-flight provenance, single-point measurement, and the grid
driver for EXP-002.

Built incrementally:
  1. provenance -- environment snapshot, config freeze, clean-git gate.
"""

from __future__ import annotations

from .provenance import PreflightError, preflight

__all__ = ["preflight", "PreflightError"]
