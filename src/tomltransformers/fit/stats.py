"""Paired significance testing for the section-8 bake-off.

Pre-registered protocol (fit_plan sections 8 and 12): paired per-point
absolute percentage errors on the identical held-out set, one-sided Wilcoxon
signed-rank per must-beat comparison (alternative: the TOML winner's APEs
are smaller), Holm-Bonferroni across the three comparisons, alpha 0.05.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

ALPHA = 0.05


def ape_pct(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return np.abs(y_true - y_pred) / np.abs(y_true) * 100.0


def wilcoxon_less(ape_winner, ape_baseline) -> float:
    """One-sided p-value that the winner's paired APEs are smaller."""
    a = np.asarray(ape_winner, float)
    b = np.asarray(ape_baseline, float)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have identical shape")
    if np.allclose(a, b):
        return 1.0
    stat, p = wilcoxon(a, b, alternative="less")
    return float(p)


def holm_adjust(pvals) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, clipped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return [float(v) for v in adjusted]
