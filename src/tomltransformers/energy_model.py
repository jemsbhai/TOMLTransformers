"""Energy-model formulation family and selection machinery.

This module is hardware- and architecture-agnostic. It consumes TO *features*
(aggregate transistor-operation counts produced by the architecture front-ends)
together with measured energy, fits a family of nested models, and reports
which formulation generalizes best.

The family, smallest to largest (the "exhaust, then select" program):

    M0_flops      energy ~ a * (to_compute + to_memory)
                  one coefficient. The calibrated-FLOPs null baseline.
    M1_comp_mem   energy ~ a_c*to_compute + a_m*to_memory
                  separates compute from memory (two coefficients).
    M2_overhead   M1 + a constant overhead/intercept term.
    M3_dispatch   M2 + a_o*n_launches   (kernel-dispatch cost; the signals-paper
                  "Python-loop dispatch" term, which maps onto naive decode).
    M4_fused      M3 + a_f*n_fused_steps (fused-sequential per-step cost).

All coefficients are constrained non-negative via NNLS: energy cannot decrease
with additional computation, memory traffic, or sequential overhead. This is the
same constraint and rationale used in the signals paper.

Model selection:
    - AIC and BIC are computed on the fitting (train) set; they penalize
      in-sample fit by parameter count, so an unnecessary term is not rewarded.
    - R^2 and MAPE are reported on a held-out (test) set to assess
      generalization.
    - No single criterion is hard-wired; fit_and_select returns the full ranked
      table and best_by(...) lets the caller choose (default: AIC).

The base feature record is a plain dict with these keys (missing keys default
to 0.0):

    to_compute     total compute TOs
    to_memory      total memory TOs (already in TO units; tier costs applied
                   upstream by the architecture front-end)
    n_launches     CUDA kernel launches (dispatch count), for M3
    n_fused_steps  fused-sequential timesteps, for M4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

# Canonical base features. Front-ends emit dicts keyed by these names.
FEATURES: tuple[str, ...] = ("to_compute", "to_memory", "n_launches", "n_fused_steps")

Record = Mapping[str, float]


# ------------------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------------------
@dataclass
class EnergyModel:
    """A single nested energy model: a fixed feature transform plus NNLS fit.

    Parameters
    ----------
    name : human-readable identifier (e.g. "M2_overhead").
    feature_spec : which base features form the (non-intercept) columns.
    use_intercept : append a constant column (non-negative baseline overhead).
    combine_total : if True, sum the feature_spec columns into a single column
        (used by M0 to model energy as a function of total TOs only).
    """

    name: str
    feature_spec: tuple[str, ...]
    use_intercept: bool = False
    combine_total: bool = False
    coef_: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_params(self) -> int:
        base = 1 if self.combine_total else len(self.feature_spec)
        return base + (1 if self.use_intercept else 0)

    @property
    def column_names(self) -> list[str]:
        if self.combine_total:
            cols = ["+".join(self.feature_spec)]
        else:
            cols = list(self.feature_spec)
        if self.use_intercept:
            cols.append("intercept")
        return cols

    def design_matrix(self, records: Sequence[Record]) -> np.ndarray:
        """Build the (n_samples x n_params) design matrix from feature dicts."""
        vals = np.array(
            [[float(r.get(f, 0.0)) for f in self.feature_spec] for r in records],
            dtype=float,
        )
        if vals.ndim == 1:  # single feature, keep 2D
            vals = vals.reshape(len(records), -1)
        if self.combine_total:
            cols = vals.sum(axis=1, keepdims=True)
        else:
            cols = vals
        if self.use_intercept:
            cols = np.hstack([cols, np.ones((cols.shape[0], 1))])
        return cols

    def fit(self, records: Sequence[Record], energy: Sequence[float]) -> "EnergyModel":
        A = self.design_matrix(records)
        y = np.asarray(energy, dtype=float)
        if A.shape[0] != y.shape[0]:
            raise ValueError(f"{A.shape[0]} records but {y.shape[0]} energies")
        coef, _ = nnls(A, y)
        self.coef_ = coef
        return self

    def predict(self, records: Sequence[Record]) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError(f"model '{self.name}' is not fitted")
        return self.design_matrix(records) @ self.coef_


def model_family() -> dict[str, EnergyModel]:
    """The full nested family, in increasing complexity."""
    return {
        "M0_flops": EnergyModel("M0_flops", ("to_compute", "to_memory"),
                                use_intercept=False, combine_total=True),
        "M1_comp_mem": EnergyModel("M1_comp_mem", ("to_compute", "to_memory"),
                                   use_intercept=False),
        "M2_overhead": EnergyModel("M2_overhead", ("to_compute", "to_memory"),
                                   use_intercept=True),
        "M3_dispatch": EnergyModel("M3_dispatch",
                                   ("to_compute", "to_memory", "n_launches"),
                                   use_intercept=True),
        "M4_fused": EnergyModel("M4_fused",
                                ("to_compute", "to_memory", "n_launches", "n_fused_steps"),
                                use_intercept=True),
    }


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%). Zero-energy targets are excluded."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.abs(y_true) > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rss = float(np.sum((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))
    # Numerical guard so a (near-)perfect fit does not send log to -inf; preserves ordering.
    floor = 1e-12 * max(1.0, float(np.sum(np.asarray(y_true, float) ** 2)))
    return max(rss, floor)


def aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Gaussian AIC up to an additive constant common across models.

    k = n_params + 1 (the +1 counts the estimated error variance).
    """
    n = len(y_true)
    k = n_params + 1
    return n * np.log(_rss(y_true, y_pred) / n) + 2 * k


def bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Gaussian BIC up to an additive constant common across models."""
    n = len(y_true)
    k = n_params + 1
    return n * np.log(_rss(y_true, y_pred) / n) + k * np.log(n)


# ------------------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------------------
@dataclass
class FitResult:
    name: str
    n_params: int
    coef: np.ndarray
    column_names: list[str]
    r2_train: float
    r2_test: float
    mape_test: float
    aic: float
    bic: float


def fit_and_select(
    records_train: Sequence[Record],
    energy_train: Sequence[float],
    records_test: Sequence[Record],
    energy_test: Sequence[float],
    models: Iterable[EnergyModel] | None = None,
) -> list[FitResult]:
    """Fit every model on train, evaluate on test, return results sorted by AIC.

    AIC/BIC use the train fit (in-sample, complexity-penalized). R^2/MAPE use
    the held-out test set (generalization).
    """
    if models is None:
        models = list(model_family().values())

    results: list[FitResult] = []
    for m in models:
        m.fit(records_train, energy_train)
        yhat_train = m.predict(records_train)
        yhat_test = m.predict(records_test)
        results.append(
            FitResult(
                name=m.name,
                n_params=m.n_params,
                coef=m.coef_.copy(),
                column_names=m.column_names,
                r2_train=r2_score(energy_train, yhat_train),
                r2_test=r2_score(energy_test, yhat_test),
                mape_test=mape(energy_test, yhat_test),
                aic=aic(energy_train, yhat_train, m.n_params),
                bic=bic(energy_train, yhat_train, m.n_params),
            )
        )
    results.sort(key=lambda r: r.aic)
    return results


def best_by(results: Sequence[FitResult], criterion: str = "aic") -> FitResult:
    """Pick the winning model. criterion in {aic, bic, r2_test, mape_test}."""
    if criterion in ("aic", "bic", "mape_test"):
        return min(results, key=lambda r: getattr(r, criterion))
    if criterion == "r2_test":
        return max(results, key=lambda r: r.r2_test)
    raise ValueError(f"unknown criterion '{criterion}'")


def summary_table(results: Sequence[FitResult]) -> str:
    """Human-readable ranking for the console / logbook."""
    lines = [
        f"{'model':14s} {'k':>3s} {'R2_train':>9s} {'R2_test':>9s} "
        f"{'MAPE%':>8s} {'AIC':>10s} {'BIC':>10s}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r.name:14s} {r.n_params:>3d} {r.r2_train:>9.4f} {r.r2_test:>9.4f} "
            f"{r.mape_test:>8.2f} {r.aic:>10.1f} {r.bic:>10.1f}"
        )
    return "\n".join(lines)
