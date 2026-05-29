"""Energy-model formulation family and selection machinery.

This module is hardware- and architecture-agnostic. It consumes TO *features*
(aggregate transistor-operation counts, already weighted by the to_costs priors,
produced by the architecture front-ends) together with measured energy, fits a
family of nested models by non-negative least squares, and reports which
formulation generalizes best.

--------------------------------------------------------------------------------
Feature vocabulary (a record is a dict; missing keys default to 0.0)
--------------------------------------------------------------------------------
Compute (split):
    to_mac          MAC TOs (linear projections, attention matmuls), prior-weighted
    to_nonlinear    softmax / activation / norm TOs, prior-weighted
Memory (split):
    to_sram         on-chip SRAM-access TOs, prior-weighted
    to_hbm          off-chip DRAM-class TOs (HBM/GDDR, device-resolved), prior-weighted
Dispatch:
    n_launches      CUDA kernel launches (the signals-paper "Python-loop dispatch")
    n_fused_steps   fused-sequential timesteps

Aggregates used by the coarser models are sums of the above:
    compute = to_mac + to_nonlinear
    memory  = to_sram + to_hbm
    total   = compute + memory

--------------------------------------------------------------------------------
Why split models matter (the node question, made empirical)
--------------------------------------------------------------------------------
Because features are pre-weighted by the to_costs priors, a model that gives
memory a single coefficient assumes the prior SRAM/HBM cost ratio is correct.
The split model (separate to_sram, to_hbm coefficients) fits a correction to that
ratio from measured energy. If the split model wins on held-out error and
information criteria, the 45 nm-era ratio needed correcting at the modern node;
if it does not, the prior ratio is adequate. The same applies to to_mac vs
to_nonlinear (softmax/activation weighting).

--------------------------------------------------------------------------------
The family (nested lattice, increasing complexity)
--------------------------------------------------------------------------------
    M0_flops          total                                   (1)  calibrated-FLOPs null
    M1_comp_mem       compute, memory                         (2)
    M2_overhead       compute, memory, b                      (3)
    M3_dispatch       compute, memory, launches, b            (4)
    M4_fused          compute, memory, launches, fused, b     (5)
    M5_mem_split      compute, sram, hbm, b                   (4)  fits SRAM/HBM ratio
    M6_compute_split  mac, nonlinear, memory, b               (4)  fits softmax/MAC ratio
    M7_full_split     mac, nonlinear, sram, hbm, b            (5)
    M8_split_dispatch mac, nonlinear, sram, hbm, launches, b  (6)
    M9_full           mac, nonlinear, sram, hbm, launches, fused, b  (7)

All coefficients are constrained non-negative (NNLS): energy cannot decrease with
additional computation, memory traffic, or sequential overhead.

Selection: AIC/BIC computed on the fitting (train) set penalize complexity;
R^2/MAPE reported on held-out (test) data assess generalization. No single
criterion is hard-wired; fit_and_select returns the ranked table and best_by(...)
chooses (default: AIC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

# Canonical base features. Front-ends emit dicts keyed by these names.
FEATURES: tuple[str, ...] = (
    "to_mac", "to_nonlinear", "to_sram", "to_hbm", "n_launches", "n_fused_steps",
)

# Aggregate column specs (each is a set of base features summed into one column).
COMPUTE: tuple[str, ...] = ("to_mac", "to_nonlinear")
MEMORY: tuple[str, ...] = ("to_sram", "to_hbm")
TOTAL: tuple[str, ...] = COMPUTE + MEMORY

Record = Mapping[str, float]


# ------------------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------------------
@dataclass
class EnergyModel:
    """A nested energy model: fixed column specs plus an NNLS fit.

    Each entry of ``columns`` is a tuple of base-feature names that are summed
    into a single regressor column. ``use_intercept`` appends a constant
    (non-negative baseline overhead) column.
    """

    name: str
    columns: tuple[tuple[str, ...], ...]
    use_intercept: bool = False
    coef_: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_params(self) -> int:
        return len(self.columns) + (1 if self.use_intercept else 0)

    @property
    def column_names(self) -> list[str]:
        names = ["+".join(spec) for spec in self.columns]
        if self.use_intercept:
            names.append("intercept")
        return names

    def design_matrix(self, records: Sequence[Record]) -> np.ndarray:
        feats = np.array(
            [[sum(float(r.get(f, 0.0)) for f in spec) for spec in self.columns]
             for r in records],
            dtype=float,
        )
        if feats.ndim == 1:
            feats = feats.reshape(len(records), -1)
        if self.use_intercept:
            feats = np.hstack([feats, np.ones((feats.shape[0], 1))])
        return feats

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
    mac, nl = ("to_mac",), ("to_nonlinear",)
    sram, hbm = ("to_sram",), ("to_hbm",)
    launch, fused = ("n_launches",), ("n_fused_steps",)
    return {
        "M0_flops": EnergyModel("M0_flops", (TOTAL,)),
        "M1_comp_mem": EnergyModel("M1_comp_mem", (COMPUTE, MEMORY)),
        "M2_overhead": EnergyModel("M2_overhead", (COMPUTE, MEMORY), use_intercept=True),
        "M3_dispatch": EnergyModel("M3_dispatch", (COMPUTE, MEMORY, launch),
                                   use_intercept=True),
        "M4_fused": EnergyModel("M4_fused", (COMPUTE, MEMORY, launch, fused),
                                use_intercept=True),
        "M5_mem_split": EnergyModel("M5_mem_split", (COMPUTE, sram, hbm),
                                    use_intercept=True),
        "M6_compute_split": EnergyModel("M6_compute_split", (mac, nl, MEMORY),
                                        use_intercept=True),
        "M7_full_split": EnergyModel("M7_full_split", (mac, nl, sram, hbm),
                                     use_intercept=True),
        "M8_split_dispatch": EnergyModel("M8_split_dispatch", (mac, nl, sram, hbm, launch),
                                         use_intercept=True),
        "M9_full": EnergyModel("M9_full", (mac, nl, sram, hbm, launch, fused),
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
    floor = 1e-12 * max(1.0, float(np.sum(np.asarray(y_true, float) ** 2)))
    return max(rss, floor)


def aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Gaussian AIC up to an additive constant common across models (k = n_params + 1)."""
    n = len(y_true)
    k = n_params + 1
    return n * np.log(_rss(y_true, y_pred) / n) + 2 * k


def bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    """Gaussian BIC up to an additive constant common across models (k = n_params + 1)."""
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
    """Fit every model on train, evaluate on test, return results sorted by AIC."""
    if models is None:
        models = list(model_family().values())

    results: list[FitResult] = []
    for m in models:
        m.fit(records_train, energy_train)
        results.append(
            FitResult(
                name=m.name,
                n_params=m.n_params,
                coef=m.coef_.copy(),
                column_names=m.column_names,
                r2_train=r2_score(energy_train, m.predict(records_train)),
                r2_test=r2_score(energy_test, m.predict(records_test)),
                mape_test=mape(energy_test, m.predict(records_test)),
                aic=aic(energy_train, m.predict(records_train), m.n_params),
                bic=bic(energy_train, m.predict(records_train), m.n_params),
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
        f"{'model':18s} {'k':>3s} {'R2_train':>9s} {'R2_test':>9s} "
        f"{'MAPE%':>8s} {'AIC':>10s} {'BIC':>10s}",
        "-" * 74,
    ]
    for r in results:
        lines.append(
            f"{r.name:18s} {r.n_params:>3d} {r.r2_train:>9.4f} {r.r2_test:>9.4f} "
            f"{r.mape_test:>8.2f} {r.aic:>10.1f} {r.bic:>10.1f}"
        )
    return "\n".join(lines)
