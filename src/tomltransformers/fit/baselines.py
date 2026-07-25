"""Pre-registered baselines for the section-8 bake-off (fit_plan sections 8
and 12). Every constant traces to a citation or to the frozen dataset.

Roofline constants (approved 2026-07-24; RTX 4090 Laptop GPU, AD103/GN21-X11):
- 9,728 CUDA cores; rated boost 2,040 MHz at the 150 W TGP configuration.
- Peak FP32 = 2 FLOP/core/cycle x 9,728 x 2.040e9 = 39.69 TFLOP/s (formula
  stated; cores and clock cited).
- Peak FP16 = 2 x peak FP32 = 79.38 TFLOP/s. ASSUMPTION, recorded as such:
  the standard Ada dense tensor-core FP16 rate for the cuBLAS/SDPA paths the
  measured workloads use.
- Memory bandwidth 576.0 GB/s (256-bit GDDR6 at 18 Gbps effective).
- Sensitivity ceiling: the frozen dataset's maximum sustained median SM clock
  is 2,325 MHz (energy.jsonl, validation report), giving 45.24 TFLOP/s FP32;
  reported as a one-line sensitivity only, never the primary.

Raw structural counts are recovered EXACTLY by inverting to_costs with the
same constants the feature bridge used, so the baselines consume untainted
op and word counts: raw_macs = to_mac / mac(prec); off-chip words =
to_hbm / mem_word(offchip_tier(device)); SRAM words = to_sram /
mem_word('sram'); FLOPs = 2 x raw_macs; bytes = words x 4.

The layerwise regressor (decisions D3/D5) is NNLS on the winner's selected
support with the physics priors stripped: columns raw_macs, sram_words,
hbm_words, n_launches, intercept. to_nonlinear is excluded: it aggregates
ops with different TO costs (no exact raw inverse) and the winner fitted it
to zero under both estimators.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .. import to_costs as tc

CITATIONS = {
    "techpowerup_4090m": "TechPowerUp GPU Database, RTX 4090 Mobile / Max-Q "
                         "(AD103, GN21-X11): 9728 shading units, 256-bit GDDR6, "
                         "2250 MHz (18 Gbps effective), 576.0 GB/s.",
    "techspot_4090m": "TechSpot, 'Nvidia GeForce RTX 4090 Laptop GPU Review', "
                      "Feb 2023: 2,040 MHz rated boost at the 150 W "
                      "configuration; 18 Gbps GDDR6.",
    "videocardz_4090m": "VideoCardz.net, 'NVIDIA GeForce RTX 4090 Laptop GPU': "
                        "16 GB GDDR6, 256-bit, 576 GB/s, boost 2040 MHz.",
    "measured_clock": "energy.jsonl (frozen EXP-002 dataset): maximum sustained "
                      "median SM clock 2325 MHz across 296 points.",
}

CORES = 9_728
RATED_BOOST_HZ = 2.040e9
PEAK_FP32_FLOPS_S = 2.0 * CORES * RATED_BOOST_HZ          # 39.69e12
PEAK_FP16_FLOPS_S = 2.0 * PEAK_FP32_FLOPS_S               # assumption (see above)
PEAK_BY_PRECISION = {"fp32": PEAK_FP32_FLOPS_S, "fp16": PEAK_FP16_FLOPS_S}
BANDWIDTH_BYTES_S = 576.0e9

MEASURED_MAX_SM_HZ = 2.325e9                              # frozen dataset
PEAK_FP32_MEASURED = 2.0 * CORES * MEASURED_MAX_SM_HZ     # 45.24e12
PEAK_BY_PRECISION_MEASURED = {"fp32": PEAK_FP32_MEASURED,
                              "fp16": 2.0 * PEAK_FP32_MEASURED}

BYTES_PER_WORD = 4.0


def raw_counts(feat: dict, precision: str, device: str = "rtx4090") -> dict:
    """Invert to_costs to recover exact raw structural counts per execution."""
    raw_macs = feat["to_mac"] / tc.mac(precision)
    off = tc.offchip_tier(device)
    hbm_words = feat["to_hbm"] / tc.mem_word(off)
    sram_words = feat["to_sram"] / tc.mem_word("sram")
    return {
        "raw_macs": raw_macs,
        "flops": 2.0 * raw_macs,
        "hbm_words": hbm_words,
        "hbm_bytes": hbm_words * BYTES_PER_WORD,
        "sram_words": sram_words,
    }


def roofline_time_s(feat: dict, precision: str, *,
                    peak_by_precision: dict = PEAK_BY_PRECISION,
                    bandwidth_bytes_s: float = BANDWIDTH_BYTES_S,
                    device: str = "rtx4090") -> float:
    rc = raw_counts(feat, precision, device)
    return max(rc["flops"] / peak_by_precision[precision],
               rc["hbm_bytes"] / bandwidth_bytes_s)


class RooflineBaseline:
    """E = P_avg * max(FLOPs/peak, bytes/BW); P_avg is the single fitted
    parameter (least squares through the origin; relative variant scales
    residuals by 1/y for the R1 companion)."""

    def __init__(self, *, peak_by_precision: dict = PEAK_BY_PRECISION,
                 bandwidth_bytes_s: float = BANDWIDTH_BYTES_S,
                 device: str = "rtx4090"):
        self.peak_by_precision = dict(peak_by_precision)
        self.bandwidth_bytes_s = float(bandwidth_bytes_s)
        self.device = device
        self.p_avg_w_: float | None = None

    def times(self, feats, precisions) -> np.ndarray:
        return np.array([
            roofline_time_s(f, p, peak_by_precision=self.peak_by_precision,
                            bandwidth_bytes_s=self.bandwidth_bytes_s,
                            device=self.device)
            for f, p in zip(feats, precisions)])

    def fit(self, feats, precisions, ys, *, relative: bool = False):
        t = self.times(feats, precisions)
        y = np.asarray(ys, float)
        if relative:
            r = t / y
            self.p_avg_w_ = float(np.sum(r) / np.sum(r * r))
        else:
            self.p_avg_w_ = float(np.dot(t, y) / np.dot(t, t))
        if self.p_avg_w_ <= 0:
            raise ValueError("roofline P_avg fit is non-positive")
        return self

    def predict(self, feats, precisions) -> np.ndarray:
        assert self.p_avg_w_ is not None, "fit first"
        return self.p_avg_w_ * self.times(feats, precisions)


class LayerwiseBaseline:
    """NNLS on raw structural counts (winner support, priors stripped):
    columns raw_macs, sram_words, hbm_words, n_launches, intercept."""

    COLUMN_NAMES = ("raw_macs", "sram_words", "hbm_words", "n_launches",
                    "intercept")

    def __init__(self, device: str = "rtx4090"):
        self.device = device
        self.coef_: np.ndarray | None = None

    def design(self, feats, precisions) -> np.ndarray:
        rows = []
        for f, p in zip(feats, precisions):
            rc = raw_counts(f, p, self.device)
            rows.append([rc["raw_macs"], rc["sram_words"], rc["hbm_words"],
                         float(f["n_launches"]), 1.0])
        return np.array(rows, float)

    def fit(self, feats, precisions, ys, *, relative: bool = False):
        A = self.design(feats, precisions)
        y = np.asarray(ys, float)
        if relative:
            w = 1.0 / y
            coef, _ = nnls(A * w[:, None], np.ones_like(y))
        else:
            coef, _ = nnls(A, y)
        self.coef_ = coef
        return self

    def predict(self, feats, precisions) -> np.ndarray:
        assert self.coef_ is not None, "fit first"
        return self.design(feats, precisions) @ self.coef_

    def coef_dict(self) -> dict:
        assert self.coef_ is not None
        return dict(zip(self.COLUMN_NAMES, map(float, self.coef_)))
