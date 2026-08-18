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

A100 roofline constants (added 2026-08-17 for a100_amendment.md section 10,
decision D7, BEFORE any A100 baseline is fitted; NVIDIA A100-SXM4-40GB,
Ampere GA100, the EXP-002 A100 phase device):
- 108 SMs x 64 FP32 CUDA cores = 6,912 cores; boost 1,410 MHz (whitepaper).
- Peak FP32 = 2 x 6,912 x 1.410e9 = 19.49 TFLOP/s (formula stated; the
  datasheet rounds to 19.5). fp32 GEMMs run on the CUDA cores under
  PyTorch's default (TF32 off for matmul; the runner sets nothing).
- Peak FP16 = dense tensor-core rate: 108 SMs x 1,024 FMA/clk/SM x 2 x
  1.410e9 = 311.9 TFLOP/s (whitepaper per-SM rate; the datasheet rounds to
  312; the 624 sparse figure does not apply). Unlike the 4090's assumed 2x,
  this is the vendor figure, and it is 16x the FP32 peak: the two datapaths
  differ, which the roofline encodes and the TO priors do not (LOGBOOK
  2026-08-17).
- Memory bandwidth 1,555 GB/s (40 GB HBM2 per the datasheet; the device
  registry tier is labeled hbm2e, see to_costs.py); TDP 400 W.
- The A100 campaign's maximum sustained median SM clock is the rated boost
  (1,410 MHz, validation report), so no measured-clock sensitivity line is
  needed on that platform.

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
    "nvidia_a100_datasheet": "NVIDIA, 'NVIDIA A100 Tensor Core GPU' datasheet "
                             "(SXM4 and PCIe form factors, June 2021): FP32 19.5 "
                             "TFLOPS; FP16 Tensor Core 312 TFLOPS (624 with "
                             "sparsity); A100 40GB SXM: 40 GB HBM2, 1,555 GB/s, "
                             "400 W max TDP.",
    "nvidia_a100_whitepaper": "NVIDIA, 'NVIDIA A100 Tensor Core GPU "
                              "Architecture' whitepaper v1.0 (2020): A100 = "
                              "GA100 with 108 SMs, 64 FP32 CUDA cores per SM "
                              "(6,912), four third-generation Tensor Cores per "
                              "SM at 256 FP16/FP32 FMA per clock each (1,024 "
                              "dense FMA/clk/SM); GPU boost clock 1,410 MHz.",
    "measured_clock_a100": "a100/energy.jsonl (frozen EXP-002 A100 dataset): "
                           "median SM clocks bimodal 1095/1410 MHz across 98 "
                           "points; maximum 1410 MHz equals the rated boost.",
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

# --- NVIDIA A100-SXM4-40GB (Ampere GA100); see module docstring -------------
A100_SMS = 108
A100_CORES_PER_SM = 64
A100_CORES = A100_SMS * A100_CORES_PER_SM                          # 6,912
A100_BOOST_HZ = 1.410e9
A100_PEAK_FP32_FLOPS_S = 2.0 * A100_CORES * A100_BOOST_HZ          # 19.49e12
A100_TENSOR_FMA_PER_CLK_PER_SM = 1_024
A100_PEAK_FP16_FLOPS_S = (2.0 * A100_SMS * A100_TENSOR_FMA_PER_CLK_PER_SM
                          * A100_BOOST_HZ)                         # 311.9e12
A100_PEAK_BY_PRECISION = {"fp32": A100_PEAK_FP32_FLOPS_S,
                          "fp16": A100_PEAK_FP16_FLOPS_S}
A100_BANDWIDTH_BYTES_S = 1555.0e9

# Power envelopes for the fitted-P_avg diagnostic (a100_amendment.md section
# 10): 4090 Laptop GPU 150 W TGP configuration (techspot_4090m); A100 SXM
# 400 W max TDP (nvidia_a100_datasheet).
TDP_W_BY_DEVICE = {"rtx4090": 150.0, "a100": 400.0}

_ROOFLINE_BY_DEVICE = {
    "rtx4090": {"peak_by_precision": PEAK_BY_PRECISION,
                "bandwidth_bytes_s": BANDWIDTH_BYTES_S},
    "a100": {"peak_by_precision": A100_PEAK_BY_PRECISION,
             "bandwidth_bytes_s": A100_BANDWIDTH_BYTES_S},
}


def roofline_constants(device: str) -> dict:
    """Roofline keyword arguments (peak_by_precision, bandwidth_bytes_s) for
    a device registry name; the rtx4090 entry is the module default set."""
    try:
        c = _ROOFLINE_BY_DEVICE[device]
    except KeyError:
        raise KeyError(f"no roofline constants for device {device!r}; known: "
                       f"{sorted(_ROOFLINE_BY_DEVICE)}") from None
    return {"peak_by_precision": dict(c["peak_by_precision"]),
            "bandwidth_bytes_s": float(c["bandwidth_bytes_s"])}


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
