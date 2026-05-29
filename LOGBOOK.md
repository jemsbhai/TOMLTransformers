# Experimental Logbook — TOMLTransformers

**Project:** Transistor-level energy modeling of transformer inference (TOML extension)
**Working title:** Transistor-Level Energy Modeling for Transformer Inference across Encoder-only, Decoder-only, and Encoder-Decoder Architectures
**Target venue:** MLSys 2027 (primary); ISCA 2027 (alternative); NeurIPS 2026 (if timing permits)
**Researcher:** Muntaser Syed

**Conventions**
- Append-only. Past entries are never edited; corrections are dated addenda.
- Plan-before-run for confirmatory experiments (hypothesis/variables/protocol written before execution).
- Unit convention: 1 TO ≈ 1 fJ. MCER (memory-compute energy ratio) is dimensionless: (to_sram + to_hbm) / (to_mac + to_nonlinear).
- **All energy/MCER numbers are prior-weighted TO ratios until hardware calibration.** No confirmatory energy claim exists yet. Calibration targets: RTX 4090 (TSMC 4N, GDDR6X) primary; A100 (TSMC N7, HBM2E) cross-platform.

---

## Infrastructure (framework build, pre-experiment)

Not experiments; recorded for traceability. Through git commit `0c3947f`, 68 tests passing (`pytest -q`).

- **to_costs** — single source of truth for TO costs. Node-parameterized, physically grounded memory priors converted from cited vendor pJ/bit figures (HBM2E 6.0, HBM3E 4.05, GDDR6X 7.25, on-chip SRAM ~0.16 pJ/bit), per-device off-chip tier registry, provenance tags, and a `CALIBRATION_TARGETS` set. Tests assert the TO values reproduce the cited pJ/bit and that cited costs carry a source.
- **energy_model** — nested formulation family M0–M9 fit by non-negative least squares; split-feature variants fit corrections to the SRAM/HBM and MAC/nonlinear ratios; selection by held-out R²/MAPE and AIC/BIC.
- **architectures** — `configs` (9-model zoo across the three classes), `common` (TO-counting blocks emitting the granular feature breakdown), `attention` (standard vs FlashAttention, GQA/MQA, KV cache), `decoder` (prefill/decode assembly).

---

## EXP-001: Prefill vs decode MCER phase transition (exploratory, pre-calibration)

**Date:** 2026-05-29 (America/New_York)
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed (exploratory)
**Addendum note:** This entry is **backfilled**. The analysis was run during development as an architectural sanity check before this logbook existed. It is recorded in full for traceability and is labeled exploratory; it must not be reported as a confirmatory result. Future experiments follow plan-first discipline.

### Hypothesis
With physically grounded memory priors, a decoder-only model is compute-bound during prefill (MCER < 1, because each weight load is amortized over s tokens of compute) and memory-bound during decode (MCER ≫ 1, because every weight is re-read from off-chip for a single token plus the KV cache is read), with the decode/prefill MCER ratio of order s. FlashAttention's advantage over standard attention grows with sequence length due to O(s²) score-matrix off-chip traffic.

### Independent variables
- model ∈ {GPT-2, LLaMA-7B, Mistral-7B}
- sequence length ∈ {512, 2048}; and {2048, 8192, 16384} for the attention comparison
- attention kind ∈ {flash, standard}

### Dependent variables / metrics
- MCER (dimensionless): (to_sram + to_hbm) / (to_mac + to_nonlinear), from prior-weighted TO aggregates
- decode/prefill MCER ratio; standard/flash MCER ratio

### Control conditions
- device = rtx4090 (GDDR6X off-chip); precision = FP16; layer structure, priors held constant
- Implicit baseline: the M0 "calibrated FLOPs" framing (single total-TO term), which cannot represent the compute/memory split

### Protocol
1. `python scripts/exp001_mcer.py` (deterministic; writes `experiments/exp_001_mcer/results/mcer.json`)
2. Equivalent console view: `python -m tomltransformers.architectures.decoder`

### Environment
- **Hardware:** RTX 4090 laptop, 64 GB RAM. Compute path NOT exercised; this is a deterministic pure-Python calculation, GPU-independent.
- **Software:** see `experiments/exp_001_mcer/environment.json`
- **Git commit:** `0c3947f` (experiment code clean at this SHA; the logbook/scripts added afterward)
- **Seeds:** none — no stochastic components; fully deterministic

### Results
Prefill vs decode MCER (device rtx4090, FlashAttention, FP16):

| model | seq | prefill MCER | decode MCER | decode/prefill |
|-------|----:|-------------:|------------:|---------------:|
| GPT-2 | 512 | 0.2068 | 69.43 | 336 |
| LLaMA-7B | 512 | 0.1422 | 70.15 | 493 |
| Mistral-7B | 512 | 0.1399 | 69.17 | 494 |
| GPT-2 | 2048 | 0.0584 | 68.18 | 1167 |
| LLaMA-7B | 2048 | 0.0374 | 69.93 | 1872 |
| Mistral-7B | 2048 | 0.0353 | 66.26 | 1877 |

Standard vs FlashAttention prefill MCER (LLaMA-7B):

| seq | flash MCER | standard MCER | standard/flash |
|----:|-----------:|--------------:|---------------:|
| 2048 | 0.0374 | 0.0591 | 1.58 |
| 8192 | 0.0115 | 0.0889 | 7.75 |
| 16384 | 0.0075 | 0.1422 | 19.03 |

### Observations
- Decode MCER is nearly constant (66–70) across all models and sequence lengths. This is expected and analytically interpretable: decode memory ≈ N · (words/param) · cost_per_word and decode compute ≈ N · cost_per_mac, so MCER_decode ≈ (0.5 · 232000) / 1650 ≈ 70 at FP16 on GDDR6X, independent of model size. It confirms decode is bandwidth-bound at a per-parameter ratio set by the memory/compute cost gap, not by model scale.
- Prefill MCER scales ≈ 1/s (GPT-2: 0.2068 at 512 → 0.0584 at 2048, ratio 3.5 vs the 4× sequence increase; the gap is fixed embedding/lm_head cost), consistent with weight-load amortization.
- Standard-attention prefill MCER rises with s (0.059 → 0.089 → 0.142) while flash falls (0.037 → 0.0115 → 0.0075); standard/flash grows 1.6 → 7.8 → 19, the O(s²) score-matrix penalty emerging at long context.

### Interpretation
The qualitative phase transition (prefill compute-bound, decode memory-bound) is robust: it follows from arithmetic-intensity arguments and does not depend on the exact prior values. The quantitative MCER values DO depend on the priors, several of which are uncalibrated (`CALIBRATION_TARGETS`: softmax/norm op costs, effective per-word memory cost, kernel-launch counts, activation SRAM tiering). **These numbers cannot be reported as confirmatory.** Next: build the measurement harness (pynvml on the 4090), measure prefill/decode energy across a sequence-length grid, fit and select the energy-model family against measured energy, and recompute MCER from fitted coefficients (EXP-002+). Cross-validate TO-count stability on the A100.

### Artifacts
- Results: `experiments/exp_001_mcer/results/mcer.json`
- Environment: `experiments/exp_001_mcer/environment.json`
- Reproduce: `python scripts/exp001_mcer.py`
