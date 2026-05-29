# Findings

## Curated Summary

**Status: no confirmatory results yet.** The framework is built and validated by
unit tests, and one exploratory analysis (EXP-001) has been run. Every energy and
MCER number below is a prior-weighted transistor-operation ratio, not measured
energy, and is pending hardware calibration on the RTX 4090 and A100. Nothing in
this section may be cited as a confirmed result until calibrated.

**Central thesis (exploratory support).** A decoder-only transformer exhibits a
phase transition in the memory-compute energy ratio (MCER) between its two
inference phases. Prefill, which processes all prompt tokens in one pass and
amortizes each off-chip weight load over the whole sequence, is compute-bound
(MCER < 1). Decode, which re-reads every weight from off-chip memory for a single
token and reads back the growing KV cache, is memory-bound (MCER ≫ 1). On
physically grounded priors (GDDR6X at 7.25 pJ/bit), the exploratory decode MCER
sits near 70 essentially independent of model size, while prefill MCER scales as
roughly 1/s; the decode/prefill ratio is therefore of order the sequence length.
This is the distinction conventional FLOP/MAC counting cannot make, since the two
phases perform near-identical operation mixes. The qualitative direction is robust
to the priors (it follows from arithmetic intensity); the magnitudes are not, and
are the object of the planned calibration.

**Secondary (exploratory).** FlashAttention's benefit over standard attention is
negligible at short context and grows with sequence length: for LLaMA-7B the
prefill MCER ratio (standard/flash) rises from ~1.6 at s = 2048 to ~19 at
s = 16384, reflecting the O(s²) off-chip traffic of a materialized score matrix
that FlashAttention avoids by tiling in SRAM.

---

## Raw Findings Log

### 2026-05-29 — EXP-001: Prefill/decode MCER phase transition (exploratory)

**Key result:** Prefill MCER ≤ 0.21 and decode MCER ≈ 66–70 across
GPT-2 / LLaMA-7B / Mistral-7B; decode/prefill ratio 336–1877, increasing with
sequence length. Standard-vs-Flash prefill MCER ratio grows to ~19× at s = 16384.

**Details (device rtx4090, FP16; dimensionless MCER; baseline = M0/FLOPs framing):**
- LLaMA-7B, s=2048: prefill MCER 0.0374, decode MCER 69.93, ratio 1872
- GPT-2, s=512: prefill MCER 0.2068, decode MCER 69.43, ratio 336
- LLaMA-7B attention, s=16384: flash MCER 0.0075 vs standard MCER 0.1422 (19.03×)

**Statistical tests:** N/A. These are deterministic, single-valued computations
(no sampling, no seeds), so there is no sampling uncertainty to report. Their
*validity as energy estimates* is the open question, to be settled by calibration,
not by a statistical test on this calculation.

**Notes:** EXPLORATORY and PRE-CALIBRATION. Prior-weighted TO ratios, not joules.
Priors include uncalibrated estimates (softmax/norm op costs, effective per-word
memory cost, launch counts, activation tiering). The near-constant decode
MCER ≈ 70 is model-independent by construction: at FP16 on GDDR6X it equals
(0.5 word/param × 232000 fJ/word) / (1650 fJ/MAC) ≈ 70, a per-parameter
memory/compute ratio. Cannot be reported as confirmatory. Supersedes the
prototype's MCER ≈ 2.0, which used an HBM prior ~20× too low.
