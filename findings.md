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


### 2026-05-29 — Instrument layer brought up; three instruments live on Windows

**Status:** infrastructure, not a confirmatory result. No energy claim here.

**Key facts established by the instrument smoke test (`tomltransformers.measure`):**
- All three EXP-002 instruments run on the Windows + RTX 4090 (Ada) box: A (our
  20 Hz nvmlDeviceGetPowerUsage integration), B (our direct
  nvmlDeviceGetTotalEnergyConsumption read), and C (Zeus ZeusMonitor). This
  resolves the pre-registered Zeus-on-Windows prerequisite: C is AVAILABLE, not
  a fallback. Zeus needed `approx_instant_energy=True` so windows shorter than
  the energy-counter tick are approximated rather than returned as zero.
- B and C returned identical energy to 6 sig figs (201.624 J) on the same window,
  as they must, since both read the same Ada hardware energy accumulator. This is
  a correctness check on our B implementation against the peer-reviewed tool.

**Open issue to address in the runner (logged so it is not forgotten):** on an
uncontrolled smoke window (no idle subtraction, no warmup-excluded timing, only
19 power samples), instrument A read ~27% HIGH versus B (255.96 J vs 201.62 J).
Direction and cause are known, not noise: sampled instantaneous board power
integrates high under sustained load, and a coarse trapezoid over few samples is
sensitive to sample placement. This is far above the pre-registered 5% median
agreement target. It does not invalidate the layer (which is why the dual-our-
instrument design exists: to surface exactly this). The runner must close the gap
with thermal-settled longer windows, a higher effective sample count, and idle-
baseline subtraction, then re-measure A-vs-B agreement under controlled
conditions before any energy is reported. The 27% is an UPPER BOUND from an
uncontrolled window, not a result.


### 2026-05-29 — Controlled runner narrows A-vs-B to ~14%, still above 5% target

**Status:** infrastructure / open measurement question. No energy claim here.

**Result of the runner's controlled path** (warmup-excluded, thermal-settled,
P_idle*dt subtracted uniformly across A/B/C) on a 3.2 s sustained-matmul window:
A = 428.5 J, B = 497.4 J, C = 497.4 J. B and C identical (same hardware
accumulator, as expected). A-vs-B agreement = 13.8%, and A was the only
instrument flagged CV-exceeded (>5% across 3 repeats).

**Reading:** the controls did help (uncontrolled 27% -> controlled 14%), but 14%
on a long, clean, idle-subtracted window is ~3x the pre-registered 5% median
target and can no longer be attributed to a too-short window. Instrument A
(sampled-power integration) reads systematically LOW vs the B/C hardware energy
counter and is the noisier instrument. Leading hypothesis: A's background
sampling thread is starved under a GIL-bound, driver-contended saturated GPU, so
nvmlDeviceGetPowerUsage samples are delayed/unevenly spaced, biasing and
destabilizing the trapezoid integral. Alternative: genuine bias between smoothed
instantaneous board power and true integrated energy.

**Action (before any sweep is built on A):** run a throwaway diagnostic that dumps
A's raw (timestamp, power) samples on one long window to distinguish sampler
starvation from real bias. Outcomes: raise A's sampling rate / move sampling off
the GIL, OR designate B (hardware counter) as the PRIMARY reported instrument
with A as an independent sanity check. This decision will be recorded before
EXP-002 energy is reported. Not tuning test tolerances to mask the gap.

**RESOLVED 2026-05-29 (same day) by the diagnostic** (scripts/diag_instrument_a.py;
raw data experiments/exp_002_size_sweep/diagnostics/instrument_a.json):

The cause is UNDERSAMPLING, not sampler starvation, not a short integration span,
not a GIL problem. On a sustained-matmul window swept across sampling rates, the
A-vs-B relative difference fell monotonically with rate: 24.2% @20Hz, 7.9% @50Hz,
5.5% @100Hz, 7.4% @200Hz. Two candidate causes were explicitly REJECTED by the
data: (a) sample-window coverage was ~1.0 at every rate (so A was not integrating
a truncated span; the [first,last]-sample fix was unnecessary), and (b) the
external nvidia-smi logger did NOT agree with B either (smi-vs-B 19-51%, never
converging), so the in-process sampler is not the culprit and nvidia-smi is unfit
as a reference. 20 Hz (the prior-TOML rate) simply undersamples a fluctuating
power trace and integrates low.

**Decisions recorded:**
1. Instrument A sampling rate raised 20 -> 100 Hz (instruments.py and runner.py
   defaults, and configs/exp_002.yaml, amended with an in-file dated note). At
   100 Hz A meets the pre-registered 5% agreement target; 200 Hz adds nothing.
2. A and B both remain PRIMARY (no demotion): they agree to ~5% at 100 Hz, so the
   dual-our-instrument agreement claim stands. nvidia-smi is NOT a measurement
   path; B (hardware counter) is the reference.
3. Added measurement.min_window_s = 2.0. The GPT-2 prefill probe (40 forwards at
   s=1024) produced SUB-SECOND windows on which even B was self-inconsistent
   (B = 94/38/47/35 J across the rate runs, n~9 samples). Lesson larger than the
   rate fix: a window must be long enough that the hardware counter ITSELF is
   stable. The sweep's workload builders must loop per-execution work enough times
   to exceed ~2 s for small/fast models.

**Still to confirm (next step):** that 100 Hz holds at <=5% MEDIAN across the
runner's 5-repeat controlled path (the diagnostic was one window per rate, a
direction, not a distribution). The matmul window itself was clean and stable
(B ~520-540 J across rates, n=59-299), which is the regime the sweep must keep
workloads in.
