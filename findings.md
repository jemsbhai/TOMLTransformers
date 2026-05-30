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


### 2026-05-29 — Prior-series measurement protocol (context for the 100 Hz change)

**Status:** documentation of prior work; basis for a methods contribution. Not a
result of this project.

Reading the code of all earlier repos in the TOML series confirms a single,
consistent GPU measurement protocol across every prior paper:
- FLAIRS-39 origin (`toml/benchmark_gpu_rigorous.py`): NVML power at 50 ms = 20 Hz;
  idle baseline; delta_power = load - idle; delta_energy = delta_power * duration;
  warmup before timing; min 3 s window via auto-scaled iteration count
  (num_batches = max(50 or 100, ceil(min_duration / time_per_batch))).
- Cloud / IC2E (`TOMLCloud/shared/power_monitor.py`): same 50 ms = 20 Hz sampling,
  same delta-power-over-idle, plus temperature/clock/util telemetry and 10 s
  windowed stats for thermal profiling.
- Signals / MLSP (`TOMLSignals/shared/harness.py`): same 50 ms = 20 Hz, thermal
  settle to +/-1 C over 5 s, 3 s idle baseline, continuous >=10 s measurement
  loop, energy_per_call = delta_energy / iterations; batch-size auto-calibration
  to a 1 ms/call target.

Two things follow. First, our controlled runner is in-family with this protocol
(same idle subtraction, thermal settle, warmup, and a min-window floor that
generalizes their min-duration auto-scaling), and our 4 s floor is slightly more
conservative than their 3 s. Second, and more important: EVERY prior TOML GPU
number was 20 Hz power-integration with NO hardware energy counter. Our
instrument-A diagnostic showed 20 Hz integrates ~14-24% LOW vs the on-die counter
(nvmlDeviceGetTotalEnergyConsumption). So the GPU energy figures across the whole
published series carry this systematic underestimate. This grounds a concrete
methods contribution for the transformers paper: prior work (our own included)
sampled at 20 Hz; we show this underestimates dynamic GPU energy relative to the
hardware counter, and we adopt 100 Hz plus the counter (instrument B) as primary,
with Zeus (C) as an independent cross-check. Traceable to the prior repos' code,
not speculation.

NOTE: This does not retroactively invalidate the prior papers' CONCLUSIONS, which
rest on cross-architecture ratios and rankings (largely invariant to a uniform
~15-20% scale bias on the GPU channel). It does mean absolute GPU joule figures
there are low, and the transformers paper should measure absolute energy correctly
from the start.


### 2026-05-29 — Controlled A-vs-B agreement confirmed on a real transformer (100 Hz)

**Status:** measurement-infrastructure validation. Not a calibrated energy result.

The GPU integration test (`test_gpu_prefill_through_runner_agreement`) ran a
shape-faithful decoder prefill (probe config: 12 layers, d=1024, s=512, FP16)
through the full controlled runner at 100 Hz, with measure_until_floor sizing the
window to ~4 s and 5-repeat... (repeats=3 in the test) statistics. It passed the
pre-registered checks: instruments A, B, C all present; window cleared the floor
(short_window False); and A-vs-B agreement under the 12% test gate. This closes
the "still to confirm" item from the instrument-A diagnostic entry above: the
100 Hz fix holds through the controlled path on a real transformer workload, not
just a single matmul window.

CAVEAT (honest scope): the test asserts <=12% to stay robust against laptop-GPU
run-to-run noise, so "passed" means <=12%, NOT "=5%". The exact A-vs-B figure
printed in the test's [workload] line was not captured this run; the true median
agreement will be recorded from the first real EXP-002 measurement, where it is
the headline cross-instrument check. Until that number is logged, treat "~5% at
100 Hz" as supported-by-the-diagnostic but not yet confirmed at full repeat count
on transformer workloads.

**UPDATE (same day, number now captured):** the [workload] line WAS captured on a
repeat run: A = 623.54 J, B = 690.73 J, C = 690.73 J; A-vs-B = 9.73%, B-vs-C =
0.0%; inner_iters = 1408, wall = 4.64 s, short_window False, A flagged CV-exceeded.
So on a REAL transformer prefill through the controlled path, A-vs-B is ~10%, NOT
the ~5% the matmul diagnostic suggested. This corrects the optimistic reading
above and revises decision #2 from the diagnostic entry (see correction below).
The matmul trace was smoother/more steady-state than a real transformer forward
(which has more kernel-launch boundaries and power fluctuation), so 5.5% on matmul
did not transfer. B and C remain bit-identical (same hardware counter), and A is
consistently the low, noisy instrument.


### 2026-05-29 — CORRECTION: instrument B is primary; A is a ~10% sanity check

**Status:** measurement-policy decision, correcting an earlier over-statement.

The diagnostic entry above recorded "A and B both PRIMARY; they agree to ~5% at
100 Hz." The real-transformer measurement (A-vs-B = 9.73%, A again low and
CV-flagged) shows that was over-optimistic. Corrected policy, on the consistent
evidence across every test run to date:

- **B (nvmlDeviceGetTotalEnergyConsumption, the on-die counter) is the PRIMARY
  reported instrument.** It is stable across repeats, model-independent in its
  reliability, and bit-identical to Zeus (C) on every shared window (B-vs-C =
  0.0%), which is strong corroboration from an independent peer-reviewed tool.
- **A (our 20->100 Hz power integration) is a SANITY-CHECK / cross-method
  instrument, not co-primary.** It agrees with B to within ~10% on real
  transformer workloads after all controls, and is consistently the low, noisier
  reading (CV-flagged). ~10% is a reasonable agreement for sampled-power vs a
  hardware accumulator, and is reported as such, NOT claimed as 5%.
- **C (Zeus) remains the independent cross-check** and, because it reads the same
  counter as B, functions as a third-party validation that our B integration is
  correct.

What this means for EXP-002 reporting: energy figures are B, with A reported
alongside as an independent method and the A-vs-B agreement (~10%) stated honestly
as a measurement-robustness check. The pre-registered 5% agreement TARGET in
configs/exp_002.yaml was not met by A on transformer workloads; that target should
be read as aspirational for the sampled-power method, with the hardware counter
(B/C agreement ~0%) carrying the actual rigor. This will be reflected when the
first real EXP-002 results are written. Not tuning anything to hide the 10%.


### 2026-05-29 — Decode regime is noisier than prefill (sweep-design implication)

**Status:** hardware observation from the workload GPU integration tests. Not a
calibrated energy result; magnitudes are uncalibrated and not for citation.

First hardware look at decode vs prefill through the controlled runner (probe
config: 12L, d=1024, FP16, RTX 4090 Laptop), both sized to a ~4 s window by
measure_until_floor:
- prefill (s=512): inner_iters=1367, wall 4.72 s, B=668.03 J, A=614.66 J,
  A-vs-B=7.99%, B-vs-C=0.0%; only A CV-flagged.
- decode growing (ctx=256, K=32): inner_iters=38, wall 4.11 s, B=161.59 J,
  A=138.08 J, A-vs-B=14.55%, B-vs-C=0.0%; ALL THREE (A, B, C) CV-flagged.

Two robust qualitative facts (not the numbers, which are uncalibrated):
1. Decode energy is NOISIER run-to-run than prefill, even on the hardware counter
   B/C (which were never CV-flagged on prefill but are on decode). Physical
   reading: decode is memory-latency-bound with tiny per-step compute, so the
   power trace is lower and choppier (GPU waiting on memory, not saturated), and
   far fewer executions (38 vs 1367) are averaged into each repeat. This is a
   property of the regime, not a measurement defect.
2. The sampled-power instrument A diverges MORE from the counter on decode
   (14.55% vs 7.99% on prefill), consistent with choppy low-power traces being
   harder to integrate from 100 Hz samples. Reinforces B-as-primary: A degrades
   exactly where the workload is hardest to sample.

**Sweep-design implications (to apply when building the driver):**
- Decode points likely need MORE repeats than prefill to bring the B/C CV under
  the gate (prefill's 3-5 repeats were enough; decode may need more). Do not
  silence the CV flag; raise repeats or widen the window for decode.
- A's agreement gate must be regime-aware: ~8% is normal for prefill, ~15% for
  decode. A single tight threshold across both phases would false-flag decode.
- Energy-magnitude sanity (NOT a result): per-execution B is ~0.49 J prefill vs
  ~4.3 J decode, and one decode execution does prefill-to-256 PLUS 32 growing
  steps, so decode-per-exec > prefill-per-exec is the expected ordering. Nothing
  looks broken; full interpretation waits on calibration.

Also a second prefill A-vs-B data point (7.99%) alongside the earlier 9.73%: the
real-transformer prefill agreement is variable in an ~8-10% band, not a fixed
value. The "~10%" characterization above covers this; no correction needed.

DECODE WORKLOAD CORRECTNESS (separate from the energy numbers) is established by
CPU structural tests: the incremental KV cache grows exactly one position per
step (10 -> 11 -> 12), decode_step processes a single token, cached K/V carry
n_kv_heads under GQA, and both growing/fixed_step modes execute. The memory-bound
signature (cache READ scaling with context) is therefore built correctly; whether
it produces MCER >> 1 in JOULES is the calibration question EXP-002 will answer.
