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


### 2026-05-29 — Encoder workload runs on hardware; B-vs-C is ~1%, not always 0

**Status:** measurement-infrastructure validation + a small correction to a
stated invariant. Not a calibrated energy result.

The encoder GPU integration test (`test_gpu_encode_through_runner`, probe config
BERT-large-shaped: 24L, d=1024, s=512, FP16, bidirectional) ran through the
controlled runner: inner_iters=638, wall 4.03 s, B=599.71 J, A=543.97 J,
C=606.98 J; A-vs-B=9.29%, A-vs-C=10.38%, B-vs-C=1.2%; only A CV-flagged; window
cleared the 4 s floor. Encode behaves like prefill (compute-bound, steady power,
A in the ~8-10% band, only A noisy) — as expected, since a single bidirectional
pass amortizes weights over the sequence just as prefill does.

**Correction to a stated invariant:** earlier entries said B and C are
"bit-identical (B-vs-C = 0.0%)". Here B-vs-C = 1.2%. The accurate statement is:
B and C read the SAME on-die counter but bracket their own measurement windows
independently, so their windows are not bitwise-coincident; when they differ by a
few ms they can diverge ~1% on a high-energy window. On the earlier prefill/decode
runs the brackets happened to coincide (0.0%); they need not. So: B and C agree to
~1% (often exactly), reading the same counter — NOT "always identical". This does
not change the B-primary / C-cross-check policy; it slightly softens the
"bit-identical" phrasing to "~1%, same counter".

ENCODER WORKLOAD CORRECTNESS is established by CPU structural tests: text configs
use a token-embedding gather (no patch_proj), vision configs use a patch
projection PATCH_DIM->d_model (no token embedding), ViT fixes s = num_patches + 1
and ignores seq_len, QKV width is GQA-aware, and the classifier head emits
num_classes logits (vision) or d_model (text, num_classes=0) on the pooled token.
Matches encoder.py (bidirectional, no KV cache, no prefill/decode split).


### 2026-07-18 - Production sweep chunk 1 (85/296): measurement quality at scale

**Status:** chunk-level data-quality observations from the first production
chunk of the EXP-002 sweep (commit ba3347e, 2026-07-17 16:54 to 2026-07-18
00:58 EDT, `--max-hours 8`). Source: `experiments/exp_002_size_sweep/energy.jsonl`
analyzed by `scripts/validate_exp002.py`; full distributions in
`validation_report.{txt,json}` committed beside the data. Instrument B
(hardware counter) unless stated. Magnitudes remain uncalibrated (no fit yet)
and are not citable results.

Coverage so far: decoder-only class only, flash attention only, both
precisions. DistilGPT2 / GPT-2 / GPT-2-medium / GPT-2-large complete (20
points each: 5 prefill seq_lens x 2 precisions + 5 decode contexts x 2
precisions); GPT-2-XL partial (its fp16 prefill series, 5 points). Encoder,
enc-dec, and eager-attention points are still unmeasured, so cross-class
statements wait.

Outcomes: 85/85 ok. Zero failures, zero OOM skips, zero short windows, zero
superseded records; repeat protocol held exactly (5 forward, 10 decode-like).
Pace: median 292 s/point (p25 254, p75 437, max 628), projecting ~17 h for
the remaining 211 points.

**1. A-vs-B agreement at production scale is tighter than the probe-era
bands.** Forward class: median 5.45%, p75 5.94%, max 9.58% (probe band was
~8-10%). Decode-like: median 4.20%, p75 4.89%, max 10.86% (probes suggested
~13-15%). At the median, decode A-B is no longer categorically worse than
prefill; only its tail is heavier. Plausible drivers: the production protocol
(measure_until_floor to a >= 4 s window, regime-aware repeats, median
statistics) versus single probe runs; also note chunk 1's decode class
includes fp32 points, whose higher and steadier power is easier for A to
sample, so the fp16-only comparison to the probes is not exact. The per-class
medians straddle the pre-registered <= 5% median A-B target (4.20% decode,
5.45% forward); final assessment waits for the full grid including
encoder/enc-dec/eager points.

**2. B-vs-C:** median 0.000%, max 1.695% across 85 points. Consistent with
the corrected invariant (same counter, independent window brackets, ~1%
worst case).

**3. Regime-aware repeats worked.** The 2026-05-29 decode finding predicted
decode would need more repeats to bring B under the CV gate. At 10 repeats,
B is CV-flagged on only 3/40 decode points (and 1/45 prefill); A remains
flagged on 84/85 points, as expected for the sampled-power method. CV(B):
forward median 1.56% (max 5.92%); decode median 2.58% with one outlier
(next item).

**4. The noisiest regime is smallest-context decode.** GPT-2-medium fp16
ctx128 decode is the worst point on three independent views at once: CV(B)
16.3%, A-B 10.86%, and the lowest median SM clock in the chunk (1110 MHz).
The ctx128 decode points of the other models also sit at the low-clock end
(DistilGPT2 1215, GPT-2 1230 MHz). Physical reading: tiny-context decode is
launch/latency-bound at low power with choppy clocks. Per the settled design
the CV gate FLAGS and never retries; the point stands, flagged, and the fit
can weight it or sensitivity-test it.

**5. Window floor vs inner_iters.** Decode inner_iters spans 2-35 (median
8); GPT-2-large fp32 ctx4096 decode sits at inner_iters=2 while still
meeting the wall floor (chunk-wide minimum wall time 4.00 s). The 4 s wall
window, not inner_iters, is the invariant, and it held on all 85 points.
The contamination fraction of the naive decode per-unit is largest exactly
at these big-prefill points (one execution = prefill-to-4096 plus 64 steps),
reinforcing that decode per-token is ONLY the fit-time prefill-subtracted
derivation. GPT-2-XL long-context decode may reach inner_iters=1; that is
still valid under the wall floor.

**6. First measured evidence on the precision axis (Fork 2).** 40 fp32/fp16
matched-shape pairs, zero inversions (fp32 > fp16 everywhere). Forward-phase
per-unit ratio: median 3.26, range 2.04-3.90 (n=20). A pure byte-doubling
picture would suggest ~2x; ratios approaching 4x are plausibly tensor-core
fp16 versus non-tensor-core fp32 execution paths, but confirming the
mechanism is out of scope here. The robust part today is the monotone
ordering and that the multiplier is not a trivial constant, which is exactly
why Fork 2 fits it rather than assuming it.

**7. Scaling sanity:** all 9 available (model, precision) prefill series are
monotone in per-forward energy vs seq_len, zero violations.

**8. Thermal and clock state varies between points and is logged.** Settle
temps 47-68 C; peaks reach 83 C on GPT-2 / GPT-2-medium fp32 prefill. Decode
points downclock to 1110-1605 MHz median SM clock, a workload-induced
memory-latency behavior distinct from thermal throttling. The idle baseline
drifted 3.69-7.31 W across the chunk, which is the drift the per-point idle
baseline design exists to absorb.

No systematic anomaly found. The sweep continues unchanged.


### 2026-07-18 - Chunk 2 (158/296): patterns hold as encoder and enc-dec classes arrive

**Status:** chunk-level QC, same provenance discipline as the chunk 1 entry
(data spans commits ba3347e and f5ab202; validator reports recommitted beside
the data). Uncalibrated; not citable results.

- Chunk 2: 73 new points, 85 skipped by resume, all ok. Still zero failed /
  OOM / short-window across all 158 points, zero superseded records. Coverage
  now: decoder-only complete (100), encoder-only text complete (DistilBERT /
  BERT-base / BERT-large, 30), ViT partial (4), T5-small partial (24). All
  158 points are flash; the eager attention_compare block sits later in the
  grid order and is still unmeasured.
- Encode and decoder_prefill enter the distributions TIGHTER than their
  probes, completing the chunk-1 picture: pooled forward A-B median 5.48%
  with max 9.98% at the smallest fp16 encode points (the encode probe read
  9.29% at s512), and all five decoder_prefill points sit at or below 8.75%
  versus the 13.2% probe. Decode-like A-B median 4.15%. B-C max 1.92%.
- The noisy regime is a family, not a one-off: all four CV(B) points above
  7.5% (16.3, 11.9, 9.9, 9.5%) are fp16 small-target decode (ctx128 or
  ctx512), now including GPT-2-XL and T5-small's target-128 sub-sweep point.
  Their fp32 siblings are quiet. All flagged and kept per the no-retry
  policy.
- inner_iters=1 occurred as anticipated (GPT-2-XL fp32 ctx4096 decode) with
  the 4 s wall floor met (chunk-wide minimum wall 4.00 s). The decode-like
  inner_iters maximum of 1831 is a decoder_prefill point, which packs many
  forwards per window exactly like prefill; expected, not an anomaly.
- Scaling/physics: 18/18 monotone prefill+encode series; 72 fp32/fp16
  matched pairs, zero inversions, forward ratio median 3.24 (max 4.07, first
  encoder pairs included). No ViT invariance data yet (only one point per
  precision group so far).
- Remaining 138 points skew to enc-dec, ViT, and the eager block, so the
  11 h median-pace projection is optimistic; plan on ~12-15 h. The BART-large
  target-4096 decode points ahead are the first realistic exercise of the
  still-untested OOM skip-and-log path; oom_skipped records there would be
  the policy working, not a failure.

Sweep continues unchanged.


### 2026-07-20 - EXP-002 measurement campaign complete: full-grid QC at 296/296

**Status:** consolidated QC for chunks 3-6 plus the full-grid validator pass
(2026-07-20 09:59 EDT over all 296 points; reports committed beside the
data). Uncalibrated magnitudes; the confirmatory analyses (fit,
extrapolation, baselines) have NOT been run. This entry is the
measurement-quality record for the paper.

**Campaign ledger.** Six resumable chunks, one grid, no manual intervention
beyond launch and commit: 85 (ba3347e) + 73 (f5ab202) + 81 (d434da2) + 30
(99024e8) + 24 (33592b3) + 3 (b597262) = 296. The three 8 h chunks measured
8.03-8.08 h each; chunks 4-6 ran within 3 h, 2 h, and 0.36 h budgets, so
the campaign cost ~29.5 h wall time against the ~30 h pre-launch estimate.
Final pace: median 297 s/point (p25 265, p75 519, max 1475).

**Perfect outcome record.** 296/296 ok; zero failed, zero OOM-skipped, zero
short-window, zero superseded records; repeat protocol exact (134 forward
at 5 repeats, 162 decode-like at 10); resume never re-measured a completed
point across five restarts. Consequence worth recording honestly: the OOM
skip-and-log path never fired, so every pre-registered point fit in 16 GB,
and that path remains UNEXERCISED on real hardware going into the A100
work.

**Two grid facts discovered during QC** (both expander behavior, both
sensible, neither previously written down):
- ViT entries are deduped to one point per precision (2 per model, 4
  total) because the vision workload ignores seq_len; the planned ViT
  seq-len-invariance replicate check is therefore vacuous by design, not
  data-starved.
- The attention_compare block adds 10 points (DistilGPT2 and GPT-2: eager
  fp16 at s512/1024/2048/4096, plus flash fp16 s4096, which the main grid
  lacks), bringing those models to 25 points each. All eager data landed
  in the final chunks; eager-vs-flash energy ratios are a fit-time
  analysis, not repeated here.

**Full-grid instrument story (the paper's measurement-quality numbers):**
- A-B: forward median 5.53% (p25 4.92, p75 6.04, max 9.98); decode-like
  median 4.03% (p25 3.39, p75 4.67, max 10.86). Per-class medians straddle
  the pre-registered <= 5% median target: decode-like meets it, forward is
  half a point over. The pooled-median verdict will be computed exactly in
  the fit-time analysis rather than eyeballed here.
- B-C: median 0.000%, p75 0.241%, max 1.922% over 296 points.
- CV(B): forward median 1.54% (max 6.24%); decode-like median 2.29%. The
  noisy family froze at five members, all small-target decode (four fp16,
  one fp32 at 7.9%); BART-large's small-target decode points did NOT join
  it. B was CV-flagged on 18/296 points, A on 293/296.
- Window floor: minimum wall 4.00 s across all 296 points; the three
  low-inner_iters WARNs (2, 2, 1) are all large fp32 decoder decode where
  a single execution approaches the floor by itself; valid under the floor
  invariant.
- Thermal/clocks: settle 44-68 C, peak 83 C (GPT-2 / GPT-2-medium fp32
  prefill); small fp16 decode/encode points downclock to 1110-1605 MHz;
  idle baseline spanned 3.52-7.31 W across the campaign, absorbed by the
  per-point baseline design.

**Physics sanity, final:** 26/26 prefill+encode series monotone in seq_len
(including both eager series through s4096); 143/143 fp32/fp16 matched
pairs correctly ordered; forward-phase precision ratio median 3.24x, range
2.04-4.11x (n=62); contamination flags exact (decode 122 True, all other
phases False).

**Data freeze.** energy.jsonl (296 records across 6 commits) is now the
frozen EXP-002 RTX 4090 dataset. Any future re-measurement appends under a
new commit and stays distinguishable via last-write-wins provenance. Next
per the pre-registration: the M0-M9 fit against instrument B (Step 2),
then the pre-registered extrapolation split and baseline bake-off (Step
3), then the representativeness check (Step 4).


### 2026-07-24 - EXP-002 confirmatory fit results (pre-registered), plus labeled exploratory R1

**Status:** CONFIRMATORY results per fit_plan.md, recorded exactly as they
fell, produced by scripts/fit_exp002.py at commit aac684f (artifacts in
experiments/exp_002_size_sweep/fit/). Deterministic: the confirmatory
sections reproduced byte-identical across the 50a15b3 and aac684f runs,
which doubles as a reproducibility check. The exploratory subsection is
post-hoc, approved 2026-07-24 before its run, labeled everywhere, and never
presented as pre-registered. Baselines and significance tests (plan section
8) have NOT been run yet.

**Pre-registered verdicts:**
1. Pooled A-B median 4.80% over all 296 points: the <= 5% target is MET
   (forward 5.53%, decode-like 4.03%). The 100 Hz plus hardware-counter
   methods contribution now carries a met pre-registered target.
2. Model selection: M8_split_dispatch wins by AIC (R2_test 0.987, R2_train
   0.973). M9 lands at exactly M8 plus the information-criterion penalty,
   the degeneration predicted in writing before any fit (n_fused_steps is 0
   by construction). The launch term is retained by selection at ~0.93 mJ
   per launch (absolute fit), carrying the launch-bound small-context
   decode floor the QC data showed.
3. Extrapolation: E2 broad (PRIMARY) FAILS its pre-registered band, pooled
   MAPE 50.37% vs 25% (decoder 33.1%, enc-dec 61.0%, encoder 42.5%). E1
   strict-literal PASSES at 14.34%. Both verdicts stand as registered.
4. Calibrated MCER (absolute estimator, winner refit on all 296): decode
   12.6-13.1 vs forward 3.0-4.1 medians. The prefill-vs-decode phase
   transition is confirmed in calibrated joules, no longer prior-weighted
   TO ratios.
5. Clean per-token decode: positive and monotone in context on all 74
   measured-subtracted rows. Examples (fp16): DistilGPT2 87 -> 202 mJ/token
   over ctx 128 -> 4096; GPT-2-XL 1.52 -> 1.90 J/token over ctx 128 -> 2048.

**Central methods finding: the pre-registered estimator is scale-blind on
this data.** R2_test 0.987 coexists with MAPE_test 88% on the same held-out
set: absolute NNLS over a target spanning roughly four decades is dominated
by the largest-energy points and butchers the smallest in relative terms.
The distortion is visible in the coefficients (to_sram 1.97e-12, about 350x
to_hbm, physically inverted, and unstable under R3, which zeroes to_mac
entirely) and in the three model-subtracted ctx4096 per-token rows, which
break monotonicity only because the inflated SRAM coefficient overpredicts
the subtracted prefill component.

**Exploratory (post-hoc, labeled; R1 relative-error NNLS, winner form):**
- Full data: R2 0.924, MAPE 18.2%. Coefficients are physically sensible and
  stable between the train-split and all-296 fits (to_mac 3.4e-15, to_sram
  2.5-2.7e-14, to_hbm 7.5-7.7e-15 J/TO, launch 0.76-0.82 mJ). The fitted
  sram/hbm per-TO coefficients land within ~3.5x of each other, meaning the
  45 nm priors already carry most of the memory-hierarchy ratio.
- E2 under R1: pooled MAPE 15.24% on the IDENTICAL split that fails at
  50.37% under the absolute estimator (E1: 13.25%). Same features, same
  splits, estimator swapped: the E2 failure is attributable to estimator
  scale-weighting, not to the TO feature set. Stated with the exploratory
  label; the pre-registered E2 verdict remains FAIL.
- MCER under R1 recovers the EXP-001 regime split qualitatively: forward
  phases 0.22-0.43 (below 1, compute-dominated), decode 9.8-10.5 (far above
  1, memory-dominated). Calibrated decode MCER sits well below EXP-001's
  prior-weighted TO ratio (~66-70); part of the gap is that the fitted
  launch term absorbs decode energy outside both the MCER numerator and
  denominator, part is the priors' HBM weighting. Quantifying that split is
  paper analysis, not asserted here.
- Per-token monotonicity is restored on the three model-subtracted rows
  (GPT-2-medium 0.92 J, GPT-2-large 1.60 J, GPT-2-XL 2.60 J per token at
  ctx4096), with all measured-subtracted rows identical by construction.

**Implications recorded now, before further analysis:** (a) the paper
reports both estimators with the confirmatory/exploratory boundary
explicit; (b) the A100 pre-registration amendment (Step 5) will
pre-register the relative-error estimator as PRIMARY for the cross-platform
test, informed by this exploratory result and written before any A100 data
exists; (c) the baseline bake-off (plan section 8) runs next on the
identical splits, with the absolute-estimator comparison as the
pre-registered one and an R1 companion reported under the same exploratory
label.


### 2026-07-24 - Section-8 baseline bake-off (pre-registered), completing Steps 2-3 on the 4090

**Status:** CONFIRMATORY bake-off per fit_plan sections 8 and 12, recorded
as it fell; artifacts at commit db1f984 supersede aac684f. Registry-
correction footprint verified exactly as predicted: every to_hbm
coefficient rescaled by precisely 232/240 (e.g. winner 5.555e-15 ->
5.37e-15), split-column results otherwise identical, and the combined-
column models drifted only in trailing digits (M0 MAPE_test 49.44 ->
47.93; the M0/M1 tail order in the AIC table swapped at a 0.7-point AIC
gap). No verdict moved.

**Pre-registered primary comparison (58-point main test, absolute
estimator, one-sided Wilcoxon, Holm, alpha 0.05):** the TOML winner
(MAPE 88.26%) significantly BEATS layerwise (96.50%, Holm p = 5.1e-4) and
does NOT beat M0_flops (47.93%) or roofline (29.81%) on MAPE. Secondary E2
set (n=14): winner 50.37%, M0 56.52%, roofline 30.42%, layerwise 49.53%,
no comparison significant. These verdicts stand. They are the
heteroscedasticity finding expressing itself through the pre-registered
protocol: the winner dominates every model on likelihood and variance
explained (R2_test 0.987 vs M0's 0.972) while the absolute loss butchers
relative error on small points, and single-scale baselines are naturally
MAPE-robust.

**Two diagnostics worth the paper's ink:**
- The roofline's fitted P_avg is 378.4 W (379.0 W at the 2325 MHz
  sensitivity ceiling), 2.2-2.5x the part's 150-175 W envelope. A fitted
  average power that exceeds the physical power limit is direct evidence
  the roofline's time model underestimates real time (single-batch
  workloads run far below peak, and launch/latency floors are absent), so
  its decent MAPE comes from a compensating scale, not physics.
- The layerwise regressor (priors stripped, D3/D5) zeroes raw_macs
  entirely and still lands last (96.5%): removing the precision-aware MAC
  weighting and the cost hierarchy is what the winner's single
  pre-registered significant victory is measuring.

**Exploratory R1 companion (labeled; main test set, MAPE only):** with the
scale-aware loss the identical feature set leads everything: winner_R1
20.10% < layerwise_R1 24.61% < roofline_R1 30.21% < M0_R1 32.41%. Together
with E2 under R1 at 15.24% (vs the absolute winner's 50.37% and roofline's
30.42% on the same predict set), the full-story ordering is consistent:
the TO feature set carries the signal; the estimator must be scale-aware
to express it in relative error. The paper reports the pre-registered
table and this companion side by side with the boundary explicit.

**Implication, recorded before Step 5:** the A100/Lambda pre-registration
amendment will name relative-error NNLS as the PRIMARY estimator and the
absolute fit as the secondary, with this section cited as the motivating
evidence, written before any A100 data exists. Steps 2-3 are now complete
on the 4090; remaining for EXP-002: Step 4 representativeness spot checks
(requires explicit go-ahead for HF downloads and brief GPU time), Step 5
amendment, Steps 6-7 A100 runs and transfer analysis, Step 8 figures and
manuscript.


### 2026-08-10 - Step 4 representativeness check: FAIL at 0.3303 vs the 0.33 band, with strong regime structure

**Status:** pre-registered verdict, recorded as it fell (configs/exp_002.yaml
`representativeness`; operationalization LOGBOOK 2026-07-24; data
representativeness.jsonl, 12/12 cells ok; reports beside the data). The
sweep is FLAGGED per the pre-registration. The margin is the narrowest
possible: one cell (GPT-2 prefill fp16, init seed 1234) at 0.3303 against
0.33; the other eight ratios lie within the band.

**The structure is more informative than the binary verdict:**
- GPT-2 decode (memory-bound): ratios 0.003-0.011, init-seed CV 0.72%.
  Weight values are irrelevant where traffic dominates; also arithmetically
  consistent, since the ~0.5 J weight-sensitive prefill inside the 13.5 J
  composite caps its influence at ~1%.
- BERT-base encode: ratios 0.133-0.160, init CV 1.36%. Mild.
- GPT-2 prefill (compute-bound fp16): ratios 0.243, 0.330, 0.321; pretrained
  HIGHER than random in every cell and SLOWER per forward (~6.0 ms vs
  3.0-4.0 ms at identical shape).

**Implementation-free DDEV evidence:** the random-arm init-seed CV (same
code, same shapes, only weight values differ) is 5.60% on compute-bound
prefill vs 0.72% on decode. Value-dependent energy is real in our own stack
and concentrates exactly where physics predicts: in the arithmetic, not the
traffic.

**Attribution caveat, recorded with equal weight:** the pretrained arm runs
the HF implementation (Conv1D with bias, HF norms, framework overhead; SDPA
was pinned), and its ~70% per-forward wall-time overhead at identical shape
means the pretrained-vs-random ratio entangles implementation cost with
genuine data-dependent variation. BERT's 0.13-0.16 with the same HF overhead
present suggests implementation alone plausibly contributes O(0.1); the
GPT-2 prefill excess above that is consistent with real DDEV. A candidate
mechanism for the direction and the fp16 specificity: random-init
activations through a deep stack saturate/overflow in fp16, collapsing bit
entropy and hence switching activity, so random-init draws LESS energy;
LayerNorm-bounded BERT is milder; decode is immune. Mechanism is a
hypothesis, not a claim.

**Pre-committed trigger fired:** the 2026-07-24 note reserved an fp32
follow-up "only if the band is approached"; it was exceeded. fp32 does not
saturate, so if fp32 prefill ratios collapse toward the BERT level, the
fp16-numerics mechanism is confirmed and the flag can be scoped to
compute-bound fp16 cells. The decisive implementation-free experiment is
porting pretrained GPT-2 weights INTO our own stack (transposing HF Conv1D
weights), isolating values from implementation entirely.

**Consequences:** (1) the frozen 4090 dataset carries the pre-registered
flag, scoped by this structure: decode calibration is representativeness-
robust, compute-bound fp16 cells may under-represent pretrained energy by
up to ~25-33%; (2) the Step 5 A100 amendment must state the weights policy
in light of this verdict (random-init with the scoped flag, pretrained
arms on a subset, or the weights-porting control) BEFORE any A100 data;
(3) the paper reports the FAIL as registered, with this structure and the
attribution caveat.


### 2026-08-10 - Follow-up A (fp32 mechanism probe): fp16-saturation SUPPORTED; dataset flag scopes to compute-bound fp16

**Status:** post-verdict mechanism probe under the fired 2026-07-24 trigger,
run on the RTX 4090 Laptop GPU; NOT part of the pre-registered verdict,
which remains FAIL 0.3303 and re-printed identically from the frozen
records. Both predictions stated in the LOGBOOK BEFORE measurement came
true.

**Result:** GPT-2 prefill s512 fp32 ratios 0.1251 / 0.1254 / 0.1212 (from
0.243-0.330 at fp16), below even the BERT fp16 level; random-arm init-seed
CV collapsed 25x, 5.60% (fp16) -> 0.22% (fp32), i.e. to measurement-noise
level, matching decode's 0.72%.

**Decomposition this supports:** the pretrained-vs-random ratio is
(implementation floor) + (fp16-specific value dependence). The floor is
~0.12-0.16, value-independent (fp32 ratios are seed-flat at 0.22% CV; BERT
fp16 sits at 0.13-0.16 with the same HF overhead), attributable to the HF
stack (Conv1D biases, HF norms, extra launches). The fp16-specific
value-dependent component on compute-bound GPT-2 prefill is the excess
above the floor, ~0.12-0.21, and is the part that varies with init seed.
Mechanism (supported, not proven; n=1 model, 3 seeds): random-init fp16
activations saturate through the deep stack, collapsing bit entropy and
switching energy; fp32 does not saturate; LayerNorm-bounded BERT is mild;
traffic-dominated decode is immune.

**Internal consistency:** the fp32/fp16 random-arm energy ratio here is
2.8-3.1, inside the frozen sweep's forward-phase precision band
(2.04-4.11, median 3.24), from fully independent measurements.

**Flag scoping (supersedes the blanket reading of the FAIL):** the frozen
4090 dataset's representativeness flag applies to COMPUTE-BOUND fp16 cells;
decode is measured-immune, fp32 is measured-immune, encoder fp16 is mild
(0.13-0.16, at the implementation floor). The pre-registered FAIL verdict
itself is unchanged.

**Predictions for Follow-up B, stated NOW before any B code or data:**
porting pretrained GPT-2 weights into OUR stack (implementation held fixed)
should yield, at prefill s512: fp16 ported-vs-random ratio ~0.1-0.2 (the
pure value effect) and fp32 ported-vs-random ratio ~0.00-0.03; and
HF-vs-our-stack at IDENTICAL pretrained weights should reproduce the
~0.12-0.16 implementation floor. B is the decisive implementation-free
isolation and directly informs the Step 5 A100 weights policy.


### 2026-08-10 - Follow-up B (implementation-free isolation): the FAIL was implementation, not values; flag narrows to ~6.5%

**Status:** Follow-up B complete on the RTX 4090 Laptop GPU; ported cells
logit-verified against real gpt2 (fp32 gate) before measurement; reports
beside the data. The pre-registered FAIL verdict (0.3303 vs 0.33) stands
unchanged and re-printed from frozen records. Prediction scorecard against
the pre-stated bands (findings.md 2026-08-10, written before any B code):
TWO CONFIRMED, TWO REVISED BY THE DATA, recorded plainly:
- structure delta 0.0051 [predicted ~0]: CONFIRMED (bias+wpe cost nothing).
- fp32 value effect 0.0020-0.0067 [predicted 0.00-0.03]: CONFIRMED.
- pure fp16 value effect 0.0648 [predicted 0.10-0.20]: SMALLER than
  predicted, and ported (0.4131 J) sits INSIDE the random-init spread
  (0.3886-0.4418 J incl. random_v).
- implementation floor fp16 0.4046 [predicted 0.12-0.16]: FAR LARGER than
  predicted; fp32 floor 0.1357 (in the predicted band).

**Revised decomposition (all measured, GPT-2 prefill s512):**
- The fp16 primary gap (0.24-0.33) is reconstructed almost entirely by the
  HF-fp16 implementation overhead: E_HF/E_ported = 1.405 at IDENTICAL
  weights, and 1 - E_random/E_HF over the random spread reproduces
  0.24-0.33 exactly.
- The implementation-free fp16 value effect is ~5-6.5% (ported vs random_v
  0.065; ported vs plain random 0.048-0.063; init-seed CV 5.60%), i.e. the
  trained-value energy sits within the random-init family's own spread
  scale.
- fp32: implementation overhead 13.6%, value effect <= 0.7%, init CV 0.22%.
- Coherence check: the HF fp32 floor (0.136) matches the BERT fp16 floor
  (0.13-0.16), while the HF-GPT2 fp16 floor is an outlier at 0.40,
  indicating GPT2-specific fp16 handling in the HF stack (e.g. internal
  upcasts) rather than generic framework overhead. The fp16-saturation
  hypothesis survives only as a candidate driver of the ~5.6% init spread,
  NOT of the headline gap; Follow-up A's collapse is now read as HF's
  fp16-specific overhead vanishing at fp32, plus the genuine init-CV
  collapse (5.60% -> 0.22%), which remains implementation-free evidence of
  fp16 value-dependence at the few-percent scale.

**Scientific conclusion for the paper (verdict untouched):** the
pre-registered check compared ACROSS implementations, so its FAIL measures
the HF-vs-our-stack delta more than weight values. Measured
implementation-free, random-init energies represent trained-weight energies
within ~6.5% in the worst regime studied, comfortably inside the 33% band.
The frozen dataset's flag NARROWS accordingly: Fork-1 stands at the ~6%
level for the measurement stack the sweep actually used; the reported FAIL
is retained, explained, and decomposed. Honest limits: one model (GPT-2),
one shape (prefill s512), three seeds; the ported path exists for GPT-2
only.

**A100 consequence (recorded before the Step 5 amendment):** the amendment
will specify random-init for the full A100 grid, citing this decomposition;
optional ported/HF spot cells on the A100 replicate the floor and value
effect cross-platform if budget allows.


### 2026-08-17 - A100 fp32/fp16 matched-shape check: 34 pairs, 0 inversions; forward precision ratio about 2.3x the 4090's

**Status:** validator output (scripts/validate_exp002.py section 10, with the
seed-agnostic pairing added this date; a100/validation_report.{txt,json}),
descriptive; not a pre-registered test; recorded before any Step 7 fit.

**Result (A100-SXM4-40GB, per-unit instrument-B energy):** 34 fp32/fp16
matched-shape pairs (prefill 12, decode 12, encode 8, decoder_prefill 2),
0 inversions (fp32 > fp16 in every pair). Forward-phase fp32/fp16 ratio
(n=20): min 6.20, p25 6.68, median 7.43, p75 8.76, max 10.20.

**4090 comparison (frozen 296-pt sweep, committed validation_report.txt):**
143 pairs, 0 inversions; forward-phase ratio (n=62): min 2.04, p25 2.55,
median 3.24, p75 3.79, max 4.11. The precision penalty is a platform
property: ratio of medians 2.29, and the two ranges are disjoint (A100 min
6.20 above 4090 max 4.11).

**Coverage note:** 36 fp32 points, 34 paired; the two unpaired are the
enc-dec decode fp32 anchors at s1024/ctx1024 (T5-small, BART-base), whose
fp16 twin cell is not in the frozen grid; frozen-grid property, not a data
gap. Decode-like ratios are not emitted by the validator; they belong to
Step 7's descriptive tables.

**Hypothesis (not a claim):** Ampere runs true fp32 GEMMs on CUDA cores at a
small fraction of its FP16 tensor-core peak, whereas the Ada consumer part's
fp32-to-fp16 throughput gap is much smaller; energy per unit follows. Whether
the runner leaves PyTorch's default TF32 setting (matmul TF32 off) must be
checked before this appears in paper text; vendor peak figures to be cited
if used.

**Consequence for Step 7 (prediction registered in the LOGBOOK this date):**
T2 calibrates one scalar on fp16-only cells; the 36 fp32 evaluation cells
are expected to be systematically under-predicted and to dominate T2's
residual. Thresholds and mechanisms unchanged.


### 2026-08-17 - A100 pre-registered tests: T0 PASS; T1, T2, T3 FAIL (verdicts as they fell)

**Status:** CONFIRMATORY, pre-registered (a100_amendment.md sections 3, 9,
16, approved 2026-08-10, amendment 2 recorded 2026-08-17 before any fit).
Computed once by scripts/fit_exp002_a100.py at commit 176183d; artifacts in
experiments/exp_002_size_sweep/a100/fit/. Estimator R1 relative-error NNLS
(primary), form M8_split_dispatch frozen, device registry a100 -> hbm2e.
Verdicts stand as recorded; nothing was re-selected.

**T0 (pooled A-B median <= 5%): PASS, 0.60%** over all 98 points (forward
0.58%, decode-like 0.63%; shared-84 median 0.59%); B-C (Zeus) median
0.000%. The 100 Hz + hardware-counter methodology holds on Ampere with 8x
margin, versus 4.80% on the 4090.

**T1 (on-platform R1 refit, held-out MAPE <= 25%): FAIL, 35.60%** on the
17-point stratified test split (67 train; strata (arch, phase-class,
precision), seed 42, same machinery as the 4090). Absolute companion 71.68%.
R1 coefficients on train: to_mac 3.264e-15, to_hbm 3.664e-15, to_sram 0,
n_launches 0, intercept 0 J/TO (two of five terms fit to zero on the A100;
the 4090 R1 vector had to_sram 2.69e-14 and n_launches 7.57e-4). Held-out
breakdown (section 16.4): fp16 MAPE 28.4% (mean signed +19.9%), fp32 45.8%
(mean signed -40.0%); fp16/forward +38%, fp32/forward -50%, fp16/decode-like
14%, fp32/decode-like -26%.

**T2 (scale-calibrated transfer, MAPE <= 30% on the remaining 76): FAIL,
52.93%** with s* = 0.386 fitted on the 8 fp16 cells of section 16.2 against
the frozen 4090 R1 vector (db1f984 lineage, values pinned). Zero-shot (s=1)
105.7%. Absolute companion 106.7%. Breakdown on the 76: fp16 41.1% (mean
signed -7.5%), fp32 66.0% (-60.3%); fp16/decode-like +21.5%, fp16/forward
-33.8%, fp32/decode-like -41.0%, fp32/forward -75.7%. Sensitivity companion
(section 16.3): with the enc-dec decode calibration cell replaced by each
alternate, 52.1 / 51.9 / 54.2%, all FAIL; the cell choice is immaterial.
Zero-shot y/yhat is 0.68-0.70 on four of the five fp16 forward calibration
cells (0.58 on GPT-2-XL prefill) but 0.24-0.29 on the small-model decode
cells and 0.69 on GPT-2-XL decode: even at fp16, one scalar cannot serve
both regimes because the dispatch term does not scale with the compute and
memory terms across hosts.

**T3 (R1 fit on all 84, predict the 10 7B-class points, pooled MAPE <=
30%): FAIL, 42.66%.** Absolute companion 192.5%. Per regime: decode (4)
5.8% MAPE, mean signed -5.8%; prefill s1024/4096 (4) +67.8%; prefill s8192
(2) +66.2%. Every 7B prefill is over-predicted by +65 to +71%; every 7B
decode is within -1 to -13%. R1 coefficients on all 84: to_mac 3.118e-15,
to_hbm 3.62e-15, others 0; in-sample MAPE 35.8%, R2 0.766.

**Pre-registered predictions (LOGBOOK 2026-08-17), scored:** T1 fp16 +20% /
fp32 -50% and MAPE 25-35% predicted; observed +19.9% / -40.0%, 35.6%:
CONFIRMED (top of the range). T2 fp32 cells expected to carry the bulk of
the residual: CONFIRMED (fp32 66% vs fp16 41%), but the fp16 subset also
exceeds the band, which was NOT predicted (see mechanism 2). T3 forward
over-prediction predicted at about +20% with moderate risk: direction
CONFIRMED, magnitude EXCEEDED (+67%). Roofline competitive on the A100:
CONFIRMED (below). Model-implied fp32/fp16 ratio capped at 3.03 vs measured
above 6: CONFIRMED (forward pairs, 20: measured median 7.52 [6.23, 10.18],
model-implied 2.96 [2.51, 3.01]; decode-like pairs, 14: 4.12 vs 2.18).

**Post-hoc descriptive analysis of the residual structure (NOT
pre-registered; labeled as such):** from per_point_predictions.jsonl joined
with the frozen records' per-point median SM clocks and repeat temperatures.
Mechanism 1, precision datapath (predicted): fp32 forward cells -52.8% mean
signed, uniform (-47 to -57) across every model and shape. Mechanism 2,
operating point (not predicted): the fp16 flash forward residual of the
all-84 fit tracks the recorded SM clock. Cells at 1410 MHz (9): +6.0% mean,
range 0 to +16, the model works there. Cells at the low DVFS state, 1095
MHz (5, all s128, 20-60 W active): +44 to +73%. Cells that ran power-capped
at 1250-1330 MHz with about 320-330 W above the 71 W idle, i.e. at the 400
W envelope (GPT-2-XL fp16 prefill at s128/1024/2048; every 7B prefill):
+40 to +71%. fp32 cells ran at 1410 MHz at 260-300 W (the CUDA-core path
does not saturate power). Energy per TO on the A100 therefore depends on the
operating point by roughly a factor of two across the grid, and the frozen
form has no term for it; this is the whole of T3's prefill error, while
memory-bound 7B decode at 1410 MHz transfers within 6%. Mechanism 3, minor:
the six eager cells are under-predicted by 71% (math SDPA backend on
Ampere); to_sram and n_launches fit to zero, so the eager penalty (extra
launches, materialized scores) has little representation. Pipeline checks:
units gate passed on all 98; the 7B decode composites reconcile
arithmetically (prefill over-prediction offset by a slightly
under-predicted decode part); T0 at 0.60%; no artifact found.

**Secondaries (pre-registered as such).** D6 coefficient transfer, R1
all-84 over R1 all-296: to_mac 0.906, to_hbm 0.489, to_sram and n_launches
0 (4090 nonzero); effective per-word off-chip energy (alpha_hbm x prior
word cost) 695 pJ/word on the A100 vs 1,778 pJ/word on the 4090, ratio 0.39
against the recorded expectation of about 1.0 (0.96 under an HBM2 6.25
pJ/bit prior; the 40 GB SXM4 part is HBM2 per the vendor datasheet, the
registry tier is labeled hbm2e). Caveat: the A100 alpha_hbm also absorbs the
vanished sram and launch terms and the precision compromise. MCER on the
A100 (R1 all-84) vs the 4090 (R1 all-296): decoder decode 4.6 vs 10.5;
enc-dec decode 5.2 vs 9.8; decoder prefill 0.11 vs 0.43; encoder encode
0.046 vs 0.32; enc-dec encode 0.044 vs 0.27; decoder_prefill 0.087 vs 0.23.
The prefill/decode MCER phase transition (the central thesis) reproduces on
Ampere; the magnitudes are platform-dependent. D7 baselines on the T1 split
(secondary Wilcoxon + Holm, n=17): under R1, M8 35.6%, roofline 32.2%, M0
36.6%, layerwise 39.8%, none beaten (Holm p 0.43-0.66); under absolute NNLS,
M8 71.7%, roofline 33.0%, M0 66.0%, layerwise 174% (only layerwise beaten,
Holm p 3e-4). Roofline fitted P_avg 368.7 W (R1) / 413.0 W (absolute)
against the 400 W envelope: on the A100 the roofline is physically sane and
is the best predictor by MAPE, though not significantly at n=17 (the 4090
diagnostic was 378 W against a 150 W part). Roofline constants: FP32 19.49 /
FP16 311.9 TFLOP/s dense, 1,555 GB/s, vendor-cited in fit/baselines.py.
Spot cells (descriptive, excluded from every fit): GPT-2 prefill s512 fp16,
random 0.1933, random_v 0.1977, ported 0.1908, HF pretrained 0.2140 J per
forward; 1 - E_random/E_HF = 0.097 (4090: 0.24-0.33); E_HF/E_ported = 1.122
(4090: 1.405); E_ported/E_random_v - 1 = -3.5%, E_ported/E_random - 1 =
-1.3% (4090: about +5 to +6.5%). The Follow-up B decomposition replicates on
Ampere with smaller magnitudes: the HF-vs-random gap is implementation
overhead, and random-init represents trained weights within about 3.5% on
this platform, with the sign reversed.

**Scientific reading for the paper (verdicts untouched):** the cross-platform
claims 1-3 of the amendment fail as registered, for two identifiable,
physically interpretable reasons that the framework does not model:
execution-unit routing by precision (tensor cores versus CUDA cores) and
operating-point dependence of energy per op (DVFS state and the power cap).
Both were named as limitations in the prior TOML papers (FLAIRS: DVFS; cloud
paper: TED); on the A100 they are first-order. What does transfer: the
measurement methodology (T0), the memory-bound decode regime (7B decode
within 6% under extrapolation of 4.7x in parameters), and the MCER phase
transition. What the paper must not say: that the precision multiplier was
fitted (it is asserted at 0.33) or that coefficients transfer up to a
scalar. Post-verdict exploratory work is pre-registered in amendment section
16.5: (a) a precision-split MAC model, and, motivated by mechanism 2, a
descriptive residual-vs-operating-point table; (b) optional Follow-up C
(TF32 probe). Neither alters any verdict above.


### 2026-08-18 - EXPLORATORY: precision-split MAC model (M8p) and operating-point table on the A100; correction to the 2026-08-17 T3 attribution

**Status:** EXPLORATORY, pre-registered as post-verdict work in
a100_amendment.md section 16.5(a); computed by scripts/explore_exp002_a100.py
at commit 4e774c7 after the T0-T3 verdicts were recorded; artifacts in
a100/fit/exploratory/. NO VERDICT AUTHORITY: T1/T2/T3 remain FAIL as
recorded on 2026-08-17. M8p = M8_split_dispatch with to_mac split into fp16
and fp32 columns, fitted under R1 (same math as fit_relative). "Would be
inside the band" below is a description of a number, not a verdict.

**(a1) M8p on the identical T1 split (17 held-out):** MAPE 20.02% (M8 on
record 35.60%; the band would be 25%). Coefficients (train): to_mac_fp16
1.925e-15, to_mac_fp32 7.116e-15, to_nonlinear 2.856e-14, to_sram 0, to_hbm
3.614e-15, n_launches 1.257e-05, intercept 0. Fitted fp16 MAC multiplier
(alpha_fp16 x 0.33 / alpha_fp32) 0.089 vs the asserted 0.33. Held-out
breakdown balanced: fp16/decode-like -9.4%, fp16/forward +8.9%,
fp32/decode-like -11.0%, fp32/forward +5.6% (M8: -7.6 / +38.2 / -26.3 /
-50.3).

**(a2) M8p on all 84, predicting the 10 7B-class points:** pooled MAPE
11.04% (M8 on record 42.66%; the band would be 30%): prefill s1024/4096
+4.5%, prefill s8192 +10.6%, decode -17.8%. Fitted multiplier 0.083;
in-sample MAPE 19.74%, R2 0.941 (M8: 35.8%, 0.766). In-sample:
fp16/decode-like -3.6%, fp16/forward -15.1%, fp32/decode-like -15.9%,
fp32/forward +2.5%.

**(a3) T2 repeat with an M8p refit on all 296 4090 points, one scalar on
the same 8 cells:** MAPE 52.42% on the 76 (M8 on record 52.93%; s* 0.367 vs
0.386); zero-shot 102.7%. Breakdown: fp16/decode-like +19.7%, fp16/forward
-42.9%, fp32/decode-like -42.1%, fp32/forward -73.5%. The 4090 M8p fit:
to_mac_fp16 1.352e-15, to_mac_fp32 3.437e-15, to_nonlinear 3.729e-14,
to_sram 1.525e-13, to_hbm 7.201e-15, n_launches 8.598e-04, intercept 0;
in-sample MAPE 13.97% (frozen M8 R1 on record 18.24%); fitted multiplier
0.130 vs the asserted 0.33. Cross-platform coefficient ratios A100/4090:
to_mac_fp16 1.31, to_mac_fp32 2.05, to_hbm 0.50, n_launches 0.0096 (about
100x smaller on the A100 host), to_sram 0 vs nonzero.

**(b) Operating-point table (M8 -> M8p mean signed residual of the all-84
fits, by recorded median SM clock; P_total = net + measured idle):**
extension fp16 prefill at 1290 MHz, 397 W total (the 400 W envelope): +67
-> +6.5%; shared fp16 forward at the 1095 MHz DVFS state, 88 W total: +56
-> +24.5%; shared fp16 forward power-capped at 1316 MHz, 392 W: +35 ->
+7.6%; shared fp16 forward flash at 1410 MHz, 211 W: +6 -> -11%; fp32
forward at 1410 MHz, 331 W: -53 -> +2.5%; fp32 decode-like at 1410 MHz: -50
-> -28%; fp16 decode-like at 1095 MHz: +1.5 -> +0.6%; eager fp16 forward at
1410 MHz: -71 -> -77%; 7B decode at 1410 MHz, 293 W: -6 -> -18%.

**Correction (honest, to the 2026-08-17 entry):** that entry attributed
T3's prefill error entirely to the operating point. That was wrong. The
+66 to +71% on every 7B prefill was overwhelmingly the alpha_mac
compromise: M8's shared coefficient 3.118e-15 against M8p's fp16 coefficient
1.766e-15 is a ratio of 1.77, which is the +67% (LLaMA-7B prefill s1024:
predicted 37.2 J vs measured 21.8 J under M8; +2 to +11% under M8p). The
operating-point effect is real but secondary: after the split, the 1095 MHz
cells sit at +24.5%, the power-capped cells at +7%, and the 1410 MHz cells
at -11%, a spread of about 35 points rather than a factor of two. Also
revised: fp32 decode-like cells remain under-predicted by 28% under M8p
(range -63 to +5%), a residual precision effect in the decode regime that
the split does not remove; and the eager cells remain under-predicted by
about 4x, an inadequate representation of the math-SDPA path on Ampere (six
cells; not pursued here).

**Reading (exploratory, for the journal and EXP-003; nothing here changes a
verdict):** (i) The asserted fp16 MAC multiplier 0.33 (provenance
NEW_ESTIMATE, Horowitz bit-width scaling) is not supported on either
platform: fitted 0.130 on the 4090 and 0.083 on the A100. The precision
prior is a device property set by execution-unit routing (tensor cores vs
CUDA cores), and it belongs in the device registry beside the memory tier.
(ii) With that single change, on-platform refit and 7B extrapolation on the
A100 land at 20.0% and 11.0%, numbers that would sit inside the
pre-registered bands; this is the exploratory finding to carry into a
re-pre-registered protocol (EXP-003 or the next platform), never re-fitted
here as confirmatory. (iii) Cross-platform scalar transfer still fails
(52.4%) because three device-level quantities differ, not one: the fp16
multiplier, the dispatch coefficient (about 100x; hypothesis: Windows WDDM
kernel-launch overhead on the 4090 laptop host versus Linux on the A100
host, consistent with the signals paper's host-dependent alpha_o), and the
operating-point behavior; the memory-tier coefficient ratio is 0.50, i.e.
HBM2 at about half of GDDR6 per fitted word beyond the 0.80 prior ratio.
(iv) Follow-up C (TF32 probe, section 16.5(b)) remains the direct mechanism
test for (i) and stays optional.


### 2026-08-18 - Instrument A characterization: agreement is window-length dependent; sampler ceiling about 96 Hz; GIL contention ruled out

**Status:** DIAGNOSTIC, post-campaign, on the RTX 4090 Laptop GPU. Triggered by
`tests/test_instruments.py::test_gpu_smoke_A_and_B_agree_roughly` failing on an
idle GPU (A-B 50-58% across three runs) where it had previously passed. Run
with `scripts/diag_instrument_a.py`; artifact
`experiments/exp_002_size_sweep/diagnostics/instrument_a.json`. Changes NO
verdict and NO frozen datum: both energy.jsonl files and every recorded A-B
figure stand exactly as committed.

**Environment identical to the sweep:** driver 595.79 in both
`environment.json` (2026-07-20) and today's `nvidia-smi`. Not a driver change.

**Result 1, settled 3.0 s matmul window (the healthy regime).** A-B by
sampling rate: 24.4% at 20 Hz, 7.1% at 50 Hz, 8.3% at 100 Hz, 7.1% at 200 Hz.
Trapezoid, rectangle and mean-times-duration agree to 0.1% at every rate;
coverage 0.989-1.000; first-sample delay negative (the sampler opens before
t0). The residual is therefore not integration placement, not span coverage,
and not undersampling above 50 Hz.

**Result 2, GIL contention ruled out.** At 100 Hz the in-process sampler's mean
power is 491.8 J / 3.018 s = 162.9 W; the EXTERNAL out-of-process nvidia-smi
logger reads 163.7 W over the same window. Two independent readers of
`nvmlDeviceGetPowerUsage` agree, while the accumulator implies 536.3 / 3.018 =
177.7 W on a 175 W part. The residual is a genuine NVML sampled-power versus
hardware-counter offset, the branch the diagnostic's own guide maps to "B
primary, A a sanity check". That decision was already on record and is
unchanged; this is independent support for it, not a new decision. Caveat: the
external logger's ENERGY column is unusable (218-413 J for the same window; its
loop timing is unreliable), so only its mean power is meaningful.

**Result 3, sampler ceiling about 96 Hz on this host.** A 5 ms requested
interval yields a 10.37 ms median gap, indistinguishable from the 10 ms request
(10.43 ms). 200 Hz buys nothing here. The sweep's 100 Hz was already at the
practical ceiling, which retrospectively justifies that choice.

**Result 4, BOTH instruments degrade below about 0.5 s.** On the 0.25 s GPT-2
prefill s1024 window, A-B runs 36-94%, improving monotonically with rate, and B
is itself unstable: 33.5, 35.5, 52.1, 37.5 J for identical work, a spread of
about 25%, with the 52.1 J reading implying 209.6 W on a 175 W part. That is
the accumulator's update period showing through. This is the first quantified
statement in this project of WHY the frozen protocol's 4.0 s minimum window
(measure_until_floor) is necessary; previously the floor rested only on the
20 Hz versus 100 Hz sampling result of 2026-05-29.

**Why the smoke test changed behavior.** It measured ONE unfloored window of
about 0.7-0.9 s, sitting between the 3 s case (8%) and the 0.25 s case
(36-94%), and asserted a 50% bound. It was always marginal: 50.19% inside the
full suite, where the GPU was warm from preceding tests, and 57-58% cold.
Nothing regressed; a thin margin tipped. The test was rewritten this date to
size its window to the 4 s floor, to assert the floor was actually reached, and
to gate at 25% against the 7-8% measured there. The band was NOT simply
loosened: the window was corrected so that the assertion tests the regime the
sweep actually operates in.

**Consequence for the paper (positive, methods).** The 4 s floor and the
instrument hierarchy are now supported by direct measurement rather than by
assertion. The sweep's recorded pooled A-B of 4.80% on the 4090 and 0.60% on
the A100 were obtained at or above that floor and sit at or better than the
7-8% seen here at 3 s, which is consistent. Honest limits: one host, one GPU,
two workloads, and the short-window arm confounds window length with workload
(matmul 3.0 s versus GPT-2 prefill 0.25 s), so "window length" is the supported
reading, not a proven isolation of duration from burstiness.

**A separate benign artifact, recorded so it is not re-investigated:** idle
`nvidia-smi` reports 593 W on this 175 W part at 0% utilization with Disp.A
off. This is the documented Runtime D3 suspended-sensor reading on Ada mobile
parts, not a telemetry fault; it does not touch the measured windows, where the
GPU is awake, and it is not evidence of anything about instrument A.


### 2026-08-19 - CORRECTION to the 2026-08-18 instrument entry: window floor provenance, and prior art for the sub-second instability

Two statements in the 2026-08-18 entry above need correcting. Verdicts,
datasets, and the test fix are unaffected; what was wrong is the provenance
narrative.

**Correction 1: "the frozen protocol's 4.0 s minimum window" conflates the
registration with the implementation.** The pre-registered configuration
(`configs/exp_002.yaml`, `min_window_s`, comment dated 2026-05-29) records a
floor of **2.0 s**. The implementation default in
`src/tomltransformers/sweep/point.py` (`target_s = 4.0`, `min_window_s =
4.0`) is what actually ran, and the frozen data confirms it: all 296 RTX 4090
records have `wall_time_s_median >= 4.001 s`, zero `short_window` flags. So
the operative floor was 4.0 s, the registered floor was 2.0 s, and the
divergence (conservative direction: the harness was stricter than the
registration) was never amended in the config. The 2026-08-18 entry stated
the 4.0 s figure as "the frozen protocol's" without noting the config says
2.0; the paper's methods section (main.tex, Instrument Characterization)
discloses the divergence explicitly, and this entry brings the lab record in
line with the paper.

**Correction 2: the sub-second instability of instrument B was presented as a
new finding; it was first observed on 2026-05-29.** The `min_window_s` config
comment records the original observation verbatim: on sub-second windows the
hardware energy counter itself was self-inconsistent (GPT-2 s=1024 x40 gave B
= 94/38/47/35 J across repeats, n~9 samples). What the 2026-08-18 diagnostic
ADDS is (i) the quantified 3.0 s point (A-B 7-8% at 50-200 Hz, methods
agreeing to 0.1%), (ii) the 0.25 s quantification with per-rate breakdown,
(iii) the external-logger check excluding GIL contention, and (iv) the ~96 Hz
sampler ceiling. The phenomenon itself was already on record and motivated
the floor in May; the 2026-08-18 entry's claim to be "the first quantified
statement in this project of WHY the 4.0 s minimum window is necessary"
overstates novelty and should be read as: the first characterization that
separates the candidate causes and quantifies the window-length dependence at
multiple points.

**Why this correction exists:** the 4.0 s figure was asserted from memory
during test triage on 2026-08-18 and propagated into `tests/
test_instruments.py` (as WINDOW_FLOOR_S, where it is CORRECT, matching the
implementation), `findings.md`, and `LOGBOOK.md` before the config was
re-read. The frozen data settled which floor actually governed. Lesson
already in the working protocol, restated because it was violated: verify
against the artifact before writing the number down, including in lab notes,
not only in paper prose.


### 2026-08-19 - Eager/flash energy ratio: the pre-registered derived quantity, computed

**Status:** DESCRIPTIVE, pre-registered in the fit plan as a report-only
derived quantity ("eager-vs-flash energy ratio vs s"). Pure measurement from
the frozen datasets: matched pairs on all spec fields except `attn_kind`,
instrument B per composite execution, no fitting, no coefficients.
Implementation: `figures/data.py::eager_flash_pairs`, covered by two tests in
`tests/test_figures_data.py` asserting the counts, extreme ratios, and both
scalings below.

**RTX 4090 (8 pairs, GPT-2 + DistilGPT2, prefill fp16, s = 512-4096):**
E_eager / E_flash = 1.93-1.98 at s=512, 3.73-3.75 at s=1024, 6.99-7.19 at
s=2048, 11.59-11.96 at s=4096.

**A100 (5 pairs; the sixth eager cell, DistilGPT2 s=512, has no flash twin on
the shared grid):** 3.71 at s=512 (GPT-2), 4.00-4.08 at s=1024, 6.68-6.88 at
s=2048.

**Two structural facts in the excess energy (E_eager - E_flash):**
1. Quadratic in s: each doubling multiplies the excess by 4.95 -> 4.33 ->
   4.04 (DistilGPT2, 4090) and 4.91 -> 4.42 -> 4.09 (GPT-2, 4090),
   converging to 4 from above as O(s) terms fade; A100 shows 3.16 -> 4.11
   (GPT-2) and 4.05 (DistilGPT2, one step).
2. Linear in depth: GPT-2's excess is 1.98-2.06x DistilGPT2's at every s on
   both platforms, matching its 2x layer count.

The excess therefore scales as layers x s^2, which is the off-chip
score-matrix traffic the standard-attention accounting charges and the fused
accounting does not. The two kernels perform identical arithmetic, so this is
a direct FLOPs-falsifying measurement: identical FLOP count, up to 12x
measured energy. The ratio itself is close across platforms at matched s
(3.74 vs 4.04-4.08 at s=1024; 6.99-7.19 vs 6.68-6.88 at s=2048), consistent
with it being a workload-structure quantity.

**Honest limits:** two models, one architecture family (decoder-only), one
phase (prefill), fp16 only; the quadratic and depth-linear scalings are
observed regularities on this grid, not established laws. Entered the paper
as section VI "Attention Kernel" this date; at the director's direction the
roofline physical-unsoundness paragraph in section IX was removed to make
room, with the fitted-P_avg-vs-envelope evidence retained in Table IV.
