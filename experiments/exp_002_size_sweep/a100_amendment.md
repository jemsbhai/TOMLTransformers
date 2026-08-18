# EXP-002 Amendment: Step 5 pre-registration for the A100/Lambda cross-platform phase

- Date: 2026-08-10 (America/New_York)
- Status: APPROVED 2026-08-10 (all recommended options adopted; section 15)
- Amends: frozen_exp_002.yaml (frozen 2026-05-29) and fit_plan.md sections 1-12
- Supersedes: the frozen config's deferral of the A100 to "EXP-004"; the A100
  phase is Steps 6-7 of EXP-002 per the 8-step plan (LOGBOOK).

## 0. Precondition declaration

As of this date, NO A100 measurement for EXP-002 exists. No A100 instance has
been provisioned for this experiment. Every analytical decision below is frozen,
committed, and pushed BEFORE first contact with A100 data. Once data exists,
changes to this amendment require a further dated amendment recorded before the
affected results are examined.

Decision points are marked [DECISION Dn]; each now records its ADOPTED choice
inline. The approval is recorded in section 15. This file, together with the
LOGBOOK Step 5 entry, is committed and pushed before any Step 6 work.

## 1. Purpose and claims under test

The A100 phase tests the cross-platform claims of the paper:

1. The TO feature set plus the frozen model form (M8_split_dispatch)
   calibrates on a second platform (Ampere A100 SXM4, HBM2E) with held-out
   accuracy comparable to the RTX 4090 Laptop result (T1).
2. Coefficients transfer across platforms up to a single global scale factor,
   i.e. the coefficient STRUCTURE is a workload/feature property and the
   device registry priors carry the memory-tier difference (T2).
3. The calibrated model extrapolates from GPT-2/BERT/T5-class training data to
   7B-class models and longer sequences on datacenter hardware (T3).
4. The 100 Hz + hardware-counter measurement methodology holds on a second
   platform (T0).

## 2. Platform, environment, budget

- Platform: Lambda 1x A100 (40 GB SXM4), 30 vCPU, 200 GiB RAM, 0.5 TiB SSD,
  Ubuntu / Lambda Stack. Device registry entry: `a100` -> `hbm2e`
  (192,000 fJ/word prior, 6.00 pJ/bit; to_costs.py, already present).
- Instance is provisioned by M. Syed; live rate checked at lambda.ai/pricing
  before booking. Budget envelope: 15-20 GPU-hours (planning estimate in
  section 8; the envelope, not the estimate, is binding).
- Chunked sessions with the established ritual: run chunk -> validate ->
  harvest (commit JSONL + reports + docs) -> push -> terminate instance ->
  relaunch and resume. Local instance disk does NOT survive termination;
  everything of value lives in git before terminate.
- environment.json snapshotted in the A100 output directory at first run
  (driver, CUDA, torch, transformers, zeus versions; instance type; actual
  clocks). Software versions on Lambda will differ from the 4090 box; they are
  recorded, not constrained.

## 3. Primary estimator: R1 relative-error NNLS

**This is the central analytical amendment.** For all A100 confirmatory fits
and tests (T1, T2, T3):

- PRIMARY estimator: R1 relative-error NNLS, defined as
  min over beta >= 0 of sum_i ((x_i . beta - y_i) / y_i)^2,
  exactly as implemented in the R1 path of scripts/fit_exp002.py
  (artifacts lineage: commit db1f984). Equivalent to weighted NNLS with
  weights 1/y^2.
- SECONDARY (reported alongside, no verdict authority): the absolute NNLS
  used as primary in the 4090 confirmatory fit.
- Model FORM is frozen: M8_split_dispatch, coefficients refit on A100 data.
  NO model selection (no AIC sweep over M0-M9) is run on the A100; the form
  was selected once, confirmatorily, on the 4090. This forecloses
  selection-level HARKing on the new platform.
- Target variable identical to the 4090:
  y = per_execution_median_j["B"] / inner_iters (instrument B,
  idle-subtracted, J per composite execution). The per-record units gate
  from the 4090 fit script applies unchanged.

Evidence chain motivating R1-as-primary, all recorded BEFORE this amendment
and labeled exploratory at the time (findings.md, both 2026-07-24 entries):
E2 pooled MAPE 50.37% (absolute, pre-registered FAIL) vs 15.24% under R1 on
the IDENTICAL split and features; physically sensible, stable coefficients
under R1 (to_mac ~3.4e-15, to_sram ~2.5-2.7e-14, to_hbm ~7.5-7.7e-15 J/TO,
launch ~0.8 mJ) vs the absolute fit's inverted to_sram; regime-correct MCER
under R1 (forward 0.22-0.43, decode 9.8-10.5); bake-off R1 companion ordering
winner 20.10% < layerwise 24.61% < roofline 30.21% < M0 32.41%. The 4090
pre-registered verdicts (including the E2 FAIL) are untouched by this
amendment and are reported as registered.

## 4. Weights policy: random-init full grid

The entire A100 grid (sections 5 and 6) runs RANDOM-INIT weights.

Evidence chain (findings.md 2026-08-10, Follow-up B): the 4090
representativeness FAIL (0.3303 vs 0.33) decomposed as HF-implementation
overhead, not weight values; measured implementation-free, random-init
energies represent trained-weight energies within ~6.5% in the worst regime
studied (compute-bound fp16 prefill), fp32 within ~0.7%, decode immune. The
A100 dataset inherits the NARROWED Fork-1 caveat (~6.5%, compute-bound fp16),
not the blanket 33% band. Honest limits carry over: the decomposition rests on
one model (GPT-2), one shape (prefill s512), three seeds.

Random-init additionally removes any HF checkpoint download from the A100
measurement path (no gated-checkpoint access needed for the 7B cells).

Optional spot cells replicating the Follow-up B decomposition on the A100 are
specified in section 7 [DECISION D5, ADOPTED: include].

## 5. Shared transfer grid (84 points, mirrors 4090 strata)

All cells: batch 1, SDPA/flash attention (the main-sweep default) except the
eager subset, decode = growing-window K = 64, seq/context values chosen as an
exact subset of the frozen 4090 grid values.

5a. Decoders (48 points):
    {DistilGPT2, GPT-2, GPT-2-medium, GPT-2-XL}
    x {prefill at s, decode at ctx = s} x s in {128, 1024, 2048}
    x {fp16, fp32}.
    GPT-2-large is omitted from the mirror (budget; interior size point).

5b. Encoders (12 points):
    {BERT-base, BERT-large} x encode x s in {128, 1024, 2048} x {fp16, fp32}.
    ViT models are not mirrored (budget; text-side coverage suffices for
    transfer).

5c. Encoder-decoder (18 points):
    {T5-small, BART-base} x {encode, decoder_prefill, decode}, following the
    4090 convention of the two coordinated sub-sweeps anchored at 1024
    (source-scaling and target-scaling arms), restricted to the anchor
    stratum: 12 fp16 cells (both arms) + 6 fp32 cells (the anchor cells,
    one per model x workload). The exact tuples are enumerated MECHANICALLY
    by the same grid builder used for the 4090 sweep (sweep/grid.py) in the
    Step 6 grid file, which is committed before any measurement. No
    hand-built cells.

5d. Eager-attention subset (6 points) [part of DECISION D5, ADOPTED: include]:
    {DistilGPT2, GPT-2} x eager x prefill x s in {512, 1024, 2048} x fp16.
    Mirrors the 4090 attention_compare stratum on the shared platform.

[DECISION D4] Grid tier. RECOMMENDED: the minimal 84-point grid above.
ADOPTED 2026-08-10: minimal 84-point grid.
Alternative (comfortable, approx +30 points, +3 GPU-h): add s = 512 to 5a
(16 points) and 5b (4 points), reinstate GPT-2-large at s in {128, 1024,
2048} x {prefill, decode} x fp16 (6 points), and decode ctx 4096 fp16 for
the four 5a decoders (4 points). NOT adopted.

## 6. A100-only extension (10 points)

{LLaMA-7B, Mistral-7B} x fp16 x flash x random-init:
- prefill at s in {1024, 4096}: 4 points
- decode at ctx in {1024, 4096}, K = 64: 4 points
- prefill at s = 8192: 2 points

Constraints recorded: 7B fp32 EXCLUDED (weights alone 28 GB on a 40 GB part);
all 13B EXCLUDED from this plan. Both architectures exercise RMSNorm, SiLU
gated MLP, and (Mistral) GQA; these enter the frozen feature columns through
the existing TO cost registry, so extrapolation is over feature VALUES and
prior mixes, not new columns. That is part of what T3 tests. OOM feasibility
is checked in the Step 6 smoke run; any OOM cell is skipped and logged per
the frozen policy.

## 7. Optional spot cells (4 points) [DECISION D5, ADOPTED: include]

GPT-2 prefill s512 fp16 x weights arm in {hf_pretrained, ported, random,
random_v}, exactly the Follow-up B configuration, replicating the
implementation-vs-values decomposition on Ampere. Descriptive only; EXCLUDED
from every fit and every confirmatory test. RECOMMENDED: include, together
with 5d (combined cost approx 1 GPU-h). ADOPTED 2026-08-10: include (spot
cells + 5d eager subset).

## 8. Point budget and time estimate (planning, non-binding)

84 shared + 10 extension + 4 spot = 98 points. At the measured ~6 min/point
(4090), approx 10 h; 7B cells add load/settle overhead (~10 min/point
planning number); setup, smoke, validation, and chunk overheads approx 3-4 h.
Total approx 14-16 GPU-h, inside the 15-20 envelope. If the envelope is
threatened mid-phase, the eager subset and spot cells are dropped FIRST (they
feed no confirmatory test), then the comfortable-tier additions if chosen;
the shared core (5a-5c) and extension (6) are protected.

## 9. Confirmatory tests and acceptance criteria

All confirmatory tests use the R1 primary estimator (section 3), instrument B
target, and the frozen M8 form. Absolute-NNLS companions are reported as
secondary for each test.

**T0 (instrument agreement, FIXED, mirrors the met 4090 target):**
pooled A-B median relative difference <= 5% over all A100 points. Zeus (C)
agreement reported descriptively; C unavailability is recorded and never
blocks (frozen-config policy).

**T1 (on-platform refit quality):** stratified train/test split of the
84-point shared grid, materialized by the SAME split machinery and
stratification keys as the 4090 fit (fit/splits.py), split seed from
seed.json, approx 80/20 (approx 17 test points). Verdict: held-out MAPE of
the R1 refit within the band.
[DECISION D1] band options 20% / 25% / 30%. RECOMMENDED 25% (mirrors the
pre-registered extrapolation band scale; R1 full-data MAPE on the 4090 was
18.2%). ADOPTED 2026-08-10: 25%.

**T2 (cross-platform coefficient transfer, scale-calibrated):**
- Freeze the 4090 R1 coefficient VECTOR (winner form, all-296 fit, artifacts
  at db1f984 lineage).
- Compute A100 features with the A100 device registry (hbm2e column).
- Fit ONE global scalar s* on the frozen 8-cell calibration subset below, by
  the same relative loss: s* = argmin_s sum_c ((s * yhat_c - y_c)/y_c)^2.
- Verdict: MAPE of s* * yhat on the REMAINING 76 shared-grid points within
  the band.
- Frozen calibration subset (8 cells, fp16, spanning size x regime x family):
  GPT-2 prefill s1024; GPT-2 decode ctx1024; GPT-2-XL prefill s1024;
  GPT-2-XL decode ctx1024; BERT-base encode s1024; BERT-large encode s1024;
  T5-small decode anchor-1024; BART-base encode anchor-1024.
[DECISION D2] band options 25% / 30% / 35%. RECOMMENDED 30% (transfer is
strictly harder than on-platform refit; prior cross-platform E/TO spread was
within 1.5x for 5 of 6 architectures). ADOPTED 2026-08-10: 30%.
Zero-shot transfer (s = 1 forced) is reported DESCRIPTIVELY, no verdict.

**T3 (extrapolation to 7B-class):** fit R1 on ALL 84 shared-grid points;
predict the 10 extension points; verdict: pooled MAPE within the band.
Per-regime breakdown (prefill vs decode vs s8192) reported descriptively.
[DECISION D3] band options 25% / 30% / 35%, or demote T3 to descriptive-only.
RECOMMENDED confirmatory at 30% (params extrapolate ~4.7x beyond GPT-2-XL and
s extrapolates 4x beyond the shared grid; the 4090 E2-under-R1 analogue was
15.24% at a smaller leap). ADOPTED 2026-08-10: confirmatory at 30%.

Multiplicity: T0-T3 are four pre-registered tests with distinct claims; each
is reported against its own band with no cross-test correction (stated in the
paper). Any additional significance testing follows section 10.

## 10. Secondary and descriptive analyses (pre-registered as such)

- Coefficient-transfer table: per-coefficient ratio alpha_A100 / alpha_4090
  (R1 fits), plus the effective per-word off-chip energy ratio
  (alpha_hbm_A100 x 192,000) / (alpha_hbm_4090 x 240,000). EXPECTATION
  recorded now: because the device registry priors already carry the
  GDDR6 -> HBM2E ratio (192/240 = 0.80) in the feature columns, a perfect
  prior yields alpha ratio ~1.0; the measured deviation from 1.0 quantifies
  prior misfit and is a paper figure.
  [DECISION D6] RECOMMENDED report-only (no pass/fail band); alternative:
  a soft band (e.g. alpha_hbm ratio in [0.5, 2.0]) as a named sanity check.
  ADOPTED 2026-08-10: report-only.
- Calibrated MCER on the A100 (R1 refit), decode vs forward medians, versus
  the 4090 values; the cross-platform MCER comparison is a paper figure.
- [DECISION D7] Baseline companion on the A100: refit M0_flops, roofline
  (A100 peak FLOP/s and 1555 GB/s HBM2E bandwidth, vendor-cited in
  fit/baselines.py before use), and the layerwise regressor on the SAME
  T1 split under BOTH estimators; report the MAPE table; Wilcoxon one-sided
  + Holm on the T1 test split as SECONDARY significance. RECOMMENDED:
  include (analysis-only, no extra GPU time). Alternative: descriptive
  table only, no significance testing. ADOPTED 2026-08-10: include, with
  the secondary Wilcoxon + Holm.
- Spot-cell decomposition (if D5 yes): HF/ported/random ratios vs the 4090
  Follow-up B values; descriptive.
- A100 roofline fitted P_avg vs the 400 W envelope: the 4090 diagnostic
  (378 W vs a 150-175 W part) predicts the compensating-scale pathology;
  whether it reproduces on the A100 is recorded either way.

## 11. Measurement protocol (identical to the 4090 as-operated protocol)

Instruments: B = nvmlDeviceGetTotalEnergyConsumption (PRIMARY);
A = 100 Hz NVML power integration (cross-method check, expected ~5-10%);
C = Zeus ZeusMonitor (independent third instrument; Linux path fully
supported). Idle baseline measured per point and subtracted uniformly across
instruments. 4.0 s minimum window via measure_until_floor (inner_iters sized
from real wall time). Warmup 50 iters; torch.cuda.synchronize around timed
regions; thermal settle +/-1 C over 5 s. Regime-aware repeats: 10 for
decode/decoder_prefill, 5 for prefill/encode. CV flag threshold 5%. Clock
locking attempted via nvidia-smi (datacenter A100 normally permits it);
ACTUAL clocks logged per point regardless. Resumable last-write-wins JSONL;
resume skips only records with ok=True AND short_window=False. Provenance
gate: git-clean enforced; the A100 output files are allowlisted analogously
to the representativeness outputs (Step 6 implementation).

Seeds: every A100 point carries an EXPLICIT per-point init seed, derived
deterministically from the seed.json master seed and the point key by the
Step 6 grid builder (same pattern as the representativeness harness).
Cross-platform weight IDENTITY with the 4090 grid is NOT claimed (the 4090
main grid ran without per-point seeds); the value-variation contribution to
cross-platform deltas is bounded by the Follow-up B result (~6.5% worst case,
compute-bound fp16; init CV 0.2-5.6%), an order below the T2/T3 bands.

## 12. Artifacts and file layout

- Grid: configs/exp_002_a100.yaml, created in Step 6 by the grid builder,
  frozen and committed BEFORE any measurement; frozen copy lands beside the
  data at run start.
- Data: experiments/exp_002_size_sweep/a100/energy.jsonl
  (+ environment.json, validation reports, fit/ subdirectory for T1-T3
  artifacts). The validator is extended to this path with identical gates
  (ok, short_window, units gate) in Step 6.
- Every chunk ends with validator -> reports -> LOGBOOK/findings entries ->
  commit -> push before instance terminate.

## 13. Step 6 smoke checklist (gate before the first measured chunk)

1. Install stack on Lambda; snapshot environment.json.
2. pytest GPU subset green on the A100.
3. to_costs self-test and device resolution (a100 -> hbm2e).
4. Instrument agreement smoke: one sustained window; A-B within ~5-10%,
   C reporting.
5. OOM feasibility probe for the section 6 extension cells (7B s8192 prefill,
   7B ctx4096 decode).
6. One full measured point end-to-end through the resumable driver +
   validator on the A100 path.

## 14. What this amendment does NOT change

- Every 4090 pre-registered verdict stands as recorded (A-B 4.80% MET; M8 by
  AIC; E2 broad FAIL 50.37% / E1 PASS 14.34%; bake-off outcomes; Step 4
  representativeness FAIL 0.3303 with the Follow-up A/B decomposition and the
  narrowed flag). The paper reports them as registered.
- The frozen 4090 dataset (energy.jsonl, 296 points) is untouched.
- No new model forms, no new features, no re-selection. R1 changes the LOSS,
  not the form.
- H100 remains OUT of scope for this paper. EXP-003 (quantization) remains
  deferred.

## 15. Sign-off

Decisions adopted 2026-08-10 (all recommended options):
D1 = 25% / D2 = 30% / D3 = confirmatory at 30% / D4 = minimal 84-point grid /
D5 = include (spot cells + eager subset) / D6 = report-only / D7 = include
(with secondary Wilcoxon + Holm).

Approved by: M. Syed, 2026-08-10.

This amendment and the LOGBOOK Step 5 entry are committed together and pushed
before any Step 6 work begins.

## 16. Amendment 2 (2026-08-17): T2 calibration-set resolution and descriptive companions, recorded before any Step 7 fit

- Date: 2026-08-17 (America/New_York)
- Status: recorded per section 0 (a further dated amendment before the
  affected results are examined). No A100 fit of any kind has been computed.
  What has been examined: the validator reports and the matched-pair
  fp32/fp16 per-unit(B) ratios by phase (LOGBOOK, 2026-08-17). This section
  changes NO band, estimator, model form, split, evaluation set, or
  exclusion; it resolves one ambiguous cell name and pre-registers
  descriptive companions and post-verdict exploratory work.

16.1 Root cause of the ambiguity (documented in the LOGBOOK, 2026-08-17).
Section 5c's enc-dec budget (12 fp16 + 6 fp32) requires the fp16 decode
arms to omit the (s1024, ctx1024) center cell, and the frozen grid
(configs/exp_002_a100.yaml) does omit it, by explicit comment. Section 9's
"T5-small decode anchor-1024" (fp16) therefore names a cell that does not
exist in the frozen enumeration; the fp16 T5-small decode cells are
(s128, ctx1024), (s2048, ctx1024), (s1024, ctx128), (s1024, ctx2048). This
is internal to the amendment; the 98-point dataset is exactly the frozen
grid, and no further measurement is needed.

16.2 Resolution. The T2 calibration subset (8 cells, fp16, flash) is fixed
as the following seed-less keys (the per-point seed suffix joins each key
at expansion; matching is on the seed-less prefix). Rule for the enc-dec
decode cell, stated blind to any fit outcome: ctx = 1024 mirrors the
decoder calibration cells ("decode ctx1024"), and among the source-arm
cells s = 2048 is nearer the 1024 anchor in log space than s = 128. The
cell's repeat noise (CV(B) 15 percent) is immaterial to a one-parameter fit
over 8 cells whose target is the median of 10 repeats.
  1. decoder_only|GPT-2|prefill|fp16|flash|random|s1024|b1
  2. decoder_only|GPT-2|decode|fp16|flash|random|s1024|b1|ctx1024|k64|growing
  3. decoder_only|GPT-2-XL|prefill|fp16|flash|random|s1024|b1
  4. decoder_only|GPT-2-XL|decode|fp16|flash|random|s1024|b1|ctx1024|k64|growing
  5. encoder_only|BERT-base|encode|fp16|flash|random|s1024|b1
  6. encoder_only|BERT-large|encode|fp16|flash|random|s1024|b1
  7. encoder_decoder|T5-small|decode|fp16|flash|random|s2048|b1|ctx1024|k64|growing
  8. encoder_decoder|BART-base|encode|fp16|flash|random|s1024|b1
Cells 1 and 3 are the flash main-grid cells, not their eager companions
(section 5d). These keys are locked by tests/test_a100_calibration_cells.py
against the frozen enumeration and live in fit/a100_calibration.py, the
single source the fit script imports.

16.3 T2 sensitivity companion (DESCRIPTIVE, no verdict authority). T2 is
recomputed with cell 7 replaced in turn by each of the other three fp16
T5-small decode cells; each variant fits s* on its own 8 cells and evaluates
on its own remaining 76. Reported next to the verdict so the cell choice is
demonstrably immaterial or its fragility is on record.

16.4 Precision breakdown (DESCRIPTIVE, no verdict authority). For T1
(held-out) and T2 (remaining 76), the MAPE and signed relative error are
reported split by precision (fp16 / fp32) and by phase class (forward /
decode-like). Purpose: separate transfer of the memory-tier prior (D6,
alpha_hbm ratio, read on fp16 cells) from transfer of the precision prior.
Predictions for these tables are registered in the LOGBOOK (2026-08-17)
before any fit.

16.5 Post-verdict EXPLORATORY work (no verdict authority; run only after
T0-T3 are recorded in findings.md; reported as the 4090 R1 exploratory
was, under its own label):
  (a) A precision-split MAC model: to_mac split into fp16 and fp32 columns,
      all else identical to M8_split_dispatch, fitted under R1 on the same
      T1 split, and the T2 scalar transfer repeated with the corresponding
      4090 refit; quantifies what a device-level precision multiplier
      recovers.
  (b) Follow-up C (optional, about 1 GPU-hour on a fresh A100 instance):
      re-measure a small set of matched fp32 cells with TF32 enabled.
      Prediction registered now: the fp32/fp16 ratio collapses from about
      7.4x toward 2-3x because the GEMMs move to tensor cores. Follow-up C
      is a mechanism probe; it does not enter any confirmatory test.

16.6 Unchanged: T0-T3 bands (5 / 25 / 30 / 30 percent), R1 primary with
absolute NNLS secondary, M8_split_dispatch form, the 84 / 76 / 10 sets, spot
cells excluded from every fit and test, all section 10 secondaries.

Approved by: M. Syed, 2026-08-17 (plan approved in chat; this text records
it before any fit).
