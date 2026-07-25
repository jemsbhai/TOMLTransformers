# EXP-002 fit plan (Steps 2-3): frozen BEFORE any model fitting

Date: 2026-07-20. Status: APPROVED 2026-07-20, all four decisions resolved
in section 11; operative upon commit. No fit code has been written or run at
freeze time. This
document operationalizes the pre-registered modeling/extrapolation/baseline
sections of configs/exp_002.yaml against the frozen dataset. Where the
pre-registration is ambiguous, both readings are declared here and BOTH will
be computed and reported, so no post-hoc selection occurs.

## 1. Inputs

- Data: experiments/exp_002_size_sweep/energy.jsonl, frozen at 296 records
  (six commits, ba3347e..b597262), last-write-wins latest per key, all ok.
- Target y: per_execution_median_j["B"] / inner_iters, joules per ONE
  composite execution as defined in section 2. Instrument B is primary per
  the settled decision. A and C are not fit.
  (2026-07-20 correction: the median lives in per_execution_median_j; the
  per_execution_j field is the mean. 2026-07-23 UNITS correction, made
  pre-acceptance: the per_execution_*_j fields are per repeat WINDOW, i.e.
  inner_iters composite executions sized by measure_until_floor to the 4 s
  floor. The first fit run surfaced the error unmistakably, R2 ~ 0 with a
  ~539 J intercept matching the window energy the floor targets by
  construction and negative per-token values, and was discarded before any
  result was accepted or committed. A fit-time consistency gate now
  cross-checks the derived target against the independently stored
  per_unit_j on every record: forward phases y ~ per_unit, decode y ~
  per_unit x decode_tokens.)
- Uncertainty carried per point: window std over repeats / inner_iters
  (joules per composite execution), used for error bars and the decode
  per-token propagation, NOT as fit weights (see section 6).

## 2. Execution boundary table (verified against workload code)

One measured execution contains, per phase (verified in workloads/*.py):

| phase            | execution contents                                              |
|------------------|-----------------------------------------------------------------|
| prefill          | 1 decoder forward over s (embed, L layers, final norm, lm_head) |
| encode           | 1 encoder forward over s (text) or fixed patches+CLS (ViT)      |
| decoder_prefill  | cross-cache build (src) + decoder prefill over tgt_len          |
| decode (growing) | cross-cache build (enc-dec only) + decoder/self prefill over    |
|                  | ctx + 64 decode steps with growing self-cache                   |

Decoder-only decode has no cross-cache; its execution is prefill(ctx) + 64
steps. All 122 decode records in the dataset are growing mode; fixed_step was
not swept. The encoder pass of enc-dec phases is unmeasured setup and MUST
NOT appear in features.

## 3. Feature bridge (design-matrix construction)

For each record, resolve TransformerConfig via architectures.configs.get
(spec.model), device="rtx4090", prec=spec.precision, attention kind mapped
spec "flash"->"flash", "eager"->"standard" (front-end naming). Emit the
energy_model.FEATURES record by summing front-end phase calls that mirror
section 2 exactly:

- prefill: decoder.prefill(cfg, s)
- decode (decoder-only): decoder.prefill(cfg, ctx) + decoder.decode_total(
  cfg, ctx, 64)
- encode: encoder front-end encode counting at the workload's effective
  sequence (text: s; ViT: native num_patches+1, seq_len ignored)
- decoder_prefill: enc-dec front-end decoder_prefill(src, tgt_len) including
  the cross-K/V projection
- decode (enc-dec): enc-dec decoder_prefill(src, tgt_ctx) + sum of
  enc-dec decode_step over t = 1..64 at growing target context

Implementation gates before any fit: (a) bridge unit test asserting emitted
keys match energy_model.FEATURES; (b) one hand-checked feature record per
architecture class committed to the test suite; (c) runtime assert that
every one of the 296 records resolves to a config and a phase call.

## 4. Dispatch features (decision D1)

The measured workloads contain no Python-loop-per-timestep dispatch in the
signals-paper sense for forward phases, but decode executions dispatch 64+1
Python-level forward calls. PROPOSED (D1): n_launches = count of Python-level
forward dispatches per execution (prefill/encode/decoder_prefill = 1;
decode = 65 for decoder-only, 65 for enc-dec counting its internal prefill
as 1). n_fused_steps = 0 everywhere (no fused-sequential kernels exist in
these workloads), so M4 and M9 degenerate to M3 and M8 plus an AIC penalty;
they remain in the family for completeness and this degeneration is stated
in the paper. ALTERNATIVE: all dispatch features 0, making M3/M4/M8/M9
inert. Decision required before fitting. (Resolution: see section 11, D1',
amended 2026-07-20.)

## 5. Splits

- Step 2 main table (exploratory generalization; decision D2): stratified
  random 80/20 split, seed 42 (seed.json master), stratified on (arch,
  phase-class, precision). fit_and_select over M0-M9; selection by AIC on
  train (pre-registered), BIC reported; held-out R2/MAPE headline.
- Step 3 confirmatory extrapolation (pre-registered): both declared readings
  computed and reported (decision D-E on which is PRIMARY in the abstract):
  - E1 strict-literal: train = forward-phase points (prefill, encode) of the
    pre-registered train models with s <= 1024; predict held-out models'
    forward points at s = 2048.
  - E2 broad: train = ALL phases of train models with every sequence
    dimension <= 1024; predict ALL held-out-model points with any sequence
    dimension = 2048.
  - Recorded clarification 1: ViT-L/16 has no sequence axis (grid deduped to
    one point per precision at native 197); its pre-registered "at s=2048"
    target is evaluated as held-out-MODEL prediction at native shape, both
    precisions.
  - Recorded clarification 2: held-out target points include both precisions
    wherever both exist.
  - Pass band: held-out MAPE <= 0.25 per pre-registration; reported pooled
    and per architecture class; the result stands whichever way it falls.

## 6. Estimator, weighting, robustness (pre-specified)

- Primary: unweighted NNLS on absolute joules, exactly as energy_model.py
  implements and as the signals-paper precedent.
- Pre-specified robustness reports (computed alongside, never used to pick
  the winner): (R1) relative-error NNLS (rows of A and y scaled by 1/y_i);
  (R2) refit excluding the five CV(B)-flagged points; (R3) refit excluding
  the three inner_iters <= 2 points. Report coefficient stability and
  held-out MAPE deltas for each.

## 7. Derived quantities (after selection, from the winning model)

- Pooled A-B agreement median over all 296 points, computed exactly, to
  settle the pre-registered <= 5% target (currently known per-class:
  forward 5.53%, decode-like 4.03%).
- Calibrated MCER per point: fitted memory energy over fitted compute energy
  using the winning model's column groupings; reported by phase, giving the
  calibrated prefill-vs-decode phase transition and the enc-dec
  cross-vs-self isolation from the two decode sub-sweeps.
- Clean decode per-token: (E_exec(decode) - E_component(prefill part)) / 64,
  uncertainty propagated from both measured points' repeat std. The
  subtracted component is the MATCHED MEASURED point where it exists
  (decoder ctx in {128,512,1024,2048}; enc-dec tgt-varies sub-sweep at
  src=1024 where decoder_prefill(1024, tgt_ctx) was measured) and the
  winning model's PREDICTION where it does not (decoder ctx=4096; enc-dec
  src-varies sub-sweep), flagged per row as measured-subtracted vs
  model-subtracted.
- Eager-vs-flash energy ratio vs s from the attention_compare points, and
  the fp32/fp16 fitted multiplier discussion (Fork 2).

## 8. Baselines (Step 3 bake-off, pre-registered list)

Identical held-out points and metric (MAPE) for all methods:
- flops_mac: M0 from the family (single calibrated total-TO term).
- roofline: E = P_avg * max(FLOPs / peak_FLOPs, bytes / peak_BW) with
  RTX 4090 Laptop peak FP throughput and bandwidth taken from vendor
  documentation, citations recorded in the baselines module; P_avg fitted on
  the train split (one parameter, same information budget as M0).
- layerwise_regressor (NeuralPower-style; decision D3): NNLS on RAW
  structural counts (unweighted op and byte counts per phase, no to_costs
  priors), same column structure as the winning model. This isolates the
  value of the physics priors specifically.
- Statistics: paired per-point absolute percentage errors, Wilcoxon
  signed-rank vs each must-beat baseline, Holm-Bonferroni across the three
  comparisons, alpha 0.05. differentiate_from competitors are cited, not
  re-implemented, except LLMCO2 remains a stretch goal.

## 9. Deliverables

- scripts/fit_exp002.py: builds the bridge, runs sections 5-8, writes
  experiments/exp_002_size_sweep/fit/{fit_results.json, fit_report.txt,
  per_point_predictions.jsonl} and prints the summary tables.
- Bridge unit tests per section 3 gates.
- findings.md entry with the confirmatory results (whichever way they fall)
  and LOGBOOK addendum; figures are Step 8, not here.

## 10. Decisions required before any fit runs

- D1: dispatch feature definition (proposed: per-execution Python dispatch
  count; alternative: zeros).
- D2: Step-2 main-table split (proposed: stratified 80/20 seed 42).
- D-E: which extrapolation reading (E1 strict / E2 broad) is PRIMARY in the
  paper; both are computed regardless.
- D3: layerwise_regressor operationalization as specified.

Sign-off, with any modifications, is recorded by committing this file plus a
LOGBOOK line BEFORE fit code lands.

## 11. Resolutions (2026-07-20, approved before any fit code)

- D1: originally ADOPTED as proposed at sign-off, then AMENDED the same day,
  still pre-fit, to D1' after code review found that architectures/common.py
  already implements a documented kernel-launch convention designed to drive
  the M3+ dispatch term: each GEMM, norm, and embedding gather counts one
  launch; fused elementwise ops count zero; standard attention counts 3 vs
  flash 1; KV-cache reads/writes count 0. D1': the bridge uses these
  built-in GEMM-level launch counts exactly as emitted and summed through
  the execution composition; no overwriting with flat Python-dispatch
  counts. Rationale: matches the signals-paper So semantics (CUDA kernel
  launches), carries the launch-bound small-context-decode floor the QC data
  showed, and is sequence-independent per forward so the column decouples
  from the TO features. These launch counts are a structural PROXY for
  kernel launches, not a profiled count; the fitted coefficient absorbs the
  proxy scale, and this is stated in the paper. n_fused_steps remains 0 by
  construction; the M4/M9 degeneration statement is unchanged. Amendment
  approved with an explicit thoroughness condition, honored by
  launch-specific invariants in the bridge test gates.
- D2: ADOPTED. Stratified random 80/20, seed 42, strata (arch, phase-class,
  precision).
- D-E: E2 (broad reading) is PRIMARY; E1 (strict-literal) is computed and
  reported alongside. ViT-L/16 evaluated at native shape per recorded
  clarification 1.
- D3: ADOPTED as specified (raw structural counts, no to_costs priors).


## 12. Section-8 operationalization (2026-07-24, approved pre-bake-off)

- Roofline constants, cited in fit/baselines.py: RTX 4090 Laptop GPU
  (AD103, GN21-X11), 9,728 CUDA cores, rated boost 2,040 MHz at the 150 W
  TGP configuration, 256-bit GDDR6 at 18 Gbps, 576.0 GB/s (TechPowerUp GPU
  Database; TechSpot Feb 2023 review; VideoCardz). Peak FP32 = 2 x 9728 x
  2.040e9 = 39.69 TFLOP/s (formula stated). Peak FP16 = 2 x peak FP32 =
  79.38 TFLOP/s, recorded explicitly as the standard Ada dense tensor-core
  assumption for the cuBLAS/SDPA paths the workloads use. P_avg is the
  roofline's single fitted parameter (LS through the origin). Sensitivity
  line only: the frozen dataset's maximum sustained median SM clock,
  2,325 MHz, gives a 45.24 TFLOP/s machine-observed ceiling.
- Raw structural counts for roofline and layerwise are recovered EXACTLY by
  inverting to_costs with the same constants the bridge used.
- Layerwise regressor (D3, refined as D5): NNLS on the winner's selected
  support with the physics priors stripped: columns raw_macs, sram_words,
  hbm_words, n_launches, intercept. to_nonlinear is excluded: it aggregates
  ops with different TO costs (no exact raw inverse) and the winner fitted
  it to zero under both estimators.
- Significance sets: PRIMARY = the 58-point main test split under the
  pre-registered absolute estimator (best paired-test power); SECONDARY =
  the E2 predict set (n=14). One-sided Wilcoxon signed-rank (winner APEs
  smaller), Holm-Bonferroni across the three must-beats, alpha 0.05. R1
  companions of everything are reported under the existing exploratory
  label, MAPE only.
- Device-registry correction (2026-07-24, discovered while wiring the
  inversion): to_costs registered "rtx4090" with the desktop GDDR6X tier;
  the Laptop GPU uses GDDR6 at 18 Gbps. Corrected in to_costs.py (desktop
  entry preserved as "rtx4090_desktop"). Effect: a uniform 240/232 rescale
  of the to_hbm feature column, absorbed exactly by the fitted coefficient;
  predictions, fit quality, AIC/BIC, MCER (a ratio), extrapolation MAPE,
  and every pre-registered verdict are invariant. Only the printed to_hbm
  coefficient shifts by that factor versus the aac684f artifacts, and the
  regenerated artifacts supersede them.
