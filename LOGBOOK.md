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


---

## EXP-002: Calibrated transformer-inference energy (size + precision sweep), RTX 4090

**Date:** 2026-05-29 (America/New_York)
**Researcher:** Muntaser Syed
**Type:** computational (real PyTorch inference + GPU power/energy measurement)
**Status:** planned
**Pre-registration note:** This is the plan, written before any measurement. It is the first experiment in this line to produce measured energy in joules and to fit the M0-M9 family against it. Hypotheses, variables, controls, the extrapolation split, the representativeness acceptance band, and the baseline set are all fixed here, in advance, to prevent post-hoc rationalization (HARKing). Config: `configs/exp_002.yaml`. Seeds: `experiments/exp_002_size_sweep/seed.json`.

### Hypothesis
Following EXP-001 (which showed the prefill/decode MCER phase transition in prior-weighted TO ratios), we now test whether that picture survives contact with measured energy, and whether the transistor-operation features explain energy better than FLOPs once calibrated.
- **H1 (prefill).** Measured prefill energy is compute-dominated: fitted MCER < 1 and falling roughly as 1/s.
- **H2 (decode).** Measured decode energy is memory-dominated: fitted MCER >> 1 and approximately model-size-independent (EXP-001 predicted ~70 from the per-parameter memory/compute cost gap).
- **H3 (split models).** The split-feature models (M5+) achieve lower AIC/BIC and higher held-out R2 than M0 (calibrated FLOPs), and their fitted SRAM/HBM and softmax/MAC coefficients quantify whether the 45 nm reference ratios need a correction at TSMC 4N.
- **H4 (extrapolation + baselines).** A model fit on small architectures and short sequences predicts the held-out large model at s = 2048 within MAPE <= 0.25, and beats the FLOPs/MAC, Roofline, and layer-wise baselines on the identical held-out split.
- **H5 (attention).** Measured eager-attention energy carries an O(s^2) penalty over FlashAttention that grows with sequence length (EXP-001 predicted the standard/flash MCER ratio rising from ~1.6 at s=2048 to ~19 at s=16384).

### Independent variables
- model: 14 configs across the three classes (decoder-only DistilGPT2/GPT-2/GPT-2-medium/GPT-2-large/GPT-2-XL; encoder-only DistilBERT/BERT-base/BERT-large/ViT-B16/ViT-L16; encoder-decoder T5-small/T5-base/BART-base/BART-large). 7B/8B deferred to EXP-004.
- precision: {fp16, fp32}, paired (Fork 2).
- phase: {prefill, decode}, measured separately.
- prefill seq_len: {128, 256, 512, 1024, 2048}; decode context_len: {128, 512, 1024, 2048, 4096}.
- attention kind (sub-sweep): {eager, flash} on {DistilGPT2, GPT-2}, seq {512,1024,2048,4096}, fp16.

### Dependent variables / metrics
- Measured energy per call (joules), via three independent instruments. Ours: A = 20 Hz pynvml power-sample integration minus idle (the method the prior TOML papers used); B = our direct read of the Ada hardware energy counter (nvmlDeviceGetTotalEnergyConsumption) minus idle. Cross-check: C = Zeus ZeusMonitor, an independent peer-reviewed implementation. A and B are always present; C is supplementary and never the sole source for any number.
- Instrument agreement: pairwise |dE_i - dE_j| / dE_j across {A,B,C} per point (pre-registered median target <= 5%).
- Fitted coefficients of M0-M9 (NNLS); held-out R2 and MAPE; AIC/BIC.
- MCER recomputed from fitted coefficients (dimensionless).
- Held-out MAPE of each baseline vs this model (paired; Holm-Bonferroni corrected).

### Control conditions
- Held constant: device = rtx4090 (GDDR6X, 4N), batch = 1, warmup = 50 iters, torch.cuda.synchronize around the timed region, thermal settle to +/-1 C over 5 s, per-point idle baseline (3 s) subtracted, GPU clocks locked where the laptop permits (actual clocks logged per point regardless), 5 physical repeats per point with CV gate 5%.
- Single init seed (42) across the full grid; multiple init seeds confined to the representativeness check (where weight/input variation is the object of study), so the grid is not confounded by seed variance while the Fork-1 assumption is still tested.
- Baselines (FLOPs/MAC = M0, analytical Roofline, layer-wise learned regressor) evaluated on the identical held-out split with the identical metric. Differentiate-from set (LLMCO2, Accelergy/Timeloop, LLMCarbon, Zeus/ML.ENERGY, Li et al. 2022) cited, not necessarily beaten head-to-head.
- Baseline being improved upon: EXP-001's prior-weighted TO ratios (uncalibrated) and the M0 single-term framing.

### Protocol
1. **Set up and check Zeus (cross-check instrument C).** `pip install zeus`, confirm it imports and that ZeusMonitor returns nonzero energy on this box (NVML/nvml.dll ships with the driver, so the GPU path should work; Windows is undocumented, RAPL/AMD/daemon paths are Linux-only). If it cannot return energy here, record C as "unavailable" and proceed with our instruments A and B; C is never required and is never the sole source. Our own A+B measurement is built and run regardless.
2. **Snapshot environment.** `python scripts/snapshot_env.py experiments/exp_002_size_sweep/environment.json` on the measurement machine (records driver, CUDA, package versions; never hand-authored).
3. **Freeze config.** Copy `configs/exp_002.yaml` to `experiments/exp_002_size_sweep/config.yaml`; the frozen copy is what runs.
4. **Commit clean.** Repo has no uncommitted changes before any measurement; record the commit SHA in this entry as an addendum.
5. **Build the measurement harness** (next step after this commit): reads the frozen config, drives prefill/decode/attention sweeps under both instruments, writes `results/energy.json` with per-point mean/std/CV, actual clocks, temperatures, and any OOM skips.
6. **Representativeness check.** At the two spot-check points, compare random-init vs HF-pretrained (gpt2, bert-base-uncased) energy across 3 init seeds; accept only if within DDEV band 0.33, else flag the sweep and log the failure as a finding. (Real-weight runs trigger a transformers model download; per project rules the user confirms that download explicitly.)
7. **Fit and select.** Fit M0-M9 by NNLS on measured energy, select by AIC (report BIC), report held-out R2/MAPE, recompute MCER from fitted coefficients.
8. **Baseline bake-off** on the pre-registered held-out split; paired comparison with Holm-Bonferroni correction.
9. **Reviewer-adversary pass** (lab-runner 9c) before any result enters findings.md.

### Environment
- **Hardware:** RTX 4090 Laptop GPU (Ada, 16 GB GDDR6X, hard VRAM cap), i9-14900HX, 64 GB RAM.
- **Software:** see `experiments/exp_002_size_sweep/environment.json` (snapshotted at run time). Measurement extras: torch 2.6.0+cu124, transformers, pynvml, zeus, pyyaml.
- **Git commit:** RECORDED AT FREEZE (addendum, before measurement).
- **Seeds:** `experiments/exp_002_size_sweep/seed.json` (master 42; representativeness seeds 42/1234/2025).

### Results
_To be completed after the run. No values may be entered here until measured._

### Observations
_To be completed after the run._

### Interpretation
_To be completed after the run._

### Artifacts
- Config (source): `configs/exp_002.yaml`
- Frozen config / environment / seeds / results: under `experiments/exp_002_size_sweep/`
- Harness: to be added (next step), entry point will be recorded here.
- Instrument layer: `src/tomltransformers/measure/instruments.py` (+ `tests/test_instruments.py`), committed 64c8f0c.

### Addendum 2026-05-29 — instrument layer up; Zeus-on-Windows prerequisite RESOLVED
The three-instrument measurement layer is built and all 91 tests pass. Smoke test
on the actual Windows + RTX 4090 (Ada) machine confirms all three instruments are
live: A (our 20 Hz nvmlDeviceGetPowerUsage integration), B (our direct
nvmlDeviceGetTotalEnergyConsumption read), C (Zeus ZeusMonitor, with
approx_instant_energy=True). Protocol step 1 is therefore satisfied:
`measurement.instrument_C` is AVAILABLE on this box, not a fallback; B and C read
identical energy on a shared window (same hardware accumulator), validating B.
One known issue carried to the runner: on an uncontrolled window (no idle
subtraction, no warmup-excluded timing, 19 samples) instrument A read ~27% high
vs B; this is an upper bound, not a result, and the runner must re-measure A-vs-B
agreement under thermal-settled, idle-subtracted conditions before any energy is
reported. See findings.md (2026-05-29 instrument-layer entry) for detail. No
measurement of the EXP-002 grid has been run; Results/Observations/Interpretation
remain open.

### Addendum 2026-07-18 - Production sweep launched; chunk 1 validated
Build-phase decisions since the 2026-05-29 addendum (100 Hz instrument-A
sampling, regime-aware repeats 5 forward / 10 decode-like, measure_until_floor
with a 4 s window floor, JSONL last-write-wins resume, --max-hours budget) are
recorded in findings.md, the dated amendment inside configs/exp_002.yaml, and
the commit history; not re-narrated here.

- 2026-07-17 16:54 EDT: sweep launched at commit ba3347e via
  `python scripts/run_exp002.py --max-hours 8`, after a clean-tree preflight.
  The actual results artifact is `experiments/exp_002_size_sweep/energy.jsonl`
  (supersedes the `results/energy.json` path named in Protocol step 5).
- Chunk 1 stopped on budget 2026-07-18 00:58 EDT: 85/296 points, all ok, zero
  failed / OOM / short-window. Coverage: decoder-only, flash, fp16+fp32
  (DistilGPT2/GPT-2/GPT-2-medium/GPT-2-large complete, GPT-2-XL partial).
  Median pace 292 s/point; ~17 h projected for the remaining 211 points
  (~2-3 more chunks).
- `pytest -q tests/test_driver.py`: 11 passed, closing the previously
  unverified `test_max_hours_stops_early_and_resumes`; the real chunk also
  exercised the budget stop and clean summary write end to end.
- QC pass added and run: `scripts/validate_exp002.py` (read-only) writes
  `validation_report.{txt,json}` beside the data. Two WARNs, both explained,
  neither systematic: CV(B)=16.3% on GPT-2-medium fp16 ctx128 decode
  (smallest-context decode is the noisiest regime; flagged and kept per the
  no-retry CV policy) and inner_iters=2 on GPT-2-large fp32 ctx4096 decode
  (4 s wall floor still met). Detail: findings.md, 2026-07-18 entry.
- Decision: continue the sweep unchanged. Inter-chunk ritual: run the
  validator, commit the harvest (energy.jsonl, sweep_summary.json, refreshed
  environment.json, frozen config, validation reports), relaunch the
  identical command.
- Scope note (2026-07-18, before any A100 data exists): the A100 plan moves
  from a time-boxed collaborator machine to self-provisioned Lambda Labs
  instances (the collaborator run is now optional), making the full shared
  grid feasible cross-platform. The formal pre-registration amendment
  (instance type, GPU-hour budget, transfer comparison and acceptance bands,
  A100-only extrapolation targets) remains to be written BEFORE any A100
  measurement, per Step 5 of the plan.

Results/Observations/Interpretation for EXP-002 remain open pending the fit.

### Addendum 2026-07-18 - Chunk 2 validated; 158/296
Chunk 2 (commit f5ab202, `--max-hours 8`, 8.08 h): 73 new points, 85
resume-skipped, all ok; zero failed / OOM / short-window at 158/296 (53.4%).
Resume semantics exercised at scale for the first time and behaved exactly as
specified. QC: findings.md 2026-07-18 chunk-2 entry; validation reports
recommitted with the harvest. All seven validator WARNs fall in the two
known-benign families (fp16 small-target decode CV; low inner_iters at large
fp32 decode with the 4 s wall floor met). Decision: continue unchanged.
Remaining ~138 points (enc-dec, ViT, eager block) projected ~12-15 h.

### Addendum 2026-07-20 - EXP-002 measurement campaign COMPLETE: 296/296
Grid exhausted naturally on chunk 6 (stopped_early=False). Ledger: 85
(ba3347e, 8 h) + 73 (f5ab202, 8 h) + 81 (d434da2, 8 h) + 30 (99024e8, 3 h
budget) + 24 (33592b3, 2 h budget) + 3 (b597262, 0.36 h); ~29.5 h wall total
versus the ~30 h pre-launch estimate. 296/296 ok; zero failed / OOM /
short-window / superseded across the campaign; resume semantics behaved
exactly as specified across five restarts. Full-grid QC is the findings.md
2026-07-20 entry (the paper's measurement-quality record). Final WARN
ledger: 8, all inside the two characterized benign families (fp16-dominated
small-target decode CV; low inner_iters at large fp32 decode with the 4 s
wall floor met); frozen as-is. The OOM skip-and-log path never fired
(everything fit in 16 GB) and remains unexercised going into A100 work.
Decisions:
- energy.jsonl is FROZEN as the EXP-002 RTX 4090 dataset; any future
  re-measurement appends under a new commit via last-write-wins.
- Next: Step 2, fit the M0-M9 NNLS family against instrument B per the
  pre-registration (CPU-only). No further 4090 measurement is planned for
  EXP-002 except the Step 4 representativeness spot checks, which require
  explicit go-ahead for HF downloads.
- The A100/Lambda pre-registration amendment (Step 5) must still be
  written BEFORE any A100 measurement.

### Addendum 2026-07-20 - Step 2-3 fit plan frozen (pre-fit)
experiments/exp_002_size_sweep/fit_plan.md committed as the operative
analysis plan BEFORE any fit code or results exist. Decisions resolved at
sign-off: D1 dispatch features = per-execution Python dispatch counts
(decode 65, forward phases 1, n_fused_steps 0); D2 main-table split =
stratified 80/20, seed 42, strata (arch, phase-class, precision); D-E
extrapolation = E2 broad reading primary with E1 strict-literal reported
alongside, ViT-L/16 evaluated at native shape; D3 layerwise baseline = raw
structural counts without to_costs priors. Ambiguity policy: both
pre-registration readings computed and reported, no post-hoc selection.
Next: feature bridge with unit-test gates (fit_plan section 3), then
scripts/fit_exp002.py.

### Addendum 2026-07-20 - Pre-fit amendments: D1 -> D1', target field name
Two dated amendments to fit_plan.md, both made BEFORE any fit code or
results exist: (1) D1 amended to D1': the dispatch feature uses the
GEMM-level kernel-launch convention already built into
architectures/common.py and attention.py (each GEMM/norm/embedding gather =
1 launch, fused elementwise = 0, standard attention 3 vs flash 1, KV-cache
ops 0), summed through the execution composition, instead of flat
Python-dispatch counts. Signals-paper So semantics; a structural proxy
whose scale the fitted coefficient absorbs. Amendment approved with an
explicit thoroughness condition, honored via launch-specific invariants in
the bridge test gates. (2) Target field name corrected: the fit reads
per_execution_median_j["B"], the plan's committed median intent; the
records' per_execution_j field is the mean. Next: feature bridge plus test
gates; scripts/fit_exp002.py only after the gates are green.

### Addendum 2026-07-23 - First fit run INVALID: target units error, caught pre-acceptance
The first execution of scripts/fit_exp002.py produced physically impossible
outputs: R2 ~ 0 with all of M2-M9 identical, a zero memory coefficient, a
~539 J intercept, and negative decode per-token energies. Diagnosis: the
record fields per_execution_*_j are per repeat WINDOW (inner_iters composite
executions, sized by measure_until_floor to the 4 s floor), not per
execution, so the fitted target was nearly constant by construction while
the features varied by orders of magnitude. NOT a measurement problem: the
GPU was uninvolved (CPU-only analysis) and the frozen dataset is unaffected
(its physics QC already passed). Fix, recorded as a dated units correction
in fit_plan.md section 1: y = per_execution_median_j["B"] / inner_iters
(joules per composite execution, matching the plan's section 2 boundary
table), std treated identically, and a new fit-time consistency gate
cross-checks the derived target against the independently stored per_unit_j
on all 296 records (forward phases y ~ per_unit; decode y ~ per_unit x
decode_tokens), aborting on any disagreement. The invalid artifacts were
never committed and are overwritten by the corrected run. Lesson recorded:
the bridge gates verified features only; targets now have their own gate.

### Addendum 2026-07-24 - Confirmatory fit landed; exploratory R1 extension approved (post-hoc, labeled)
The corrected fit ran cleanly (units gate passed on all 296; artifacts at
commit 50a15b3). Pre-registered verdicts, recorded as they fell: pooled A-B
median 4.80%, <= 5% target MET; winner M8_split_dispatch (R2_test 0.987;
M9 degenerates to M8 plus the AIC penalty exactly as predicted pre-fit);
extrapolation E2 broad (PRIMARY) FAIL at 50.4% pooled MAPE, E1
strict-literal PASS at 14.3%, both stand. Calibrated MCER confirms the
phase transition (decode ~12.6-13.1 vs forward ~3-4 median). Central
methods finding: strong heteroscedasticity under the pre-registered
absolute-NNLS estimator (R2 0.987 with MAPE 88% on the same held-out set),
with coefficient distortion (to_sram 1.97e-12, ~350x to_hbm, unstable under
R3) that also breaks monotonicity in the three model-subtracted per-token
rows; the pre-specified R1 robustness probe reaches 20.1% test MAPE with
physically sensible coefficients. DECISION (post-hoc, approved before the
rerun): extend scripts/fit_exp002.py with an explicitly labeled EXPLORATORY
section applying the R1 estimator to both extrapolation readings, MCER, and
the model-subtracted per-token rows; confirmatory outputs unchanged; the
exploratory results inform the paper's estimator discussion and the A100
pre-registration amendment only, and are never presented as pre-registered.

### Addendum 2026-07-24 - Exploratory R1 run complete; results recorded
Rerun at commit aac684f: confirmatory sections byte-identical to the
50a15b3 run (determinism check passed); units gate green on all 296.
Exploratory R1 headline: E2 pooled MAPE 15.24% on the identical split that
fails at 50.37% under the pre-registered absolute estimator (E1 13.25%),
attributing the E2 failure to estimator scale-weighting rather than the TO
feature set; MCER under R1 recovers the EXP-001 regime split (forward
0.22-0.43, decode 9.8-10.5); per-token monotonicity restored on the three
model-subtracted rows. Full record: findings.md 2026-07-24 entry; artifacts
committed with this addendum. Next, in order: Step 3 baselines (roofline
with cited RTX 4090 Laptop peak specs, layerwise regressor per D3, Wilcoxon
plus Holm), then Step 4 representativeness (requires explicit go-ahead for
HF downloads), then the Step 5 A100/Lambda pre-registration amendment,
which will pre-register R1 as the primary estimator for the cross-platform
test before any A100 data exists.

### Addendum 2026-07-24 - Section-8 baselines built; device registry corrected
Roofline constants sourced and cited (RTX 4090 Laptop GPU: 9728 cores, 2040
MHz rated boost at 150 W, 256-bit GDDR6 18 Gbps, 576.0 GB/s; TechPowerUp /
TechSpot / VideoCardz concordant; peak FP32 39.69 TFLOP/s by stated formula;
FP16 = 2x as a recorded Ada tensor-core assumption; 2325 MHz machine ceiling
from energy.jsonl as sensitivity only). fit/baselines.py (roofline with
fitted P_avg; layerwise NNLS on raw structural counts per D3/D5),
fit/stats.py (one-sided Wilcoxon, Holm), tests, and the section-8 extension
of scripts/fit_exp002.py landed; operationalization frozen as fit_plan
section 12 BEFORE the bake-off runs. Found and corrected during the build:
to_costs registered rtx4090 with the desktop GDDR6X tier; the Laptop GPU is
GDDR6 (18 Gbps). Correction is verdict-invariant (uniform 240/232 rescale of
the to_hbm column, absorbed exactly by NNLS; only the printed to_hbm
coefficient shifts vs the aac684f artifacts, which the regenerated artifacts
supersede). Desktop entry preserved as rtx4090_desktop; self-test and a new
unit test lock the mapping.

### Addendum 2026-07-24 - Bake-off complete; Steps 2-3 closed on the 4090
Section-8 run at commit db1f984 (registry-correction footprint verified:
to_hbm coefficients scaled by exactly 232/240; no verdict moved).
Pre-registered primary (58-pt test, absolute): winner beats layerwise
(Holm p=5.1e-4), does not beat M0 or roofline on MAPE; secondary E2 (n=14)
nothing significant. Diagnostics recorded: roofline P_avg fits to 378 W,
2.2-2.5x the part's power envelope, exposing its time model as
compensating rather than physical; priors-stripped layerwise zeroes
raw_macs and finishes last. Exploratory R1 companion (labeled): winner_R1
20.10% leads layerwise_R1 24.61%, roofline_R1 30.21%, M0_R1 32.41%.
Verdicts stand as registered; full record in findings.md (second
2026-07-24 entry). Decision reaffirmed: the Step 5 amendment names R1 as
primary for the A100, citing this evidence, before any A100 data exists.
Next: Step 4 representativeness (gated on explicit go-ahead for HF
downloads), then Step 5.

### Addendum 2026-07-24 - Step 4 representativeness: operationalization frozen BEFORE the run
Pre-registered spec (configs/exp_002.yaml `representativeness`) operationalized
and frozen before any Step-4 measurement; approvals recorded in-session:
- Cells: 12 = {GPT-2 prefill s512, GPT-2 decode ctx512 K64 growing, BERT-base
  encode s512} x {pretrained, random seed 42/1234/2025}; fp16, flash, batch 1.
  Recorded interpretation: BERT-base has no decode phase, so it contributes the
  compute-bound point only; the decode point is decoder-only by construction.
  fp16 only (deployment-relevant precision; ~50 min GPU); fp32 available as a
  follow-up only if the band is approached.
- Metric: y = per_execution_median_j[B]/inner_iters per cell; per (point, seed)
  ratio |y_rand - y_pre|/y_pre; VERDICT = max over the 9 ratios <= 0.33
  (pre-registered band); mean-based per_unit ratios reported as a secondary
  column. Pretrained arm input-seeded at 42; matching seeds yield identical
  input ids across arms (gpt2 vocab == config vocab), so weights are the only
  varied factor.
- APPROVED parity fix (pre-run; the pretrained builder paths were NEVER
  exercised by the frozen sweep): pretrained prefill now runs the bare
  transformer stack + lm_head on the LAST token only (HF's LMHeadModel
  all-position logits are ~2e10 extra MACs at s512 on GPT-2, a ~45% compute
  confound); the decode cache build runs the bare stack with no head (parity
  with prefill_into_cache); attn_implementation="sdpa" requested at load with
  graceful fallback, both decoder and encoder loaders. BERT parity note: the
  HF pooler (d->d + tanh on [CLS]) matches our pooled head's magnitude;
  embedding adds are negligible.
- Machinery: PointSpec gains an optional `seed` field (default None: every
  frozen-sweep key and behavior byte-identical); when set, torch RNGs are
  seeded inside the builder dispatch so measure_until_floor rebuilds draw
  identical weights/inputs; seed joins the key and the record. HF cache
  handling promoted to measure/hf_cache.py (ephemeral_hf_repos: delete only
  what this run fetched, keep pre-existing; cleanup on crash).
- Harness: scripts/run_exp002_representativeness.py; provenance gate (clean
  tree), resumable JSONL experiments/exp_002_size_sweep/representativeness.jsonl,
  regime repeats (5 forward / 10 decode), 4 s floor, 100 Hz; report artifacts
  representativeness_report.{txt,json}; verdict logged as a finding either way.

### Addendum 2026-08-10 - Step 4 verdict: FAIL 0.3303 vs 0.33; sweep flagged as pre-registered
Two harness defects were found and fixed before the verdict (recorded here
for the run's honesty): an analysis KeyError from hand-duplicated key
construction (analysis now derives every key from the cell objects) and the
provenance gate blocking its own resume (gate now permits untracked copies
of the harness's own three outputs only). All 12 cells measured ok on the
first pass; the fixes changed analysis and gating only, no re-measurement.
Verdict: FAIL by the narrowest margin (one GPT-2 prefill fp16 cell at
0.3303; eight of nine ratios in-band), with strong regime structure: decode
0.003-0.011 (immune), BERT encode 0.13-0.16, GPT-2 prefill 0.24-0.33 with
pretrained consistently higher and slower. Implementation-free signal:
random-arm init-seed CV 5.60% (prefill) vs 0.72% (decode). Attribution
between HF-implementation overhead and genuine DDEV is entangled and
recorded as such; full analysis in findings.md 2026-08-10. The 2026-07-24
fp32-follow-up trigger FIRED (band exceeded). Decision pending (Muntaser):
scope of follow-ups (fp32 prefill cells; pretrained-weights-into-our-stack
control) and the A100 weights policy, all to be settled in or before the
Step 5 amendment.

### Addendum 2026-08-10 - Follow-ups approved: A (fp32 cells) now, B (weights-porting control) next
A: four fp32 cells (GPT-2 prefill s512 x {pretrained, random 42/1234/2025})
added to the representativeness harness under the fired 2026-07-24 trigger,
in a separate, labeled FOLLOW-UP section; the pre-registered verdict remains
computed over the frozen 12 fp16 primary cells only and re-prints
identically. Follow-up A MUST run on the RTX 4090 Laptop GPU: the flag being
scoped belongs to that device's frozen dataset. Prediction stated before
measuring (mechanism probe): if fp32 prefill ratios collapse toward the BERT
fp16 level (~0.16), the fp16-saturation mechanism is supported and the flag
scopes to compute-bound fp16 cells. B (pretrained GPT-2 weights ported into
our own stack; the implementation-free isolation) is designed and built
after A's result. Lambda Cloud noted as the platform for the A100 phase
(Steps 6-7); instance type and GPU-hour budget still to be supplied for the
Step 5 amendment.

### Addendum 2026-08-10 - Follow-up A complete: both pre-stated predictions confirmed
fp32 prefill ratios 0.121-0.125 (from 0.243-0.330 at fp16); init-seed CV
5.60% -> 0.22% (25x collapse, to noise level). fp16-saturation mechanism
SUPPORTED; the dataset flag scopes to compute-bound fp16 cells (decode and
fp32 measured-immune; encoder fp16 at the ~0.13-0.16 implementation floor).
Pre-registered FAIL verdict unchanged and re-printed from frozen records.
Internal consistency: fp32/fp16 random-arm ratio 2.8-3.1 sits inside the
sweep's forward precision band. Follow-up B predictions pre-stated in
findings.md (fp16 ported-vs-random ~0.1-0.2; fp32 ~0.00-0.03; HF-vs-ours at
identical weights reproduces the ~0.12-0.16 floor). Next: build B
(weight-porting module + tests + harness cells), then the Step 5 A100/Lambda
amendment with the full evidence.

### Addendum 2026-08-10 - Follow-up B machinery built (gate before measurement)
Weight-porting core landed ahead of any B measurement: (1) _DecoderModel
gains use_bias/use_wpe variant flags (defaults False: the sweep stack, its
parameter set, and every key remain byte-identical; a unit test locks this);
(2) workloads/port_gpt2.py maps GPT2LMHeadModel tensors into our stack
(Conv1D transposes, fused-QKV order, biases, wpe, tied lm_head) with
activation alignment (HF gelu_new -> tanh-approx GELU; numerics only,
identical op count); (3) verify_port() asserts full-position fp32 logit
agreement vs HF and the loader refuses to hand over an unverified port;
(4) five CPU-only tests build a tiny in-memory GPT2Config model (no
download) and exercise the gate, including a corrupted-weight canary and a
decode-position/wpe flow check. New builder arms: weights="ported"
(logit-verified port) and weights="random_v" (bias+wpe random structure
control). Next: three B cells in the harness (ported fp16, ported fp32,
random_v fp16) with a labeled FOLLOW-UP B section evaluated against the
pre-stated predictions, then the run.

### Addendum 2026-08-10 - Follow-up B cells added; porter gate green (238 passed)
The exact logit-equivalence gate passed on the tiny in-memory model (all five
porter tests green; full suite 238). Three B cells added to the harness in a
labeled FOLLOW-UP B section evaluated directly against the pre-stated
predictions: ported fp16, ported fp32 (both logit-verified against real gpt2
at load, on-GPU, before any energy is measured), and random_v fp16 (bias+wpe
structure control). Comparisons computed: pure value effect (ported vs
random_v, predicted 0.10-0.20), structure delta (random_v vs plain random
s42, predicted ~0), implementation floor (HF vs ported at identical values,
predicted 0.12-0.16), and fp32 value effect (predicted 0.00-0.03). Primary
verdict and Follow-up A sections re-print from frozen records.

### Addendum 2026-08-10 - Follow-up B complete; Step 4 closed end to end
Ported cells logit-verified on-GPU against real gpt2 before measurement.
Scorecard vs pre-stated predictions: structure delta 0.0051 [~0] CONFIRMED;
fp32 value effect 0.002-0.007 [0.00-0.03] CONFIRMED; pure fp16 value effect
0.0648 [0.10-0.20] SMALLER than predicted, ported inside the random spread;
implementation floor fp16 0.4046 [0.12-0.16] FAR LARGER (fp32 floor 0.1357
in band). Revised reading: the pre-registered FAIL is ~entirely a
cross-implementation artifact (E_HF/E_ported = 1.405 at identical weights
reconstructs the 0.24-0.33 primary gap); implementation-free, random-init
represents trained weights within ~6.5% worst-case. Verdict FAIL retained
as registered; flag narrowed; full record findings.md 2026-08-10 (third
entry). Step 4 is CLOSED. Next: Step 5 A100/Lambda pre-registration
amendment (R1 primary estimator; random-init full grid citing B; needs
Lambda instance type and GPU-hour budget from Muntaser).

### 2026-08-10 - Step 5: A100/Lambda pre-registration amendment APPROVED
experiments/exp_002_size_sweep/a100_amendment.md approved with all
recommended options: D1 T1 refit band 25%; D2 T2 scale-calibrated transfer
band 30% (frozen 8-cell calibration subset, evaluated on the remaining 76
shared-grid points); D3 T3 7B extrapolation confirmatory at 30%; D4 minimal
84-point shared grid; D5 include the 4 spot cells and the 6-point eager
subset; D6 coefficient-plausibility report-only; D7 A100 baseline companion
included with secondary Wilcoxon + Holm on the T1 split. Frozen before any
A100 data exists: R1 relative-error NNLS as PRIMARY estimator (absolute NNLS
secondary; evidence chain findings.md 2026-07-24), M8 form fixed with no
re-selection on the new platform, random-init full grid citing Follow-up B
(~6.5% implementation-free worst case), 10-point 7B extension (LLaMA-7B and
Mistral-7B, fp16 only, 40 GB constraints recorded), T0 A-B pooled median
<= 5%, protocol identical to the 4090 as operated (100 Hz, B primary, 4 s
floor, regime repeats 10/5, per-point explicit seeds with no cross-platform
weight-identity claim). Committed and pushed with this entry before any
Step 6 work. Next: Step 6 = configs/exp_002_a100.yaml via the grid builder
(enc-dec cells enumerated mechanically), validator and provenance-gate
extensions to the a100 output path, then the Lambda smoke checklist
(amendment section 13). No A100 measurement before the grid file is
committed.

### 2026-08-11 - Step 6 complete: A100 grid frozen, driver dispatch, gate allowlist, launch script
configs/exp_002_a100.yaml is FROZEN (98 points, expected_points integrity
guard) and expands only through the new multi-pass expander
(sweep/grid_passes.py): passes are run through the same frozen expand_grid
the 4090 sweep used, deduplicated globally on the seed-less key, and every
point receives an explicit init seed derived as sha256(master|seedless_key)
mod 2^31 with master 42 from seed.json. tests/test_grid_passes.py locks the
exact enumeration (strata 48 decoder + 12 encoder + 12 enc-dec fp16 both
arms + 6 fp32 anchors + 6 eager + 10 extension + 4 spot cells).
Driver: build_points_from_config dispatches passes-configs to expand_passes;
classic single-pass configs take the unchanged expand_grid path (frozen-4090
behavior byte-identical, tested). Preflight gains allow_untracked_paths: an
UNTRACKED-only exemption for a run's own outputs (crash-resume before the
first harvest commit), mirroring the representativeness harness's gate note;
tracked modifications still refuse, so the between-chunk harvest-commit
ritual is unchanged; the environment snapshot records the RAW git state and
every exemption is logged as a warning. scripts/run_exp002_a100.py launches
the frozen grid into experiments/exp_002_size_sweep/a100/ with the allowlist
prefix and documents the chunk ritual and the exact validator invocation.
Decision recorded: scripts/validate_exp002.py is UNMODIFIED. It is already
path-parametrized (--data/--summary/--json/--txt) and takes its expected
total from sweep_summary.json, so it extends to the a100 path by invocation
alone. Its WARN-only attention bands (idle power, A-B by phase, CV) are 4090
observed priors; A100 bands will be set from observed smoke values in a
dated commit BEFORE the measured chunks, and idle-power warnings on the A100
until then are expected and benign. Full pytest suite green with the 30 new
CPU-only tests across test_grid_passes, test_preflight_allowlist, and
test_driver_multipass. Local Step 6 prep is CLOSED at this commit. Next:
provision Lambda, run the smoke checklist (a100_amendment.md section 13),
then chunked measurement via run_exp002_a100.py.

### 2026-08-11 - Venue strategy: UEMCON replaces MLSys/ISCA; journal consolidation endgame
Decision by Muntaser: the TOMLTransformers paper targets IEEE UEMCON,
replacing the MLSys/ISCA target recorded in this file's header (this entry
supersedes that line; the header is append-only history). The endgame is a
comprehensive journal consolidation of the ENTIRE TOML package (FLAIRS TOML,
containerized cloud, signals, vision, transformers), preference JAIR or
JMLR; an honest venue scope-fit scan is scheduled at Step 8, with
TOMPECS/TACO/TSUSC/TC recorded as natural-fit alternatives (JMLR carries
real scope risk for hardware energy measurement; JAIR is defensible under
the ML-evaluation framing of the vision paper). Conference-to-journal
extension is the sanctioned path and the extension threshold is comfortably
met by the consolidation itself plus the A100 material. Consequence for the
experimental plan: NONE before Step 8. The A100 phase runs in full exactly
as frozen (amendment T0-T3); cross-platform transfer is the consolidation's
central new material, and the UEMCON paper carries the full transformers
story including the A100 results.

### 2026-08-11 - A100 smoke complete (amendment section 13); point 1/98 measured
Platform: Lambda gpu_1x_a100_sxm4, A100-SXM4-40GB, driver 580.105.08, CUDA
13.0 (identical driver/CUDA to the containerized-cloud paper's instance),
ECC on, MIG off. Stack: clean pip-resolved venv, torch 2.13.0+cu130; an
initial system-site-packages venv was abandoned after a numpy-2 vs system
torch ABI collision (recorded; no measurement touched it). Full suite 270
passed on the A100. to_costs self-test OK, a100 -> hbm2e. Instruments A, B,
C all live. Smoke record (DistilGPT2 prefill s256 fp16): A-B 2.02 percent,
B-C 2.4e-14 (bit-identical), SM clocks flat 1095 MHz across repeats (boost
governor at light load, no lock, identical runner path as the 4090), idle
70.7 W warmed vs 42 W at cold boot. Resume verified (second smoke run
skipped 3/3); throwaway _smoke removed. 7B feasibility probes: LLaMA-7B
prefill s8192 13.6 GiB, Mistral-7B prefill s8192 14.7 GiB, LLaMA-7B decode
ctx4096 16.0 GiB, Mistral-7B decode ctx4096 15.5 GiB; zero OOM risk in the
frozen grid. Point 1/98 measured through run_exp002_a100.py (max-hours
0.001 single-point stop): DistilGPT2 prefill s128 fp16 seed1617754261, ok,
B/exec 82.14 J, B/unit 40.34 mJ per forward, inner_iters 2036, wall 4.66 s,
A-B 0.07 percent, B-C 2.13 percent, CV(B) 7.27 percent (above the 5 percent
flag; lightest cell in the grid on a cold GPU; WATCH across chunk 1), temps
31-32 C, idle 70.75 W. Validator on the a100 path: single WARN, idle
outside the 4090 band, expected and benign as pre-recorded. Timing note,
recorded openly: the A100 attention bands will be set from chunk-1
observations (n about 40 idle readings) in a dated commit, rather than from
the two smoke values as the Step 6 entry anticipated; the bands are
WARN-only and gate nothing.

### 2026-08-11 - A100 chunk 1 closed: 24/98 resume-complete, zero failures
Points 2-24 measured (DistilGPT2 and GPT-2 across all prefill/decode x
fp16/fp32 strata, flash), 24/98 resume-complete, zero failed/OOM/short-window,
repeats per protocol (5 forward, 10 decode). Instrument agreement on Ampere is
far tighter than the 4090: A-B forward median 0.62 percent (max 1.36), decode
median 0.67 percent (max 4.25), B-C median 0.000 percent (max 2.88); the
pre-registered T0 (pooled A-B median <= 5 percent) is on track with wide
margin, itself a cross-platform methods datum. Repeat noise: CV(B) forward
median 2.48 percent; decode median 5.58 percent with three WARN cells above
7.5 percent (max 9.89, GPT-2 decode fp16 s128); this reproduces the 4090's
known light-cell decode noise pattern, is why the 10-repeat protocol exists,
and the fit target remains the median across repeats. Clocks bimodal
1095/1410 MHz by load (boost governor, no lock, identical runner path); max
temp 58 C, no throttling. Idle n=24 in [70.51, 72.19] W, median 71.03
(cold-boot reference 42 W); all 24 idle WARNs are the recorded 4090-band
artifact. Physics checks: 4 seq-len series, 0 monotonicity violations;
per-unit contamination flags exactly per design. Pace median 497 s/point;
validator projects ~10.2 h for the remaining 74 points, so the 12 h chunk 2
should complete the grid (7B cells rebuild ~14 GiB weights per sizing attempt
and may run slower; a small mop-up chunk is the worst case). Band plan
refined: the idle attention band becomes a validator CLI option (--idle-band,
default preserving the 4090 values) in a dated commit prepared locally during
chunk 2 from these n=24 observations, proposed [40, 90] W to span cold-start
and warmed states; WARN-only, gates nothing.

### 2026-08-12 - A100 GRID COMPLETE: 98/98 resume-complete, zero failures across the campaign
Chunk 2 (12 h budget, 8.23 h used) measured points 25-98; the frozen grid is
fully resume-complete with zero failed, zero OOM, zero short-window records
across the whole campaign, matching the 4090's 296/296 record. Coverage
reconciles exactly with the frozen enumeration: decoder prefill 40 (24 shared
+ 6 eager + 6 extension + 4 spot), decoder decode 28 (24 + 4 extension),
encoder 12, enc-dec 18 (both arms + fp32 anchors), 7B extension 10; the spot
cells including HF-pretrained and ported measured ok on Ampere (logit port
gate passed on Lambda). Instrument agreement: A-B forward median 0.58 percent
(n=56, max 2.59), decode median 0.63 percent (n=42, max 9.20, single
GPT-2-medium decode cell), B-C median 0.000 percent (max 3.69, one WARN on
T5-small decode s1024/ctx128); the pre-registered T0 pooled A-B median is on
track to pass with roughly 8x margin. Repeat noise: CV(B) forward median 2.27
percent; decode median 3.52 percent with 7 WARN cells, worst 16.07 percent on
T5-small decode; the heavy tail concentrates in the light enc-dec decode
family (inner_iters 3-6), the same known-noisy pattern the 10-repeat protocol
covers; fit target remains the median across repeats; WATCH these cells in
fit residuals. Thermal max 61 C (BERT-large s2048; 7B prefills 60 C), no
throttling; clocks bimodal 1095/1410 MHz by load as in chunk 1. Idle n=98 in
[70.45, 72.57] W, median 71.25; all 98 idle WARNs remain the recorded
4090-band artifact. Physics: 16 seq-len series, 0 monotonicity violations;
contamination flags exactly per design. Known validator limitation found: the
fp32/fp16 matched-shape check reports 0 pairs on this dataset because it
pairs on FULL keys and per-point derived seeds differ across precision; a
data-independent matcher fix (pair on seed-stripped keys) will ride with the
dated validator update (--idle-band [40, 90] W for a100 invocations, 4090
default preserved; both WARN-only, gating nothing). Campaign totals:
measurement 8.23 + 2.64 h chunk elapsed plus smoke on one continuous
instance (~21 h wall), GPU-active well inside the 15-20 GPU-hour envelope.
Termination gate: dataset verified on origin AND the local workstation
(this entry is written locally from the pulled artifacts) before the
instance is terminated. Next: verify Lambda tree clean and terminate; dated
validator update; then Step 7, the pre-registered fit phase (T0-T3, R1
primary, frozen M8 form), which runs locally with no GPU required.

### 2026-08-17 - Dated validator update: --idle-band CLI and seed-agnostic fp32/fp16 pairing; A100 re-validated under [40, 90] W
Two WARN/report-only changes to scripts/validate_exp002.py, both gating
nothing; tests approved and added (tests/test_validate_cli.py, 8 tests, the
script's first test coverage; full suite 278 passed in 514 s). (1) --idle-band
LOW HIGH (two floats), default [1.0, 15.0] bound to the existing 4090
constants; the section 9 check and its WARN text use the parsed band; when a
non-default band is passed the report header prints one extra line naming the
override and the default (an addition beyond the two approved changes, flagged
in chat; default output unchanged). (2) fp32/fp16 matched-shape pairing is
seed-agnostic through a module-level pure function shape_pair_key(spec) with
PAIR_EXCLUDE = ("precision", "key", "seed"). Mechanism correction to the
2026-08-12 entry, which anticipated stripping a seed suffix from key strings:
the pairing code never touched key strings; it built the pair key from
spec.items() minus precision and key, and the A100 specs carry a separate
spec.seed field, derived from the precision-inclusive key, that differs
between fp16 and fp32 twins; excluding that field is the whole fix. The 4090
specs carry no seed field, so default pairing is provably identical (a test
asserts shape_pair_key equals the legacy tuple on a seedless spec). Argparse
construction factored to a module-level build_arg_parser() for testability,
behavior unchanged. Byte-identity check on the 4090 default path: the default
invocation regenerated into a temp path differs from the committed
validation_report.txt in exactly one line, the generated timestamp (git diff
--no-index: 1 insertion, 1 deletion). A100 re-validation (all four path args,
--idle-band 40 90, reports regenerated in place): warnings 106 -> 8, the 98
idle-band artifacts gone, the 8 remaining being the 7 CV(B) cells and the 1
B-C cell recorded on 2026-08-12; the precision-pairing check executed on the
A100 data for the first time: 34 matched-shape pairs, 0 inversions;
forward-phase fp32/fp16 per-unit energy ratio n=20, min 6.20, p25 6.68,
median 7.43, p75 8.76, max 10.20 (findings.md entry of this date).
Reconciliation of 34 pairs against 36 fp32 points, verified from the records:
pairs are prefill 12, decode 12, encode 8 (6 encoder-only + 2 enc-dec),
decoder_prefill 2; the two unpaired fp32 points are the enc-dec decode fp32
anchors at s1024/ctx1024 (T5-small, BART-base), whose fp16 twin cell is not in
the frozen grid (the fp16 enc-dec decode arms are s128/ctx1024, s2048/ctx1024,
s1024/ctx128, s1024/ctx2048); a frozen-grid property, not a data gap (98/98
complete). All 36 fp32 points lie in the 84 shared cells; extension, eager
and spot cells are all fp16. Open items carried into Step 7, recorded before
any fit computation: (a) verify the amendment's exact definition of the T2
calibration cell "T5-small decode anchor-1024" (fp16) against the actual fp16
cells above, since no fp16 s1024/ctx1024 decode cell exists; (b) check
whether the runner leaves PyTorch's default TF32 setting (matmul TF32 off)
before any text attributes the A100's larger fp32 penalty to fp32 GEMMs
running on CUDA cores; (c) T2 residual prediction: T2 fits one scalar on
fp16-only calibration cells against the frozen 4090 vector, and the A100's
forward fp32/fp16 ratio (median 7.43) is about 2.3x the 4090's (3.24), so to
first order the 36 fp32 evaluation cells are expected to be systematically
under-predicted under T2 and to carry the bulk of its residual; the T2
verdict is at risk on this mechanism, and if it fails, the fp32-cell residual
pattern is the pre-stated diagnosis, not a post-hoc one. Threshold, mechanism
and calibration set are unchanged; whether T1's refit can express a
platform-specific precision ratio depends on M8's feature structure (check
when building the fit path); precision-structured residuals are to be
reported descriptively; no model re-selection on the A100 (settled). Commits:
feat(validator) for the script and tests, chore(exp-002) for the regenerated
a100 reports, docs(lab) for this entry and findings.md. Next: Step 7, read
a100_amendment.md in full, then propose the fit-path shape.

### 2026-08-17 - Pre-fit record: precision-prior structural mismatch on the A100, T1/T2/T3 risk predictions, T2 calibration-cell root cause; no fit computed yet
Written BEFORE any A100 fit of any kind. What has been examined so far on the
A100 data: the validator reports (this date) and, ad hoc, the fp32/fp16
matched-pair per-unit(B) ratios by phase computed from the frozen records
with the validator's own matcher; the forward-phase numbers agree with the
committed reports. Nothing else.
(1) T2 calibration cell root cause, verified against configs/exp_002_a100.yaml,
configs/exp_002.yaml, sweep/grid.py and both datasets: the 4090 grid carries
the enc-dec decode center cell (s1024, ctx1024) in BOTH precisions (T5-small
decode: source arm s in {128,512,1024,2048,4096} at ctx1024, target arm ctx in
the same set at s1024, deduped at the center; 143 fp32 points, 143 pairs, 0
unpaired), which is the mental model behind amendment section 9's "T5-small
decode anchor-1024" (fp16). Amendment section 5c budgeted enc-dec at 12 fp16
+ 6 fp32 = 18, and 12 fp16 for two models is reachable only by omitting the
fp16 center decode cell (encode + decoder_prefill + 4 off-center decode cells
per model). The Step 6 grid encoding followed 5c exactly and states it in a
comment ("The 1024/1024 decode cell is deliberately absent here (it is the
fp32 anchor below)"). So the discrepancy is internal to the amendment
(section 5c counts vs section 9 cell naming), faithfully implemented; the 98
points are exactly the frozen enumeration; not a builder or data fault. Not
caught earlier because the enumeration-lock tests check counts and tuples
against the config, not section 9's prose; the 4090 fits never named cells;
the pairing check was broken for the whole campaign, hiding the two unpaired
fp32 anchors; and coverage reconciliation was config-vs-data. Process fix: a
spec-to-enumeration test that resolves every named cell to exactly one key.
No further A100 measurement is needed for any pre-registered test; the cell
is resolved by dated clarification (amendment section 16, this date) with a
descriptive sensitivity companion over the four candidates.
(2) The larger finding. Precision enters the frozen model only through fixed
priors: to_costs PRECISION_MAC_MULT fp16 = 0.33 (provenance NEW_ESTIMATE,
Horowitz bit-width scaling) vs fp32 = 1.00, and words per element fp16 = 0.5
vs fp32 = 1.0; to_nonlinear, n_launches and the intercept are
precision-independent, and M8_split_dispatch shares one coefficient set
across precisions. Hence the model's implied fp32/fp16 energy ratio at
matched shape lies in [1.0, 3.03] for every beta >= 0. Measured matched-pair
ratios (per-unit B): 4090 prefill n=25 median 3.39 (2.04-3.90), encode n=37
median 3.17 (2.12-4.11), decoder_prefill n=20 median 3.40 (2.96-3.92), decode
n=61 median 1.69 (1.23-2.65), all inside or at the edge of the expressible
range, which is why the prior fit there. A100 prefill n=12 median 7.43
(6.20-9.07), encode n=8 median 7.46 (6.59-10.20), decoder_prefill n=2 (6.34,
10.11), decode n=12 median 3.51 (1.54-5.52, rising with context because the
measured decode composite includes the prefill of the context). A100 forward
ratios sit about 2x above the model's ceiling; no refit of the frozen form
can express them. Mechanism (hypothesis, well supported): Ampere runs fp16 on
tensor cores and true fp32 GEMMs on CUDA cores (vendor peaks 312 vs 19.5
TFLOP/s dense, to be cited in fit/baselines.py before D7), a datapath gap the
Ada consumer part does not have to the same degree; the 0.33 prior is
arithmetic switching cost and misses operand-delivery energy on scalar FMA
units. TF32 status checked: measure/runner.py, sweep/point.py and
workloads/decoder.py set no allow_tf32 or float32_matmul_precision, so the
PyTorch default applied (matmul TF32 off in every version known here; the
torch 2.13 default on the Lambda venv is to be confirmed by release notes or a
one-liner before the paper asserts it). Framing caution: the 4090 config's
Fork 2 comment reads "to fit the precision MAC multiplier ... instead of
asserting them", but no M-family member has a precision-split column and
fit_plan.md treats the multiplier as descriptive discussion; the multiplier is
asserted, not fitted, and the paper must say so.
(3) Predictions registered now, thresholds and mechanisms unchanged. T2:
single scalar on fp16-only calibration cells against the frozen 4090 vector;
the 36 fp32 evaluation cells under-predicted by roughly half on forward
phases; verdict at high risk. T1: the R1 compromise on each forward pair
lands near fp16 +20 percent, fp32 -50 percent (relative loss favors the fp16
side); pooled held-out MAPE plausibly 25-35 percent against the 25 percent
band; risk material. T3: the all-84 fit's fp16 predictions inherit the
forward over-prediction before any extrapolation error, and 6 of the 10
targets are prefill; risk moderate. D7: the roofline baseline uses
per-precision peaks (4090: 39.69 / 79.38 TFLOP/s; A100: 19.5 / 312), so it
encodes the datapath structure M8 lacks and may be competitive on A100 fp32
cells; reported as it falls. If any test fails, the precision-structured
residual is the pre-stated diagnosis, not a post-hoc one.
(4) Decisions. Confirmatory tests run exactly as frozen (M8 form, R1
primary, T0-T3 bands, 84/76/10 sets, spot cells excluded); no
re-specification in response to these observations. Pre-registered as
DESCRIPTIVE (section 16): fp16-vs-fp32 breakdown of the T1 and T2 residuals,
separating "does the memory-tier prior transfer" (D6, alpha_hbm ratio, read
on fp16 cells) from "does the precision prior transfer"; the T2 sensitivity
companion. Named as POST-VERDICT EXPLORATORY work with no verdict authority,
to be run only after T0-T3 are recorded in findings.md: (a) a precision-split
MAC model (to_mac split into fp16 and fp32 columns, all else M8) under R1 on
the same T1 split, reported as the 4090 R1 exploratory was; (b) optional
Follow-up C, about 1 GPU-hour on a fresh A100 instance, re-measuring a few
matched fp32 cells with TF32 enabled, with the prediction registered here
that the fp32/fp16 ratio collapses from about 7.4x toward 2-3x because the
GEMMs move to tensor cores. Both feed EXP-003 (quantization), where precision
multipliers are the object of study. Next: amendment section 16; calibration
key lock test; full read of scripts/fit_exp002.py; scripts/fit_exp002_a100.py.

### 2026-08-17 - Step 7 confirmatory phase computed and closed: T0 PASS, T1/T2/T3 FAIL as they fell; two mechanisms identified
Code first, then results, in that order. Committed 176183d
(feat(fit)): scripts/fit_exp002_a100.py (imports the frozen R1 and absolute
estimators, target and units gate by path from scripts/fit_exp002.py,
which is unchanged; form M8 frozen; sets resolved mechanically: shared 84 /
extension 10 / spot 4 by pass name via the new fit/a100_strata.py, T2
calibration cells via fit/a100_calibration.py); fit/baselines.py gained the
A100 roofline constants, vendor-cited (NVIDIA A100 datasheet June 2021;
Ampere whitepaper): FP32 19.49 TFLOP/s by 2 x 6,912 x 1.410 GHz, FP16 311.9
TFLOP/s dense by 108 x 1,024 FMA/clk x 2 x 1.410 GHz, 1,555 GB/s, 400 W;
4090 defaults untouched; 12 new tests (test_fit_baselines_a100.py,
test_a100_strata.py) green, suite 300 expected. Then the fit ran once (CPU,
seconds) and wrote a100/fit/{fit_report.txt, fit_results.json,
per_point_predictions.jsonl}. Verdicts as they fell (findings.md, this
date, full numbers): T0 PASS 0.60 percent; T1 FAIL 35.60 percent (band 25,
17 held-out points); T2 FAIL 52.93 percent (band 30; s* 0.386; sensitivity
companion 51.9-54.2 percent, all FAIL, cell choice immaterial); T3 FAIL
42.66 percent (band 30; decode 5.8 percent, prefill +66 to +71 percent).
Prediction scorecard against this morning's entry: T1 CONFIRMED (fp16
+19.9 percent, fp32 -40.0 percent, 35.6 percent at the top of the 25-35
range); T2 fp32-dominant CONFIRMED, fp16 subset also failing NOT predicted;
T3 direction CONFIRMED, magnitude EXCEEDED; roofline competitive
CONFIRMED (best MAPE under both estimators, P_avg 369-413 W against the 400
W envelope, no comparison significant at n=17); precision-ratio ceiling
CONFIRMED (model-implied 2.96 vs measured 7.52 on forward pairs).
Post-hoc descriptive analysis (labeled): joining per-point residuals with
the recorded median SM clocks shows a second mechanism beside the
precision datapath. fp16 flash forward cells at 1410 MHz: +6 percent mean,
0 to +16; cells at the low DVFS state 1095 MHz (light s128 cells): +44 to
+73 percent; cells power-capped at 1250-1330 MHz drawing about 320-330 W
above the 71 W idle, i.e. at the 400 W envelope (GPT-2-XL fp16 prefill and
every 7B prefill): +40 to +71 percent; fp32 cells at 1410 MHz at 260-300 W.
Energy per TO on the A100 depends on the operating point by about a factor
of two across the grid, which the frozen form cannot express; T3's prefill
error is entirely this, while memory-bound 7B decode at 1410 MHz transfers
within 6 percent. Third, minor: eager cells under-predicted 71 percent
(math SDPA backend on Ampere); to_sram and n_launches fit to zero. Pipeline
checks passed (units gate on all 98; 7B decode composites reconcile; T0).
Secondaries recorded: D6 alpha_hbm ratio 0.489, effective per-word 695 vs
1,778 pJ/word (ratio 0.39 vs expectation about 1.0; caveat, alpha_hbm
absorbs the vanished sram and launch terms); MCER phase transition
reproduces on Ampere with smaller magnitudes; spot cells replicate the
Follow-up B decomposition (E_HF/E_ported 1.12, value effect -3.5 percent).
Decisions: verdicts recorded as they fell; no re-selection; the paper
reports T1-T3 as FAIL with the two mechanisms, and reports what transfers
(T0, decode regime, MCER transition). Next, in order: commit the artifacts
and these entries; full suite before push; then the post-verdict exploratory
work of amendment section 16.5 under its own label: (a) precision-split MAC
model on the T1 split and the T2 repeat, plus a descriptive
residual-vs-operating-point table; (b) Follow-up C remains optional. Then
Step 8 (figures, UEMCON manuscript, journal venue scan). Fidelity note for
the paper: the A100-SXM4-40GB carries HBM2 per the datasheet; the registry
tier is labeled hbm2e (6.00 pJ/bit); an HBM2 6.25 pJ/bit prior rescales
to_hbm by 200/192 uniformly, absorbed by alpha_hbm, verdict-invariant.

### 2026-08-18 - Exploratory phase (amendment 16.5(a)) computed; T3 attribution corrected; Step 7 closed
scripts/explore_exp002_a100.py committed at 4e774c7, then run once (CPU);
artifacts in a100/fit/exploratory/{explore_report.txt, explore_results.json}.
One bug found and fixed before the successful run: PRECISION_MAC_MULT
entries are TOCost records, not floats (fixed to .value; commit amended).
EXPLORATORY throughout, no verdict authority; the confirmatory artifacts and
the T0-T3 verdicts of 2026-08-17 are untouched. Numbers in findings.md, this
date. Headlines: M8p (to_mac split by precision, all else M8) gives held-out
20.02% on the identical T1 split (M8 35.60%) and 11.04% on the 10 7B points
(M8 42.66%); the fitted fp16 MAC multiplier is 0.083 on the A100 and 0.130
on the 4090 against the asserted 0.33, so the precision prior is a device
property and belongs in the device registry beside the memory tier. T2 still
fails under M8p (52.42%) because three device quantities differ, not one:
the fp16 multiplier, the dispatch coefficient (n_launches 8.6e-4 on the 4090
vs 8.2e-6 on the A100, about 100x; hypothesis, Windows WDDM launch overhead
on the laptop host vs Linux on the A100 host, consistent with the signals
paper's host-dependent alpha_o), and the operating-point behavior; to_hbm
ratio 0.50. HONEST CORRECTION recorded in findings.md: the 2026-08-17 entry
attributed T3's +67% prefill error entirely to the operating point; it was
overwhelmingly the alpha_mac compromise (M8 3.118e-15 vs M8p fp16 1.766e-15,
ratio 1.77 = the +67%). The operating-point effect is real but secondary
(about a 35-point spread after the split, not a factor of two); the table in
section (b) of the report stands as the descriptive record. Two residual
structures the split does not remove: fp32 decode-like still -28% mean, and
the six eager cells still about -77% (math SDPA path on Ampere).
Step 7 is now CLOSED: confirmatory verdicts recorded as they fell,
secondaries recorded, exploratory work done and labeled. Next: Step 8
(figures, UEMCON manuscript, journal venue-fit scan). Nothing in the
exploratory phase may be presented as a fitted result of this experiment; if
the precision-split model is to carry weight, it needs its own dated
pre-registration on new data (EXP-003 or the next platform).
