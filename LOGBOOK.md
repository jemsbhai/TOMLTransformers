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
