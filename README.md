# TOMLTransformers

Transistor-level energy modeling of **transformer inference**, extending the
TOML (Transistor Operations for Machine Learning) framework from CNNs, RNNs,
and gradient-boosted trees to transformers.

Conventional efficiency metrics (FLOPs, MACs) treat all operations as
energetically equal and ignore data movement, which dominates real energy.
TOML grounds energy estimation in CMOS switching physics. This work carries
that into the transformer setting, where the prefill and decode phases have
identical operation types but very different energy profiles that FLOPs cannot
distinguish.

## Scope

Inference energy only (training energy is a separate planned work). The
framework models three architecture classes, because they differ structurally:

- **Decoder-only** (GPT / LLaMA / Mistral): causal, two phases (prefill and
  decode), one growing KV cache. The memory-compute energy ratio (MCER) phase
  transition between prefill and decode is the headline result.
- **Encoder-only** (BERT / ViT): bidirectional, single forward pass, no KV
  cache, full sequence-by-sequence softmax.
- **Encoder-decoder** (T5 / BART): an encoder pass plus a decoder with causal
  self-attention (its own KV cache) and cross-attention (keys/values projected
  once from the encoder output and cached).

It also covers standard vs FlashAttention accounting, precision (FP32 down to
INT4), and Mixture-of-Experts in the TO accounting; the published measurements
cover fp16 and fp32 on dense models.

## Paper

Muntaser Syed and Marius Silaghi, "Transistor-Level Energy Modeling of
Transformer Inference: Prefill/Decode Phase Separation and Cross-Architecture
Transfer," IEEE Annual Ubiquitous Computing, Electronics & Mobile Communication
Conference (UEMCON), 2026. Accepted; camera-ready source in `paper/`.

```bibtex
@inproceedings{syed2026tomltransformers,
  author    = {Muntaser Syed and Marius Silaghi},
  title     = {Transistor-Level Energy Modeling of Transformer Inference: Prefill/Decode Phase Separation and Cross-Architecture Transfer},
  booktitle = {IEEE Annual Ubiquitous Computing, Electronics \& Mobile Communication Conference (UEMCON)},
  year      = {2026}
}
```

## Status

Both measurement campaigns are complete and frozen. The energy model is fit
and selected from a nested family of ten formulations (calibrated-FLOPs
baseline M0 through compute + memory + dispatch terms) against measured GPU
energy, with the form chosen by AIC on a stratified split and then frozen.

- **RTX 4090 Laptop GPU** (Ada Lovelace, Windows 11, CUDA 12.4): 296
  configurations over 14 models at fp16 and fp32, covering prefill, encode and
  decode plus a standard-vs-fused attention sub-sweep. Selected form M8;
  held-out R^2 0.987 under absolute NNLS, 18.2% MAPE under the relative
  estimator R1.
- **A100-SXM4-40GB** (Ampere, Linux, CUDA 13.0): 98 configurations under a
  pre-registered amendment with acceptance bands frozen before any A100
  measurement. Outcomes as they fell: T0 PASS (0.60% vs 5%), T1 FAIL (35.6% vs
  25%), T2 FAIL (52.9% vs 30%), T3 FAIL (42.7% vs 30%). Within T3, 7B-class
  decode energy is predicted within 5.8%; the exceedances localize to the fp16
  arithmetic multiplier (a device property set by execution-unit routing) and
  the dispatch coefficient.

## Released artifacts

Every number in the paper traces to a file in this repository.

| Artifact | Location |
|---|---|
| RTX 4090 fit plan (pre-registration, committed 2026-07-20) | `experiments/exp_002_size_sweep/fit_plan.md` |
| A100 transfer amendment (committed 2026-08-10) | `experiments/exp_002_size_sweep/a100_amendment.md` |
| Frozen configurations | `experiments/exp_002_size_sweep/frozen_exp_002.yaml`, `experiments/exp_002_size_sweep/a100/frozen_exp_002_a100.yaml` |
| RTX 4090 measurements, 296 points | `experiments/exp_002_size_sweep/energy.jsonl` |
| A100 measurements, 98 points | `experiments/exp_002_size_sweep/a100/energy.jsonl` |
| Fit artifacts (results, per-point predictions, report) | `experiments/exp_002_size_sweep/fit/`, `experiments/exp_002_size_sweep/a100/fit/` |
| Representativeness study | `experiments/exp_002_size_sweep/representativeness.jsonl`, `representativeness_report.json`, `representativeness_report.txt` |
| Validation reports, environment snapshots, seeds | `validation_report.*`, `environment.json`, `seed.json` in each experiment directory |
| Figure pipeline with lineage gate | `scripts/make_figures.py`, `paper/figures/figures_manifest.json` |
| Experimental logbook and findings | `LOGBOOK.md`, `findings.md` |

Each measurement record carries the per-execution energy from all three
instruments (A: NVML power integration at 100 Hz; B: the on-die energy
counter, primary; C: Zeus), replicate count, coefficient of variation,
pairwise instrument agreement, idle power, per-replicate temperatures and
clocks, thermal-settling state, and the git commit the point was measured at.
Raw 100 Hz power traces are not tracked (see `.gitignore`).

## Setup

```powershell
pip install -e ".[dev]"          # core + test dependencies
pip install -e ".[measure]"      # add this to run the GPU measurement harness
```

## Reproduction

Every reported number traces to a logged experiment (see `LOGBOOK.md` and
`findings.md`). Configs are frozen per run, seeds and environment are snapshot,
and figures regenerate from scripts. Measurement follows the established TOML
protocol: thermal settling, per-run idle-baseline subtraction, repeated runs
with reported coefficient of variation.

## TOML paper series

1. FLAIRS-39: the canonical TOML framework (published).
2. Container energy attribution (TED / SPP / CTI): IEEE CloudCom 2026, Paris (accepted).
3. Signals: a four-parameter model across 37 DSP algorithms: IEEE MLSP 2026 (presented).
4. "Beyond FLOPs": a position paper: ACM AI Summit '26 (presented).
5. **This repository:** the transformer extension: IEEE UEMCON 2026 (accepted).

Next planned: TOMLtraining (training energy).

## License

MIT (see `LICENSE`).
