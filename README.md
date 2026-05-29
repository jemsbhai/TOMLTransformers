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
INT4), and Mixture-of-Experts.

## Status

Framework build in progress. The energy model is fit and selected from a nested
family of formulations (calibrated-FLOPs baseline through compute + memory +
overhead + kernel-dispatch terms) against measured GPU energy, with the best
form chosen by held-out error and information criteria.

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
2. Container energy attribution (TED / SPP / CTI), under review.
3. Signals: a four-parameter model across 37 DSP algorithms, under review.
4. "Beyond FLOPs": a position paper, under review.
5. **This repository:** the transformer extension.

Next planned: TOMLtraining (training energy).

## License

MIT (see `LICENSE`).
