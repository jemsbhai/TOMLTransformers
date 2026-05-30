"""Runnable GPU workloads for EXP-002 measurement.

The measurement runner (tomltransformers.measure.runner.measure_point) consumes a
zero-argument callable that performs ONE execution of a workload. This subpackage
builds those callables from a TransformerConfig.

Key correspondence: the runnable model's layer structure must match what the
TO-counting front-end (architectures/decoder.py etc.) COUNTS, because measured
energy from these workloads is later compared against those TO predictions. The
decoder builder here mirrors architectures/decoder.py exactly: pre-norm, two
norms per layer, QKV -> attention core (causal) -> output projection -> FFN
(gated SwiGLU or standard), final norm, lm_head.

Two realization paths, selected per workload (default random-init, no downloads):
  - random-init: a shape-faithful torch module built from the config. Energy
    depends on op shapes and data movement, not weight values, so random init is
    valid for the sweep (this is the Fork-1 premise, itself tested by the
    representativeness check).
  - pretrained: load real weights (e.g. via transformers) for spot checks.

Window-length floor: the runner flags windows under ~2 s as unreliable (even the
hardware energy counter is self-inconsistent there). Builders therefore expose an
`inner_iters` knob; a workload's single execution runs the forward pass
`inner_iters` times so small/fast models clear the floor. `calibrate_inner_iters`
picks a count that puts one execution near a target duration.
"""

from __future__ import annotations

from .protocol import Workload, WorkloadSpec
from .decoder import build_decoder_workload, calibrate_inner_iters, measure_until_floor
from .encoder import build_encoder_workload
from .encoder_decoder import build_enc_dec_workload

__all__ = [
    "Workload",
    "WorkloadSpec",
    "build_decoder_workload",
    "build_encoder_workload",
    "build_enc_dec_workload",
    "calibrate_inner_iters",
    "measure_until_floor",
]
