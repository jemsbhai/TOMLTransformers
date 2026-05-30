"""Workload interface: the fixed contract every builder satisfies.

The runner and sweep depend only on this interface, not on how any particular
model is realized on the GPU. That decoupling is what lets each workload decide
its own realization (our torch.nn blocks, a transformers model, hybrid) without
changing anything downstream.

A Workload bundles:
  - run(): the zero-argument callable measure_point expects. ONE call performs
    one measured execution (which may loop the forward pass inner_iters times to
    clear the window-length floor).
  - spec: a WorkloadSpec describing what is being run (for the results record).
  - free(): release GPU memory (drop the model, empty the cache). Called by the
    sweep between points so VRAM does not accumulate across the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class WorkloadSpec:
    """Description of a single workload point (recorded in results)."""

    model_name: str
    arch: str                     # decoder_only | encoder_only | encoder_decoder
    phase: str                    # prefill | decode | attention_compare
    seq_len: int                  # prompt length (prefill) or context length (decode)
    precision: str                # fp16 | fp32
    weights: str                  # random | pretrained
    attn_kind: str = "flash"      # flash | eager (SDPA backend)
    inner_iters: int = 1          # forward passes per measured execution
    batch_size: int = 1
    extra: Dict[str, object] = field(default_factory=dict)

    def as_record(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        w = "rand" if self.weights == "random" else "pt"
        base = (f"{self.model_name}|{self.phase}|s{self.seq_len}|{self.precision}"
                f"|{w}|{self.attn_kind}")
        if self.inner_iters != 1:
            base += f"|x{self.inner_iters}"
        return base


@runtime_checkable
class Workload(Protocol):
    """What the runner/sweep rely on. Any builder returns an object like this."""

    spec: WorkloadSpec

    def run(self) -> object:
        """Perform ONE measured execution (may loop the forward inner_iters times)."""
        ...

    def free(self) -> None:
        """Release GPU memory held by this workload."""
        ...


@dataclass
class CallableWorkload:
    """Concrete Workload backing for builders: a run fn + an optional free fn."""

    spec: WorkloadSpec
    _run: Callable[[], object]
    _free: Optional[Callable[[], None]] = None

    def run(self) -> object:
        return self._run()

    def free(self) -> None:
        if self._free is not None:
            self._free()
