"""Single-point measurement: build one workload, measure it under control, and
flatten the result into one JSONL-ready record.

This is the unit the grid driver (next step) calls in a loop. It is deliberately
the only place that knows how to (a) turn a point spec into the right workload via
the three builders, (b) size the window with measure_until_floor, and (c) compute
the normalized per-unit energy with the correct divisor per phase.

Energy reported is DYNAMIC energy above idle (the runner's convention). For every
point we store BOTH:
  - per-execution energy: what one looped run() consumed (the raw measurement);
  - per-unit energy: normalized to one forward (prefill/encode/decoder_prefill) or
    one decode token (decode), so points with different inner_iters are comparable.

DECODE CAVEAT (recorded honestly): a decode growing-mode run() loops
inner_iters x (one prefill + decode_tokens steps), so the measured window MIXES
prefill and decode-step energy. The naive per-token here (exec / inner_iters /
decode_tokens) is therefore PREFILL-CONTAMINATED and is flagged as such in the
record (per_unit_contaminated = True). The clean, prefill-subtracted per-token is
derived at fit time from this point and the matching prefill point, where the
uncertainty can be propagated. Do not cite the naive per-token as the decode cost.

Instrument B (hardware energy counter) is primary; A and C are recorded alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .provenance import _run as _git_run   # reuse the same git primitive
from ..measure import runner as rn
from .. import workloads as wl


# --- per-phase unit accounting ------------------------------------------------

# "forward" phases: one run() loop == inner_iters forwards; unit = one forward.
_FORWARD_PHASES = {"prefill", "encode", "decoder_prefill"}
# decode: one run() loop == decode_tokens decode steps (growing) or 1 step
# (fixed_step), but the loop ALSO contains a prefill (contamination, see header).


def _git_commit() -> Optional[str]:
    return _git_run(["git", "rev-parse", "HEAD"])


@dataclass
class PointSpec:
    """Identifies a single measurement point. Dispatches to a workload builder."""

    model: str
    arch: str                       # decoder_only | encoder_only | encoder_decoder
    phase: str                      # prefill|decode | encode | encode|decoder_prefill|decode
    precision: str = "fp16"
    attn_kind: str = "flash"
    weights: str = "random"
    # size parameters (phase-dependent; unused ones ignored)
    seq_len: Optional[int] = None   # src_len for enc-dec; sequence for dec/enc
    tgt_len: int = 1                # decoder_prefill target prompt length
    tgt_ctx: int = 128             # decode established target context
    decode_tokens: int = 64        # K, decode growing
    decode_mode: str = "growing"   # growing | fixed_step
    batch_size: int = 1
    device_index: int = 0
    pretrained_id: Optional[str] = None

    def key(self) -> str:
        """Stable identity string for resumability / dedup (grid driver uses it)."""
        parts = [self.arch, self.model, self.phase, self.precision, self.attn_kind,
                 self.weights, f"s{self.seq_len}", f"b{self.batch_size}"]
        if self.phase == "decoder_prefill":
            parts.append(f"tgt{self.tgt_len}")
        if self.phase == "decode":
            parts += [f"ctx{self.tgt_ctx}", f"k{self.decode_tokens}", self.decode_mode]
        return "|".join(parts)


def _build_workload(ps: PointSpec, inner_iters: int):
    """Dispatch a PointSpec to the correct workload builder."""
    if ps.arch == "decoder_only":
        if ps.phase not in ("prefill", "decode"):
            raise ValueError(f"decoder_only phase must be prefill|decode, got {ps.phase}")
        return wl.build_decoder_workload(
            ps.model, phase=ps.phase, seq_len=ps.seq_len, precision=ps.precision,
            weights=ps.weights, attn_kind=ps.attn_kind, inner_iters=inner_iters,
            batch_size=ps.batch_size, device_index=ps.device_index,
            decode_tokens=ps.decode_tokens, decode_mode=ps.decode_mode,
            pretrained_id=ps.pretrained_id,
        )
    if ps.arch == "encoder_only":
        if ps.phase != "encode":
            raise ValueError(f"encoder_only phase must be encode, got {ps.phase}")
        return wl.build_encoder_workload(
            ps.model, seq_len=ps.seq_len, precision=ps.precision, weights=ps.weights,
            attn_kind=ps.attn_kind, inner_iters=inner_iters, batch_size=ps.batch_size,
            device_index=ps.device_index, pretrained_id=ps.pretrained_id,
        )
    if ps.arch == "encoder_decoder":
        if ps.phase not in ("encode", "decoder_prefill", "decode"):
            raise ValueError(
                f"encoder_decoder phase must be encode|decoder_prefill|decode, got {ps.phase}")
        return wl.build_enc_dec_workload(
            ps.model, phase=ps.phase, src_len=ps.seq_len, tgt_len=ps.tgt_len,
            tgt_ctx=ps.tgt_ctx, decode_tokens=ps.decode_tokens, decode_mode=ps.decode_mode,
            precision=ps.precision, weights=ps.weights, attn_kind=ps.attn_kind,
            inner_iters=inner_iters, batch_size=ps.batch_size,
            device_index=ps.device_index, pretrained_id=ps.pretrained_id,
        )
    raise ValueError(f"unknown arch '{ps.arch}'")


def _per_unit_energy(ps: PointSpec, per_exec: dict, inner_iters: int):
    """Normalize per-execution energy to per-unit. Returns (per_unit, unit, contaminated).

    forward phases: unit = one forward; divisor = inner_iters.
    decode: unit = one decode token (growing) or one step (fixed_step); divisor =
      inner_iters * decode_tokens (growing) or inner_iters (fixed_step). FLAGGED
      contaminated because the loop also runs a prefill (see module header).
    """
    if ps.phase in _FORWARD_PHASES:
        div = max(inner_iters, 1)
        unit = "forward"
        contaminated = False
    elif ps.phase == "decode":
        if ps.decode_mode == "growing":
            div = max(inner_iters * max(ps.decode_tokens, 1), 1)
            unit = "decode_token"
        else:  # fixed_step
            div = max(inner_iters, 1)
            unit = "decode_step"
        contaminated = True   # in-loop prefill mixed into the window
    else:
        div = max(inner_iters, 1)
        unit = "execution"
        contaminated = False
    per_unit = {k: (v / div) for k, v in per_exec.items()}
    return per_unit, unit, contaminated


@dataclass
class MeasuredPoint:
    spec: dict
    record: dict
    ok: bool


def measure_single_point(
    ps: PointSpec,
    *,
    target_s: float = 4.0,
    repeats: int = 5,
    warmup_iters: int = 50,
    idle_baseline_s: float = 3.0,
    min_window_s: float = 4.0,
    sampling_hz: float = 100.0,
    thermal_settle: bool = True,
    cv_threshold: float = 0.05,
) -> dict:
    """Measure one point end-to-end and return one JSONL-ready record (a dict).

    Sizes the window with measure_until_floor (rescaling inner_iters from the real
    measured wall-time), measures under the controlled runner, and flattens the
    PointResult + spec + per-unit normalization + provenance into a flat record.

    Never raises on OOM or a failed workload: records ok=False with the reason, so
    the grid driver can log and continue.
    """
    spec_dict = {
        "model": ps.model, "arch": ps.arch, "phase": ps.phase,
        "precision": ps.precision, "attn_kind": ps.attn_kind, "weights": ps.weights,
        "seq_len": ps.seq_len, "tgt_len": ps.tgt_len, "tgt_ctx": ps.tgt_ctx,
        "decode_tokens": ps.decode_tokens, "decode_mode": ps.decode_mode,
        "batch_size": ps.batch_size, "key": ps.key(),
    }

    def builder(inner):
        return _build_workload(ps, inner)

    def measure_fn(work):
        return rn.measure_point(
            work.run, work.spec.label(), repeats=repeats, warmup_iters=warmup_iters,
            idle_baseline_s=idle_baseline_s, min_window_s=min_window_s,
            sampling_hz=sampling_hz, thermal_settle=thermal_settle,
            cv_threshold=cv_threshold, device_index=ps.device_index,
        )

    base = {
        "schema": "tomltransformers.sweep.point.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "spec": spec_dict,
    }

    try:
        res, inner = wl.measure_until_floor(builder, measure_fn, target_s=target_s)
    except BaseException as exc:  # noqa: BLE001  (record, do not crash the sweep)
        base.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                     "oom_skipped": "out of memory" in str(exc).lower()})
        return base

    # OOM (or other skip) surfaced through the runner, not as an exception.
    if res.skip_reason:
        base.update({"ok": False, "error": None, "skip_reason": res.skip_reason,
                     "oom_skipped": res.skip_reason == "OOM",
                     "inner_iters": inner, "notes": res.notes})
        return base

    per_exec = {k: s["mean"] for k, s in res.summary.items() if k in ("A", "B", "C")}
    per_unit, unit, contaminated = _per_unit_energy(ps, per_exec, inner)
    ag = rn.pairwise_agreement(res)

    base.update({
        "ok": bool(res.ok),
        "oom_skipped": False,
        "inner_iters": inner,
        "primary": "B",
        "n_repeats": res.n_repeats,
        "instruments_available": res.instruments_available,
        # energy: per-execution (raw measurement) and normalized per-unit
        "per_execution_j": per_exec,
        "per_execution_std_j": {k: res.summary[k]["std"] for k in per_exec},
        "per_execution_cv": {k: res.summary[k]["cv"] for k in per_exec},
        "per_execution_median_j": {k: res.summary[k]["median"] for k in per_exec},
        "per_unit_j": per_unit,
        "per_unit_kind": unit,
        "per_unit_contaminated": contaminated,
        # measurement quality / telemetry
        "agreement": ag,
        "cv_exceeded": res.cv_exceeded,
        "short_window": res.short_window,
        "wall_time_s_median": res.summary.get("wall_time_s", {}).get("median"),
        "idle_power_w": res.idle_power_w,
        "temps_c": res.temps_c,
        "clocks_mhz": res.clocks_mhz,
        "thermal": res.thermal,
        "notes": res.notes,
    })
    return base
