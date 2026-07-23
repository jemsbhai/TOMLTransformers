"""Feature bridge: frozen sweep records -> energy-model feature records.

Implements fit_plan.md sections 2-4: the execution-boundary table, the
per-phase feature composition, and the D1' dispatch resolution (2026-07-20
amendment). Pure CPU counting; no torch import; safe to run anywhere.

Execution composition (mirrors workloads/*.py run() exactly; the encoder
pass of enc-dec phases is unmeasured setup and is NOT counted):

    prefill            -> decoder.prefill(cfg, s)
    decode (dec-only)  -> decoder.prefill(cfg, s) + decoder.decode_total(cfg, s, K)
    encode (enc-only)  -> encoder.encode(cfg, s)   [ViT: fixed native sequence]
    encode (enc-dec)   -> encoder_decoder.encode(cfg, src)
    decoder_prefill    -> encoder_decoder.decoder_prefill(cfg, src, tgt_len)
    decode (enc-dec)   -> encoder_decoder.decoder_prefill(cfg, src, tgt_ctx)
                          + encoder_decoder.decode_total(cfg, src, tgt_ctx, K)

Spec-field semantics are grounded in sweep/point.py::_build_workload (what
physically ran): decoder-only decode context is spec.seq_len; enc-dec uses
spec.seq_len as the SOURCE length and spec.tgt_ctx as the established target
context; decoder_prefill uses spec.tgt_len.

n_launches (D1'): the GEMM-level kernel-launch convention built into
architectures/common.py and attention.py (each dense projection, norm, and
embedding gather = 1 launch; fused elementwise ops = 0; standard attention 3
vs flash 1; KV-cache reads/writes 0), summed through the composition above.
This is a structural PROXY for CUDA kernel launches, not a profiled count;
the fitted coefficient absorbs the proxy scale. n_fused_steps is 0 by
construction for every measured workload, so M4/M9 degenerate to M3/M8 plus
an information-criterion penalty.

Every failure raises BridgeError with the offending spec key: the frozen
dataset is expected to resolve completely, and anything else is a bug to
surface loudly, never to skip silently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from ..architectures import decoder as dec
from ..architectures import encoder as enc
from ..architectures import encoder_decoder as ed
from ..architectures.common import Precision
from ..architectures.configs import get as get_config
from ..energy_model import FEATURES

DEVICE = "rtx4090"   # EXP-002 is the RTX 4090 dataset; pass device=... for A100 later.

PRECISIONS: dict[str, Precision] = {
    "fp16": Precision(),  # compute/weight/act all fp16 (front-end default)
    "fp32": Precision(compute="fp32", weight="fp32", act="fp32"),
}

# Sweep spec naming -> front-end naming ("eager" is torch SDPBackend.MATH,
# counted as materialized-score "standard" attention).
ATTN_KINDS: dict[str, str] = {"flash": "flash", "eager": "standard"}

# Features that must be strictly positive for every measured workload.
_POSITIVE = ("to_mac", "to_nonlinear", "to_sram", "to_hbm", "n_launches")


class BridgeError(ValueError):
    """A sweep record that cannot be mapped to features. Always fail loudly."""


def _key(spec: Mapping) -> str:
    return str(spec.get("key", "<no-key>"))


def _need_int(spec: Mapping, field: str) -> int:
    v = spec.get(field)
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise BridgeError(
            f"spec.{field} must be a positive int, got {v!r} (key={_key(spec)})")
    return v


def _require_growing(spec: Mapping) -> None:
    mode = spec.get("decode_mode")
    if mode != "growing":
        raise BridgeError(
            f"bridge supports decode_mode='growing' only (the entire frozen "
            f"dataset); got {mode!r} (key={_key(spec)})")


def features_for_spec(spec: Mapping, *, device: str = DEVICE) -> dict[str, float]:
    """Map one sweep spec dict to an energy_model feature record."""
    for field in ("model", "arch", "phase", "precision", "attn_kind"):
        if field not in spec:
            raise BridgeError(f"spec missing '{field}' (key={_key(spec)})")
    if spec.get("batch_size", 1) != 1:
        raise BridgeError(
            f"bridge assumes batch_size=1 (the entire frozen dataset), got "
            f"{spec.get('batch_size')!r} (key={_key(spec)})")

    precision = spec["precision"]
    if precision not in PRECISIONS:
        raise BridgeError(
            f"unknown precision {precision!r}; known {sorted(PRECISIONS)} "
            f"(key={_key(spec)})")
    prec = PRECISIONS[precision]

    attn = spec["attn_kind"]
    if attn not in ATTN_KINDS:
        raise BridgeError(
            f"unknown attn_kind {attn!r}; known {sorted(ATTN_KINDS)} "
            f"(key={_key(spec)})")
    kind = ATTN_KINDS[attn]

    model, arch, phase = spec["model"], spec["arch"], spec["phase"]
    try:
        cfg = get_config(model)
    except Exception as exc:
        raise BridgeError(f"unknown model {model!r} (key={_key(spec)})") from exc
    if cfg.arch != arch:
        raise BridgeError(
            f"config arch {cfg.arch!r} != spec arch {arch!r} for {model!r} "
            f"(key={_key(spec)})")

    if arch == "decoder_only":
        s = _need_int(spec, "seq_len")
        if phase == "prefill":
            bd = dec.prefill(cfg, s, device=device, prec=prec,
                             attn_kind=kind).breakdown
        elif phase == "decode":
            _require_growing(spec)
            k = _need_int(spec, "decode_tokens")
            bd = (dec.prefill(cfg, s, device=device, prec=prec,
                              attn_kind=kind).breakdown
                  + dec.decode_total(cfg, s, k, device=device, prec=prec,
                                     attn_kind=kind).breakdown)
        else:
            raise BridgeError(
                f"decoder_only phase must be prefill|decode, got {phase!r} "
                f"(key={_key(spec)})")

    elif arch == "encoder_only":
        if phase != "encode":
            raise BridgeError(
                f"encoder_only phase must be encode, got {phase!r} "
                f"(key={_key(spec)})")
        if cfg.is_vision:
            bd = enc.encode(cfg, spec.get("seq_len"), device=device, prec=prec,
                            attn_kind=kind)   # seq_len ignored: native patches+CLS
        else:
            s = _need_int(spec, "seq_len")
            bd = enc.encode(cfg, s, device=device, prec=prec, attn_kind=kind)

    elif arch == "encoder_decoder":
        src = _need_int(spec, "seq_len")
        if phase == "encode":
            bd = ed.encode(cfg, src, device=device, prec=prec, attn_kind=kind)
        elif phase == "decoder_prefill":
            tgt = _need_int(spec, "tgt_len")
            bd = ed.decoder_prefill(cfg, src, tgt, device=device, prec=prec,
                                    attn_kind=kind)
        elif phase == "decode":
            _require_growing(spec)
            ctx = _need_int(spec, "tgt_ctx")
            k = _need_int(spec, "decode_tokens")
            bd = (ed.decoder_prefill(cfg, src, ctx, device=device, prec=prec,
                                     attn_kind=kind)
                  + ed.decode_total(cfg, src, ctx, k, device=device, prec=prec,
                                    attn_kind=kind))
        else:
            raise BridgeError(
                f"encoder_decoder phase must be encode|decoder_prefill|decode, "
                f"got {phase!r} (key={_key(spec)})")

    else:
        raise BridgeError(f"unknown arch {arch!r} (key={_key(spec)})")

    return _validated(bd.as_record(), spec)


def _validated(rec: Mapping[str, float], spec: Mapping) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in FEATURES:
        if f not in rec:
            raise BridgeError(
                f"front-end record missing feature '{f}' (key={_key(spec)})")
        v = float(rec[f])
        if not math.isfinite(v) or v < 0.0:
            raise BridgeError(
                f"feature {f}={v!r} is not finite and non-negative "
                f"(key={_key(spec)})")
        out[f] = v
    for f in _POSITIVE:
        if out[f] <= 0.0:
            raise BridgeError(
                f"feature {f} unexpectedly zero for a measured workload "
                f"(key={_key(spec)})")
    if out["n_fused_steps"] != 0.0:
        raise BridgeError(
            f"n_fused_steps must be 0 by construction, got "
            f"{out['n_fused_steps']!r} (key={_key(spec)})")
    return out


def features_for_record(record: Mapping, *, device: str = DEVICE) -> dict[str, float]:
    """Map one full sweep JSONL record (must be ok and not short-window)."""
    spec = record.get("spec")
    if not isinstance(spec, Mapping):
        raise BridgeError("record has no spec dict")
    if not record.get("ok"):
        raise BridgeError(f"record not ok (key={_key(spec)})")
    if record.get("short_window"):
        raise BridgeError(f"record is short-window (key={_key(spec)})")
    return features_for_spec(spec, device=device)


def load_latest_records(path: str | Path) -> list[dict]:
    """Read energy.jsonl and return the latest record per spec.key.

    Last-write-wins on duplicate keys (the sweep's resume convention),
    preserving first-seen key order. Parse errors and keyless records raise:
    the frozen dataset is clean, so anything else is corruption to surface.
    """
    p = Path(path)
    latest: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BridgeError(f"{p}:{lineno}: JSON parse error: {exc}") from exc
            key = rec.get("spec", {}).get("key")
            if not key:
                raise BridgeError(f"{p}:{lineno}: record without spec.key")
            latest[key] = rec
    return list(latest.values())
