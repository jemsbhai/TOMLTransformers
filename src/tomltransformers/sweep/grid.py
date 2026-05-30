"""Grid expander: a frozen EXP-002 config (dict) -> a list of PointSpecs.

Pure and deterministic (no torch, no GPU): the driver calls this to enumerate
what to measure, and it is unit-tested independently of any measurement.

The frozen config (configs/exp_002.yaml) defines sequence sweeps under `prefill`
and `decode`, but does not separately parametrize the encoder-only and
encoder-decoder phases. The expansion below maps them in the way that best
supports the paper's claims:

  decoder_only:
    - prefill  over prefill.seq_lens                       (compute-bound; MCER<1)
    - decode   over decode.context_lens, K per window      (memory-bound; MCER>>1)

  encoder_only:
    - encode   over prefill.seq_lens (text); ViT ignores seq_len (fixed by
               num_patches), so its points dedup to one per precision.

  encoder_decoder (the architecture whose unique claim is cross- vs self-cache):
    - encode          over src in prefill.seq_lens         (bidirectional, once)
    - decoder_prefill over src in prefill.seq_lens, tgt_len = ANCHOR
                      (isolates the source-dependent cross-K/V projection+cache)
    - decode, TWO coordinated sub-sweeps so the fit can SEPARATE the two memory
      contributions (this is the enc-dec-specific evidence):
        * source sweep: src in decode.context_lens, tgt_ctx = ANCHOR
                        -> cross-attention cache-read scales with SOURCE
        * target sweep: tgt_ctx in decode.context_lens, src = ANCHOR
                        -> self-attention cache-read scales with TARGET

  attention_compare: prefill points for the named small decoders, eager vs flash,
    over its own seq_lens, to show the O(s^2) eager-attention penalty.

ANCHOR (the held-constant dimension in each enc-dec sub-sweep) defaults to 1024,
placing the fixed dimension in the memory-bound regime where the cross/self
contrast is sharpest. Override via expand_grid(..., enc_dec_anchor=...).

Vision encoder names contain '/16'; we treat is_vision by asking the config
registry, not by string-munging here (the registry is the source of truth).
"""

from __future__ import annotations

from typing import List, Optional

from .point import PointSpec
from ..architectures.configs import get as get_config


# Phases that are a single bidirectional/forward pass over the source sequence.
_DEFAULT_ANCHOR = 1024


def _is_vision(model: str) -> bool:
    try:
        return bool(get_config(model).is_vision)
    except Exception:
        return False


def _arch_of(model: str, declared_arch: str) -> str:
    """Trust the config registry's arch if resolvable; else the declared bucket."""
    try:
        return get_config(model).arch
    except Exception:
        return declared_arch


def expand_grid(
    cfg: dict,
    *,
    enc_dec_anchor: int = _DEFAULT_ANCHOR,
    include_attention_compare: bool = True,
    weights: str = "random",
) -> List[PointSpec]:
    """Expand a frozen EXP-002 config dict into the ordered list of PointSpecs.

    Order is deterministic and grouped by (arch, phase, model) so a resumed sweep
    progresses predictably. Duplicate keys (e.g. ViT across text seq_lens) are
    removed, keeping first occurrence.
    """
    models = cfg.get("models", {})
    precisions = cfg.get("precisions", ["fp16"])
    workloads = cfg.get("workloads", {})
    prefill_seqs = workloads.get("prefill", {}).get("seq_lens", [])
    decode_ctxs = workloads.get("decode", {}).get("context_lens", [])
    K = workloads.get("decode", {}).get("tokens_per_window", 64)
    batch_size = cfg.get("batch_size", 1)

    out: List[PointSpec] = []

    def add(ps: PointSpec):
        out.append(ps)

    # ---- decoder_only: prefill + decode ----
    for model in models.get("decoder_only", []):
        arch = _arch_of(model, "decoder_only")
        for prec in precisions:
            for s in prefill_seqs:
                add(PointSpec(model=model, arch=arch, phase="prefill",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=s, batch_size=batch_size))
            for c in decode_ctxs:
                add(PointSpec(model=model, arch=arch, phase="decode",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=c, tgt_ctx=c, decode_tokens=K,
                              decode_mode="growing", batch_size=batch_size))

    # ---- encoder_only: encode ----
    for model in models.get("encoder_only", []):
        arch = _arch_of(model, "encoder_only")
        vision = _is_vision(model)
        for prec in precisions:
            if vision:
                # seq_len ignored (fixed by num_patches); one point per precision.
                add(PointSpec(model=model, arch=arch, phase="encode",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=None, batch_size=batch_size))
            else:
                for s in prefill_seqs:
                    add(PointSpec(model=model, arch=arch, phase="encode",
                                  precision=prec, attn_kind="flash", weights=weights,
                                  seq_len=s, batch_size=batch_size))

    # ---- encoder_decoder: encode + decoder_prefill + decode (2 sub-sweeps) ----
    for model in models.get("encoder_decoder", []):
        arch = _arch_of(model, "encoder_decoder")
        for prec in precisions:
            # encode over source.
            for s in prefill_seqs:
                add(PointSpec(model=model, arch=arch, phase="encode",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=s, batch_size=batch_size))
            # decoder_prefill over source, target anchored (cross-K/V projection).
            for s in prefill_seqs:
                add(PointSpec(model=model, arch=arch, phase="decoder_prefill",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=s, tgt_len=enc_dec_anchor, batch_size=batch_size))
            # decode, source sweep: vary src (cross-cache), hold target = anchor.
            for s in decode_ctxs:
                add(PointSpec(model=model, arch=arch, phase="decode",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=s, tgt_ctx=enc_dec_anchor, decode_tokens=K,
                              decode_mode="growing", batch_size=batch_size))
            # decode, target sweep: vary target (self-cache), hold src = anchor.
            for c in decode_ctxs:
                add(PointSpec(model=model, arch=arch, phase="decode",
                              precision=prec, attn_kind="flash", weights=weights,
                              seq_len=enc_dec_anchor, tgt_ctx=c, decode_tokens=K,
                              decode_mode="growing", batch_size=batch_size))

    # ---- attention-compare: eager vs flash prefill on small decoders ----
    if include_attention_compare:
        ac = cfg.get("attention_compare", {})
        ac_models = ac.get("models", [])
        ac_kinds = ac.get("kinds", ["eager", "flash"])
        ac_seqs = ac.get("seq_lens", [])
        ac_prec = ac.get("precision", "fp16")
        for model in ac_models:
            arch = _arch_of(model, "decoder_only")
            for kind in ac_kinds:
                for s in ac_seqs:
                    add(PointSpec(model=model, arch=arch, phase="prefill",
                                  precision=ac_prec, attn_kind=kind, weights=weights,
                                  seq_len=s, batch_size=batch_size))

    return _dedup(out)


def _dedup(specs: List[PointSpec]) -> List[PointSpec]:
    """Remove duplicate keys, keeping first occurrence, preserving order."""
    seen = set()
    unique: List[PointSpec] = []
    for ps in specs:
        k = ps.key()
        if k not in seen:
            seen.add(k)
            unique.append(ps)
    return unique


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict (thin wrapper; needs pyyaml)."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
