"""Frozen T2 calibration subset for the A100 phase.

Source of truth for the eight fp16 calibration cells of test T2 (scale-
calibrated cross-platform transfer), as resolved to explicit seed-less keys
by a100_amendment.md section 16.2 (2026-08-17), and for the three alternate
enc-dec decode cells of the descriptive sensitivity companion (section 16.3).

Keys here are SEED-LESS: the frozen A100 grid joins a derived per-point seed
to every key (sweep/grid_passes.py), so callers match records on
seedless_key(record["spec"]["key"]). tests/test_a100_calibration_cells.py
locks every entry against the frozen enumeration and against the measured
dataset (exactly one ok record per key).

Pure Python; no torch, no data access at import time.
"""

from __future__ import annotations

import re

_SEED_SUFFIX = re.compile(r"\|seed\d+$")


def seedless_key(key: str) -> str:
    """Strip the trailing '|seed<digits>' the A100 grid appends to every key.

    Idempotent; keys without a seed suffix (the 4090 grid) are returned
    unchanged.
    """
    return _SEED_SUFFIX.sub("", str(key))


# The 8-cell fp16 calibration subset (amendment section 9, resolved in 16.2).
# Order is documentary only; the fit treats it as a set.
T2_CALIBRATION_KEYS: tuple[str, ...] = (
    "decoder_only|GPT-2|prefill|fp16|flash|random|s1024|b1",
    "decoder_only|GPT-2|decode|fp16|flash|random|s1024|b1|ctx1024|k64|growing",
    "decoder_only|GPT-2-XL|prefill|fp16|flash|random|s1024|b1",
    "decoder_only|GPT-2-XL|decode|fp16|flash|random|s1024|b1|ctx1024|k64|growing",
    "encoder_only|BERT-base|encode|fp16|flash|random|s1024|b1",
    "encoder_only|BERT-large|encode|fp16|flash|random|s1024|b1",
    "encoder_decoder|T5-small|decode|fp16|flash|random|s2048|b1|ctx1024|k64|growing",
    "encoder_decoder|BART-base|encode|fp16|flash|random|s1024|b1",
)

# The enc-dec decode calibration cell (section 16.2 rule: ctx = 1024 mirrors
# the decoder calibration cells; s = 2048 is the source-arm cell nearest the
# 1024 anchor in log space).
T2_ENC_DEC_DECODE_KEY: str = T2_CALIBRATION_KEYS[6]

# All four fp16 T5-small decode cells of the frozen grid (both sub-sweep
# arms; the (s1024, ctx1024) center exists only in fp32, section 16.1).
T5_SMALL_DECODE_FP16_KEYS: tuple[str, ...] = (
    "encoder_decoder|T5-small|decode|fp16|flash|random|s128|b1|ctx1024|k64|growing",
    "encoder_decoder|T5-small|decode|fp16|flash|random|s2048|b1|ctx1024|k64|growing",
    "encoder_decoder|T5-small|decode|fp16|flash|random|s1024|b1|ctx128|k64|growing",
    "encoder_decoder|T5-small|decode|fp16|flash|random|s1024|b1|ctx2048|k64|growing",
)

# The cell section 9 named and the frozen grid does not contain (16.1).
T5_SMALL_DECODE_FP16_CENTER_ABSENT: str = (
    "encoder_decoder|T5-small|decode|fp16|flash|random|s1024|b1|ctx1024|k64|growing"
)

# Sensitivity companion (16.3): cell 7 replaced in turn by each of these.
T2_SENSITIVITY_ALTERNATES: tuple[str, ...] = tuple(
    k for k in T5_SMALL_DECODE_FP16_KEYS if k != T2_ENC_DEC_DECODE_KEY
)


def is_calibration_record(record: dict) -> bool:
    """True if a sweep record's spec.key (seed stripped) is a calibration cell."""
    return seedless_key(record["spec"]["key"]) in T2_CALIBRATION_KEYS
