"""Locks the T2 calibration subset to the frozen A100 enumeration and dataset.

a100_amendment.md section 16.2 (2026-08-17) resolves the eight fp16 T2
calibration cells to explicit seed-less keys. These tests are the
spec-to-enumeration reconciliation gate that section 16.1 identified as
missing at grid freeze: every named cell must resolve to exactly one point of
the frozen grid (configs/exp_002_a100.yaml expanded by the frozen expander)
and to exactly one ok record of the measured dataset. Pure CPU.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tomltransformers.fit.a100_calibration import (
    T2_CALIBRATION_KEYS,
    T2_ENC_DEC_DECODE_KEY,
    T2_SENSITIVITY_ALTERNATES,
    T5_SMALL_DECODE_FP16_CENTER_ABSENT,
    T5_SMALL_DECODE_FP16_KEYS,
    is_calibration_record,
    seedless_key,
)
from tomltransformers.fit.bridge import load_latest_records
from tomltransformers.sweep.grid import load_config
from tomltransformers.sweep.grid_passes import expand_passes

_ROOT = Path(__file__).resolve().parents[1]
_A100_YAML = _ROOT / "configs" / "exp_002_a100.yaml"
_A100_DATA = _ROOT / "experiments" / "exp_002_size_sweep" / "a100" / "energy.jsonl"

_EXTENSION_MODELS = {"LLaMA-7B", "Mistral-7B"}


def _specs():
    return expand_passes(load_config(str(_A100_YAML)))


def _by_seedless_key():
    counts = Counter()
    by_key = {}
    for ps in _specs():
        k = seedless_key(ps.key())
        counts[k] += 1
        by_key[k] = ps
    return counts, by_key


# --- seedless_key ------------------------------------------------------------

def test_seedless_key_strips_only_a_trailing_seed_suffix():
    k = "decoder_only|GPT-2|prefill|fp16|flash|random|s1024|b1"
    assert seedless_key(k + "|seed1617754261") == k
    assert seedless_key(k) == k                       # 4090-style key: unchanged
    assert seedless_key(seedless_key(k + "|seed7")) == k   # idempotent


def test_every_frozen_key_carries_a_seed_that_strips_cleanly():
    for ps in _specs():
        full = ps.key()
        assert full.endswith(f"|seed{ps.seed}")
        assert seedless_key(full) == full[: -len(f"|seed{ps.seed}")]


# --- calibration set vs the frozen enumeration --------------------------------

def test_calibration_set_has_eight_distinct_keys():
    assert len(T2_CALIBRATION_KEYS) == 8
    assert len(set(T2_CALIBRATION_KEYS)) == 8


def test_each_calibration_key_resolves_to_exactly_one_frozen_point():
    counts, _ = _by_seedless_key()
    for k in T2_CALIBRATION_KEYS:
        assert counts[k] == 1, k


def test_calibration_cells_are_fp16_flash_random_shared_grid():
    _, by_key = _by_seedless_key()
    for k in T2_CALIBRATION_KEYS:
        ps = by_key[k]
        assert ps.precision == "fp16", k
        assert ps.attn_kind == "flash", k
        assert ps.weights == "random", k
        assert ps.model not in _EXTENSION_MODELS, k
        assert ps.batch_size == 1, k


def test_calibration_cells_span_size_regime_family():
    # Section 9: "spanning size x regime x family". Two decoder sizes, both
    # decoder regimes, both encoders, both enc-dec models with encode and decode.
    _, by_key = _by_seedless_key()
    cells = [by_key[k] for k in T2_CALIBRATION_KEYS]
    assert {c.model for c in cells} == {
        "GPT-2", "GPT-2-XL", "BERT-base", "BERT-large", "T5-small", "BART-base"}
    assert {c.arch for c in cells} == {"decoder_only", "encoder_only", "encoder_decoder"}
    assert {c.phase for c in cells} == {"prefill", "decode", "encode"}


def test_enc_dec_decode_cell_follows_the_section_16_2_rule():
    _, by_key = _by_seedless_key()
    ps = by_key[T2_ENC_DEC_DECODE_KEY]
    assert ps.model == "T5-small" and ps.phase == "decode" and ps.precision == "fp16"
    assert ps.tgt_ctx == 1024      # ctx = 1024 mirrors the decoder decode cells
    assert ps.seq_len == 2048      # source-arm cell nearest the 1024 anchor (log space)


def test_root_cause_center_cell_exists_only_in_fp32():
    counts, _ = _by_seedless_key()
    assert counts[T5_SMALL_DECODE_FP16_CENTER_ABSENT] == 0
    fp32_center = T5_SMALL_DECODE_FP16_CENTER_ABSENT.replace("|fp16|", "|fp32|")
    assert counts[fp32_center] == 1


def test_all_four_fp16_t5_decode_cells_exist_and_alternates_are_the_other_three():
    counts, _ = _by_seedless_key()
    assert len(T5_SMALL_DECODE_FP16_KEYS) == 4
    for k in T5_SMALL_DECODE_FP16_KEYS:
        assert counts[k] == 1, k
    assert len(T2_SENSITIVITY_ALTERNATES) == 3
    assert set(T2_SENSITIVITY_ALTERNATES) | {T2_ENC_DEC_DECODE_KEY} == set(T5_SMALL_DECODE_FP16_KEYS)
    assert T2_ENC_DEC_DECODE_KEY not in T2_SENSITIVITY_ALTERNATES


# --- calibration set vs the measured dataset ----------------------------------

def test_each_calibration_key_matches_exactly_one_ok_record_in_the_a100_data():
    records = load_latest_records(_A100_DATA)
    counts = Counter(seedless_key(r["spec"]["key"]) for r in records if r.get("ok"))
    for k in T2_CALIBRATION_KEYS:
        assert counts[k] == 1, k
    for k in T2_SENSITIVITY_ALTERNATES:
        assert counts[k] == 1, k
    assert sum(1 for r in records if is_calibration_record(r)) == 8
