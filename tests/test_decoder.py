"""Tests for the decoder-only front-end (tomltransformers.architectures.decoder)."""

import pytest

from tomltransformers import energy_model as em
from tomltransformers.architectures import configs as cf
from tomltransformers.architectures import decoder as dec

LLAMA = cf.LLAMA_7B
MISTRAL = cf.MISTRAL_7B


# --- Records and basic sanity ------------------------------------------------
def test_records_match_feature_keys_and_are_positive():
    pf = dec.prefill(LLAMA, 1024)
    ds = dec.decode_step(LLAMA, 1024)
    assert set(pf.record().keys()) == set(em.FEATURES)
    assert set(ds.record().keys()) == set(em.FEATURES)
    assert pf.breakdown.total > 0 and ds.breakdown.total > 0
    assert pf.n_tokens == 1024 and ds.n_tokens == 1


def test_arch_guard_rejects_non_decoder():
    with pytest.raises(ValueError):
        dec.prefill(cf.BERT_BASE, 128)
    with pytest.raises(ValueError):
        dec.decode_step(cf.T5_BASE, 128)


# --- The headline: prefill/decode phase transition ---------------------------
def test_prefill_compute_bound_decode_memory_bound():
    s = 2048
    pf = dec.prefill(LLAMA, s)
    ds = dec.decode_step(LLAMA, s)
    # Prefill: compute dominates (weights amortized over s tokens).
    assert pf.breakdown.compute > pf.breakdown.memory
    assert pf.mcer < 1.0
    # Decode: memory dominates (weights reloaded for one token + KV read).
    assert ds.breakdown.memory > ds.breakdown.compute
    assert ds.mcer > 5.0


def test_decode_to_prefill_mcer_ratio_scales_with_sequence():
    s = 2048
    pf = dec.prefill(LLAMA, s)
    ds = dec.decode_step(LLAMA, s)
    ratio = ds.mcer / pf.mcer
    assert ratio > 50.0          # order of s; the amortization factor


def test_phase_transition_holds_for_gqa_model():
    pf = dec.prefill(MISTRAL, 2048)
    ds = dec.decode_step(MISTRAL, 2048)
    assert pf.mcer < 1.0 < ds.mcer


# --- KV cache grows the decode memory with context ---------------------------
def test_decode_offchip_traffic_grows_with_context():
    short = dec.decode_step(LLAMA, 256)
    long = dec.decode_step(LLAMA, 8192)
    assert long.breakdown.to_hbm > short.breakdown.to_hbm


# --- Standard vs FlashAttention at long context ------------------------------
def test_standard_attention_adds_offchip_traffic_and_raises_prefill_mcer():
    s = 8192
    flash = dec.prefill(LLAMA, s, attn_kind="flash")
    standard = dec.prefill(LLAMA, s, attn_kind="standard")
    assert standard.breakdown.to_hbm > flash.breakdown.to_hbm
    assert standard.mcer > flash.mcer


# --- decode_total sums the per-step costs ------------------------------------
def test_decode_total_sums_steps():
    dt = dec.decode_total(LLAMA, prefill_len=100, n_generate=3)
    manual = (dec.decode_step(LLAMA, 101).breakdown
              + dec.decode_step(LLAMA, 102).breakdown
              + dec.decode_step(LLAMA, 103).breakdown)
    assert dt.n_tokens == 3
    assert dt.breakdown.to_mac == pytest.approx(manual.to_mac)
    assert dt.breakdown.to_hbm == pytest.approx(manual.to_hbm)
