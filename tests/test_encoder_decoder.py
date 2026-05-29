"""Tests for the encoder-decoder front-end (tomltransformers.architectures.encoder_decoder)."""

import pytest

from tomltransformers import energy_model as em
from tomltransformers.architectures import configs as cf
from tomltransformers.architectures import encoder_decoder as ed

T5 = cf.T5_BASE
BART = cf.BART_BASE


# --- Arch guard ---------------------------------------------------------------
def test_arch_guard_rejects_non_enc_dec():
    with pytest.raises(ValueError):
        ed.encode(cf.LLAMA_7B, 128)
    with pytest.raises(ValueError):
        ed.decode_step(cf.BERT_BASE, 128, 16)


# --- Records ------------------------------------------------------------------
def test_records_match_feature_keys():
    for b in (ed.encode(T5, 256), ed.decoder_prefill(T5, 256, 1), ed.decode_step(T5, 256, 32)):
        assert set(b.as_record().keys()) == set(em.FEATURES)
        assert b.total > 0


# --- Phase transition: encode compute-bound, decode memory-bound -------------
def test_encode_compute_bound_decode_memory_bound():
    enc = ed.encode(T5, 512)
    ds = ed.decode_step(T5, 512, 256)
    assert enc.compute > enc.memory and enc.mcer < 1.0
    assert ds.memory > ds.compute and ds.mcer > 5.0
    assert enc.mcer < ds.mcer


# --- Two caches: self grows with target, cross grows with source -------------
def test_self_kv_cache_grows_with_target_context():
    short = ed.decode_step(T5, 512, 64)
    long = ed.decode_step(T5, 512, 4096)
    assert long.to_hbm > short.to_hbm        # self-KV read grows with target context


def test_cross_kv_cache_grows_with_source_length():
    small_src = ed.decode_step(T5, 256, 64)
    large_src = ed.decode_step(T5, 4096, 64)
    assert large_src.to_hbm > small_src.to_hbm   # cross-KV read grows with source length


# --- decode_total sums per-step costs ----------------------------------------
def test_decode_total_sums_steps():
    dt = ed.decode_total(T5, src_len=512, tgt_prefill_len=1, n_generate=3)
    manual = (ed.decode_step(T5, 512, 2)
              + ed.decode_step(T5, 512, 3)
              + ed.decode_step(T5, 512, 4))
    assert dt.to_mac == pytest.approx(manual.to_mac)
    assert dt.to_hbm == pytest.approx(manual.to_hbm)


# --- Depth: T5 (12+12) does more encoder work than BART (6+6) ----------------
def test_t5_encoder_heavier_than_bart():
    assert ed.encode(T5, 512).total > ed.encode(BART, 512).total


# --- Cross-attention adds work beyond self-attention only --------------------
def test_decoder_prefill_projects_cross_kv():
    # Prefill projects cross-K/V from the source, so larger source -> more MACs,
    # even at fixed target length (a property absent from a self-attention-only decoder).
    small_src = ed.decoder_prefill(T5, 128, 8)
    large_src = ed.decoder_prefill(T5, 2048, 8)
    assert large_src.to_mac > small_src.to_mac
