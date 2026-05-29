"""Tests for the encoder-only front-end (tomltransformers.architectures.encoder)."""

import pytest

from tomltransformers import energy_model as em
from tomltransformers.architectures import configs as cf
from tomltransformers.architectures import encoder as en


# --- Arch guard ---------------------------------------------------------------
def test_arch_guard_rejects_non_encoder():
    with pytest.raises(ValueError):
        en.encode(cf.LLAMA_7B, 128)
    with pytest.raises(ValueError):
        en.encode(cf.T5_BASE, 128)


# --- Sequence-length resolution ----------------------------------------------
def test_sequence_length_resolution():
    assert en.sequence_length(cf.VIT_B16, None) == cf.VIT_B16.num_patches + 1   # 197
    assert en.sequence_length(cf.BERT_BASE, 256) == 256
    with pytest.raises(ValueError):
        en.sequence_length(cf.BERT_BASE, None)   # text encoder needs a length


# --- Records ------------------------------------------------------------------
def test_records_match_feature_keys_and_positive():
    b = en.encode(cf.BERT_BASE, 128)
    assert set(b.as_record().keys()) == set(em.FEATURES)
    assert b.total > 0


# --- Modality-dependent embedding --------------------------------------------
def test_vit_embedding_is_a_gemm_bert_is_a_gather():
    vit = en._embedding(cf.VIT_B16, en.sequence_length(cf.VIT_B16, None), device="rtx4090")
    bert = en._embedding(cf.BERT_BASE, 128, device="rtx4090")
    assert vit.to_mac > 0      # patch projection
    assert bert.to_mac == 0    # token gather, memory only


# --- Single pass is compute-bound and amortizes weights ----------------------
def test_compute_bound_at_typical_length():
    b = en.encode(cf.BERT_BASE, 512)
    assert b.compute > b.memory
    assert b.mcer < 1.0


def test_mcer_decreases_with_sequence_length():
    # Weights are loaded once and amortized over more tokens (and attention work
    # grows super-linearly), so MCER falls as the sequence grows.
    assert en.encode(cf.BERT_BASE, 512).mcer < en.encode(cf.BERT_BASE, 128).mcer


def test_vit_runs_at_fixed_sequence_length():
    b = en.encode(cf.VIT_B16)
    assert b.total > 0
    assert b.mcer < 1.0


# --- Bidirectional attention: standard materializes the score matrix ---------
def test_standard_attention_adds_offchip_traffic_growing_with_length():
    flash_512 = en.encode(cf.BERT_BASE, 512, attn_kind="flash")
    std_512 = en.encode(cf.BERT_BASE, 512, attn_kind="standard")
    assert std_512.to_hbm > flash_512.to_hbm

    r_512 = std_512.mcer / flash_512.mcer
    r_2048 = (en.encode(cf.BERT_BASE, 2048, attn_kind="standard").mcer
              / en.encode(cf.BERT_BASE, 2048, attn_kind="flash").mcer)
    assert r_2048 > r_512   # O(s^2) penalty grows with sequence length


# --- Head is optional and additive -------------------------------------------
def test_head_is_additive():
    with_head = en.encode(cf.VIT_B16, include_head=True).total
    without_head = en.encode(cf.VIT_B16, include_head=False).total
    assert with_head > without_head
