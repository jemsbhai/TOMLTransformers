"""Tests for attention TO accounting (tomltransformers.architectures.attention).

The off-chip tier is derived through the device registry (2026-07-24
correction: rtx4090 = Laptop GPU GDDR6), never named literally.
"""

import pytest

from tomltransformers import to_costs as tc
from tomltransformers.architectures import attention as at
from tomltransformers.architectures import configs as cf

DEV = "rtx4090"
OFFCHIP = tc.offchip_tier(DEV)   # -> "gddr6" (registry-derived)
LLAMA = cf.LLAMA_7B      # MHA (kv_heads = 32)
MISTRAL = cf.MISTRAL_7B  # GQA (kv_heads = 8)
GPT2 = cf.GPT2


# --- pair counting ------------------------------------------------------------
def test_attention_pairs():
    assert at.attention_pairs(128, 128, causal=True) == 128 * 129 // 2   # prefill
    assert at.attention_pairs(1, 500, causal=True) == 500                # decode step
    assert at.attention_pairs(128, 128, causal=False) == 128 * 128       # bidirectional
    assert at.attention_pairs(64, 200, causal=False) == 64 * 200         # cross-attn


# --- standard vs flash: identical compute, different memory/dispatch ---------
def test_standard_and_flash_have_identical_compute():
    std = at.attention_core(128, 128, LLAMA, device=DEV, causal=True, kind="standard")
    fl = at.attention_core(128, 128, LLAMA, device=DEV, causal=True, kind="flash")
    assert std.to_mac == fl.to_mac
    assert std.to_nonlinear == fl.to_nonlinear


def test_flash_has_zero_score_offchip_traffic_standard_does_not():
    std = at.attention_core(128, 128, LLAMA, device=DEV, causal=True, kind="standard")
    fl = at.attention_core(128, 128, LLAMA, device=DEV, causal=True, kind="flash")
    assert std.to_hbm > 0
    assert fl.to_hbm == 0.0


def test_launch_counts():
    std = at.attention_core(64, 64, LLAMA, device=DEV, causal=True, kind="standard")
    fl = at.attention_core(64, 64, LLAMA, device=DEV, causal=True, kind="flash")
    assert std.n_launches == 3   # QK^T, softmax, AV
    assert fl.n_launches == 1    # fused


def test_unknown_attention_kind_raises():
    with pytest.raises(ValueError):
        at.attention_core(64, 64, LLAMA, device=DEV, causal=True, kind="linear")


# --- standard score traffic is O(s^2) ----------------------------------------
def test_standard_score_traffic_quadratic_in_sequence():
    c1 = at.attention_core(128, 128, LLAMA, device=DEV, causal=False, kind="standard")
    c2 = at.attention_core(256, 256, LLAMA, device=DEV, causal=False, kind="standard")
    assert c2.to_hbm == pytest.approx(4.0 * c1.to_hbm)   # (256/128)^2


# --- causal masking roughly halves the score work ----------------------------
def test_causal_halves_score_work():
    causal = at.attention_core(128, 128, LLAMA, device=DEV, causal=True, kind="flash")
    full = at.attention_core(128, 128, LLAMA, device=DEV, causal=False, kind="flash")
    ratio = causal.to_mac / full.to_mac
    assert 0.4 < ratio < 0.6


# --- GQA shrinks projections and KV cache ------------------------------------
def test_gqa_reduces_qkv_projection_cost():
    ll = at.qkv_projection(64, 64, LLAMA, device=DEV)
    mi = at.qkv_projection(64, 64, MISTRAL, device=DEV)
    assert mi.to_mac < ll.to_mac   # fewer KV heads -> cheaper K, V


def test_kv_cache_read_matches_formula_and_gqa_is_smaller():
    b = at.kv_cache_read(100, LLAMA, device=DEV)
    assert b.to_hbm == pytest.approx(tc.mem_cost(2 * 100 * 32 * 128, OFFCHIP, "fp16"))
    mi = at.kv_cache_read(100, MISTRAL, device=DEV)
    assert mi.to_hbm < b.to_hbm    # GQA: 8 KV heads vs 32


def test_kv_cache_read_uses_device_tier():
    g = at.kv_cache_read(100, LLAMA, device="rtx4090")
    a = at.kv_cache_read(100, LLAMA, device="a100")
    assert a.to_hbm < g.to_hbm     # HBM2E cheaper per word than GDDR6


# --- output projection --------------------------------------------------------
def test_output_projection_matches_formula():
    o = at.output_projection(64, GPT2, device=DEV)
    assert o.to_mac == pytest.approx(64 * (GPT2.n_heads * GPT2.d_head) * GPT2.d_model * tc.mac("fp16"))
