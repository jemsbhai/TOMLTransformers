"""Tests for the shared TO-counting blocks (tomltransformers.architectures.common).

Expected values are computed from to_costs (not hard-coded), so these tests
verify the blocks compose the cost model correctly and stay in sync with it.
"""

import pytest

from tomltransformers import energy_model as em
from tomltransformers import to_costs as tc
from tomltransformers.architectures import common as cm

DEV = "rtx4090"  # -> GDDR6X off-chip


# --- Precision presets --------------------------------------------------------
def test_precision_presets():
    assert cm.W4A16.weight == "int4" and cm.W4A16.compute == "fp16"
    assert cm.W8A8.compute == "int8" and cm.W8A8.weight == "int8"
    assert cm.FP16.compute == "fp16"


# --- TOBreakdown algebra ------------------------------------------------------
def test_breakdown_add_and_scale():
    a = cm.TOBreakdown(to_mac=1.0, to_sram=2.0)
    b = cm.TOBreakdown(to_mac=3.0, to_hbm=4.0, n_launches=1)
    s = a + b
    assert (s.to_mac, s.to_sram, s.to_hbm, s.n_launches) == (4.0, 2.0, 4.0, 1.0)
    assert a.scaled(3).to_mac == 3.0 and a.scaled(3).to_sram == 6.0


def test_breakdown_aggregates_and_mcer():
    c = cm.TOBreakdown(to_mac=2.0, to_nonlinear=2.0, to_sram=1.0, to_hbm=3.0)
    assert c.compute == 4.0
    assert c.memory == 4.0
    assert c.total == 8.0
    assert c.mcer == 1.0
    assert cm.TOBreakdown(to_mac=0.0, to_hbm=5.0).mcer == float("inf")


def test_total_helper():
    bs = [cm.TOBreakdown(to_mac=1.0), cm.TOBreakdown(to_mac=2.0), cm.TOBreakdown(to_hbm=4.0)]
    t = cm.total(bs)
    assert t.to_mac == 3.0 and t.to_hbm == 4.0


def test_as_record_keys_match_energy_model_features():
    assert set(cm.TOBreakdown().as_record().keys()) == set(em.FEATURES)


# --- linear -------------------------------------------------------------------
def test_linear_components():
    n, i, o = 10, 64, 128
    b = cm.linear(i, o, n, device=DEV, prec=cm.FP16)
    assert b.to_mac == pytest.approx(n * i * o * tc.mac("fp16"))
    assert b.to_hbm == pytest.approx(tc.mem_cost(i * o, "gddr6x", "fp16"))
    assert b.to_sram == pytest.approx(tc.mem_cost(n * (i + o), "sram", "fp16"))
    assert b.n_launches == 1


def test_linear_int4_weights_eighth_of_fp32_hbm():
    n, i, o = 10, 64, 128
    b32 = cm.linear(i, o, n, device=DEV, prec=cm.Precision(weight="fp32"))
    b4 = cm.linear(i, o, n, device=DEV, prec=cm.Precision(weight="int4"))
    assert b4.to_hbm == pytest.approx(b32.to_hbm / 8.0)


def test_linear_uses_device_offchip_tier():
    # A100 (HBM2E) weights are cheaper per word than 4090 (GDDR6X).
    a = cm.linear(64, 128, 10, device="a100")
    g = cm.linear(64, 128, 10, device="rtx4090")
    assert a.to_hbm < g.to_hbm


# --- activation and norm ------------------------------------------------------
def test_activation_nonlinear_cost_and_fused():
    b = cm.activation(100, kind="gelu")
    assert b.to_nonlinear == pytest.approx(100 * tc.op("gelu"))
    assert b.n_launches == 0          # fused by default


def test_norm_nonlinear_cost():
    n, d = 10, 64
    b = cm.norm(n, d, norm_type="rmsnorm", device=DEV)
    assert b.to_nonlinear == pytest.approx(n * d * tc.op("rmsnorm"))
    assert b.n_launches == 1


# --- ffn ----------------------------------------------------------------------
def test_ffn_gated_macs_and_launches():
    n, d, d_ff = 8, 64, 256
    b = cm.ffn(n, d, d_ff, ffn_type="gated", activation_kind="silu", device=DEV)
    expected_macs = n * (3 * d * d_ff + d_ff)   # gate + up + down + gating multiply
    assert b.to_mac == pytest.approx(expected_macs * tc.mac("fp16"))
    assert b.n_launches == 3                     # three GEMMs; activation/multiply fused


def test_ffn_standard_macs_and_launches():
    n, d, d_ff = 8, 64, 256
    b = cm.ffn(n, d, d_ff, ffn_type="standard", activation_kind="gelu", device=DEV)
    assert b.to_mac == pytest.approx(n * (2 * d * d_ff) * tc.mac("fp16"))
    assert b.n_launches == 2


def test_ffn_unknown_type_raises():
    with pytest.raises(ValueError):
        cm.ffn(8, 64, 256, ffn_type="mixture", activation_kind="gelu", device=DEV)


# --- embedding ----------------------------------------------------------------
def test_embedding_lookup_is_memory_only():
    n, d = 12, 64
    b = cm.embedding_lookup(n, d, device=DEV)
    assert b.to_mac == 0.0
    assert b.to_hbm == pytest.approx(tc.mem_cost(n * d, "gddr6x", "fp16"))
