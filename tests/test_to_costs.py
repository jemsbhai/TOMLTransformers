"""Unit tests for the TO cost model (tomltransformers.to_costs)."""

import pytest

from tomltransformers import to_costs as tc


def test_all_op_costs_positive():
    for name, c in tc.OP_COST.items():
        assert c.value > 0, name


def test_all_mem_costs_positive():
    for name, c in tc.MEM_TIER.items():
        assert c.value > 0, name


def test_memory_hierarchy_strictly_increasing():
    costs = [tc.mem_word(t) for t in tc.MEM_TIER_ORDER]
    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)  # no ties


def test_hbm_dominates_sram_by_order_of_magnitude():
    # The SRAM->HBM gap is what creates the prefill/decode MCER transition.
    assert tc.mem_word("hbm") / tc.mem_word("sram") > 40.0


def test_precision_mac_monotone():
    assert tc.mac("fp32") >= tc.mac("fp16") >= tc.mac("int8") >= tc.mac("int4")
    assert tc.mac("fp16") == pytest.approx(tc.mac("bf16"))


def test_words_per_element():
    assert tc.words_per_element("fp32") == 1.0
    assert tc.words_per_element("fp16") == 0.5
    assert tc.words_per_element("int8") == 0.25
    assert tc.words_per_element("int4") == 0.125


def test_mem_cost_scales_with_count():
    assert tc.mem_cost(2_000, "hbm") == pytest.approx(2 * tc.mem_cost(1_000, "hbm"))


def test_int4_weights_cost_eighth_of_fp32_hbm_traffic():
    n = 1_000_000
    assert tc.mem_cost(n, "hbm", "int4") == pytest.approx(tc.mem_cost(n, "hbm", "fp32") / 8.0)


def test_softmax_is_five_macs_in_tos():
    # FLOPs counts softmax as ~5 ops; TOML counts 25000 TOs (= 5 MACs),
    # i.e. the 5000x-vs-FLOPs framing the paper leans on. Guard this ratio.
    assert tc.op("softmax") / tc.op("mac") == 5.0


def test_unknown_keys_raise():
    with pytest.raises(KeyError):
        tc.op("not_an_op")
    with pytest.raises(KeyError):
        tc.mem_word("l3")
    with pytest.raises(KeyError):
        tc.mac("fp8")


def test_calibration_targets_reference_real_keys():
    assert len(tc.CALIBRATION_TARGETS) > 0
    for key in tc.CALIBRATION_TARGETS:
        if key.startswith("mem:"):
            assert key.split(":", 1)[1] in tc.MEM_TIER
        elif key.startswith("prec:"):
            assert key.split(":", 1)[1] in tc.PRECISION_MAC_MULT
        else:
            assert key in tc.OP_COST


def test_calibration_targets_include_the_headline_estimates():
    # The values the measurement campaign must fit, not assert.
    for must in ("softmax", "silu", "gelu", "layernorm", "rmsnorm"):
        assert must in tc.CALIBRATION_TARGETS
    assert "mem:hbm" in tc.CALIBRATION_TARGETS


def test_provenance_values_valid():
    for table in (tc.OP_COST, tc.MEM_TIER, tc.PRECISION_MAC_MULT):
        for c in table.values():
            assert isinstance(c.provenance, tc.Provenance)
