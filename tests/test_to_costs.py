"""Unit tests for the TO cost model (tomltransformers.to_costs).

These verify not just internal consistency but physical consistency: the TO
values must reproduce the cited per-bit energy figures from the literature.
No GPU or measurements required.
"""

import pytest

from tomltransformers import to_costs as tc


# --- Positivity and basic structure ------------------------------------------
def test_all_op_costs_positive():
    for name, c in tc.OP_COST.items():
        assert c.value > 0, name


def test_all_mem_costs_positive():
    for name, c in tc.MEM_TIER.items():
        assert c.value > 0, name


# --- Memory hierarchy ordering -----------------------------------------------
def test_memory_hierarchy_strictly_increasing():
    costs = [tc.mem_word(t) for t in tc.MEM_TIER_ORDER]
    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)  # no ties


def test_onchip_cheaper_than_all_offchip():
    onchip_max = max(tc.mem_word(t) for t in tc.ONCHIP_TIERS)
    offchip_min = min(tc.mem_word(t) for t in tc.OFFCHIP_TIERS)
    assert onchip_max < offchip_min


def test_offchip_to_sram_ratio_matches_literature():
    # ~30-50x for A100 (HBM2E) and 4090 (GDDR6X) vs on-chip SRAM.
    assert 30 < tc.mem_word("hbm2e") / tc.mem_word("sram") < 60
    assert 30 < tc.mem_word("gddr6x") / tc.mem_word("sram") < 60


# --- Physical consistency with cited pJ/bit figures --------------------------
def test_pj_per_bit_reproduces_literature():
    assert tc.pj_per_bit("hbm2e") == pytest.approx(6.00, abs=1e-6)
    assert tc.pj_per_bit("hbm3e") == pytest.approx(4.05, abs=1e-6)
    assert tc.pj_per_bit("hbm4") == pytest.approx(2.40, abs=1e-6)
    assert tc.pj_per_bit("gddr6x") == pytest.approx(7.25, abs=1e-6)
    assert tc.pj_per_bit("gddr6") == pytest.approx(7.50, abs=1e-6)
    assert tc.pj_per_bit("sram") == pytest.approx(0.15625, abs=1e-6)  # Horowitz 8KB


def test_gddr6x_is_less_efficient_than_hbm2e():
    # The 4090's memory is worse per bit than the A100's: the cross-device point.
    assert tc.pj_per_bit("gddr6x") > tc.pj_per_bit("hbm2e")


# --- Device registry ----------------------------------------------------------
def test_device_offchip_mapping():
    assert tc.offchip_tier("rtx4090") == "gddr6x"
    assert tc.offchip_tier("a100") == "hbm2e"
    assert tc.offchip_tier("h100") == "hbm3e"


def test_unknown_device_raises():
    with pytest.raises(KeyError):
        tc.offchip_tier("rtx5090")


# --- Precision ----------------------------------------------------------------
def test_precision_mac_monotone():
    assert tc.mac("fp32") >= tc.mac("fp16") >= tc.mac("int8") >= tc.mac("int4")
    assert tc.mac("fp16") == pytest.approx(tc.mac("bf16"))


def test_words_per_element():
    assert tc.words_per_element("fp32") == 1.0
    assert tc.words_per_element("fp16") == 0.5
    assert tc.words_per_element("int8") == 0.25
    assert tc.words_per_element("int4") == 0.125


def test_mem_cost_scales_with_count():
    assert tc.mem_cost(2_000, "hbm2e") == pytest.approx(2 * tc.mem_cost(1_000, "hbm2e"))


def test_int4_weights_cost_eighth_of_fp32_offchip_traffic():
    n = 1_000_000
    assert tc.mem_cost(n, "gddr6x", "int4") == pytest.approx(
        tc.mem_cost(n, "gddr6x", "fp32") / 8.0
    )


# --- Headline ratio the paper leans on ---------------------------------------
def test_softmax_is_five_macs_in_tos():
    # FLOPs counts softmax as ~5 ops; TOML counts 25000 TOs (= 5 MACs).
    assert tc.op("softmax") / tc.op("mac") == 5.0


# --- Error handling -----------------------------------------------------------
def test_unknown_keys_raise():
    with pytest.raises(KeyError):
        tc.op("not_an_op")
    with pytest.raises(KeyError):
        tc.mem_word("l3")
    with pytest.raises(KeyError):
        tc.mac("fp8")


# --- Calibration-target integrity --------------------------------------------
def test_calibration_targets_reference_real_keys():
    assert len(tc.CALIBRATION_TARGETS) > 0
    for key in tc.CALIBRATION_TARGETS:
        head, _, name = key.partition(":")
        if head == "mem":
            assert name in tc.MEM_TIER
        elif head == "prec":
            assert name in tc.PRECISION_MAC_MULT
        else:
            assert key in tc.OP_COST


def test_calibration_targets_include_headline_estimates_and_memory():
    for must in ("softmax", "silu", "gelu", "layernorm", "rmsnorm"):
        assert must in tc.CALIBRATION_TARGETS
    # Every memory tier's effective on-GPU cost is a target.
    for tier in tc.MEM_TIER:
        assert f"mem:{tier}" in tc.CALIBRATION_TARGETS


# --- Provenance and citations -------------------------------------------------
def test_provenance_values_valid():
    for table in (tc.OP_COST, tc.MEM_TIER, tc.PRECISION_MAC_MULT):
        for c in table.values():
            assert isinstance(c.provenance, tc.Provenance)


def test_cited_costs_have_real_sources():
    # Anything claimed as measured/published must point to a real citation.
    for table in (tc.OP_COST, tc.MEM_TIER):
        for name, c in table.items():
            if c.provenance in (tc.Provenance.MEASURED, tc.Provenance.PUBLISHED_TOML):
                assert c.source in tc.SOURCES, (name, c.source)
