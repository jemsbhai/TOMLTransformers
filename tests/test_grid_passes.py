"""Locks the frozen A100 grid enumeration (Step 6 of the approved amendment).

The grid file configs/exp_002_a100.yaml is frozen: these tests pin the exact
expansion (98 points and every per-stratum count from a100_amendment.md
sections 5-7), the deterministic per-point seed derivation, and the
expected_points integrity guard. Pure CPU (no torch, no GPU).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tomltransformers.sweep.grid import load_config
from tomltransformers.sweep.grid_passes import derive_seed, expand_passes

_ROOT = Path(__file__).resolve().parents[1]
_A100_YAML = _ROOT / "configs" / "exp_002_a100.yaml"

_SHARED_DECODERS = {"DistilGPT2", "GPT-2", "GPT-2-medium", "GPT-2-XL"}
_SHARED_ENCODERS = {"BERT-base", "BERT-large"}
_ENC_DEC = {"T5-small", "BART-base"}
_EXTENSION = {"LLaMA-7B", "Mistral-7B"}
_SHARED_SEQS = {128, 1024, 2048}


def _cfg():
    return load_config(str(_A100_YAML))


def _specs():
    return expand_passes(_cfg())


def _minimal_cfg(**overrides):
    """A tiny valid multi-pass config for error-path tests (expands to 1 point)."""
    cfg = {
        "seed": {"master": 42},
        "passes": [
            {
                "name": "tiny",
                "weights": "random",
                "include_attention_compare": False,
                "config": {
                    "models": {"decoder_only": ["GPT-2"]},
                    "precisions": ["fp16"],
                    "batch_size": 1,
                    "workloads": {"prefill": {"seq_lens": [128]}},
                },
            }
        ],
    }
    cfg.update(overrides)
    return cfg


# --- strata helpers -----------------------------------------------------------

def _stratum_shared_decoder(specs):
    return [ps for ps in specs
            if ps.arch == "decoder_only" and ps.model in _SHARED_DECODERS
            and ps.attn_kind == "flash" and ps.weights == "random"
            and ps.seq_len in _SHARED_SEQS]


def _stratum_shared_encoder(specs):
    return [ps for ps in specs if ps.arch == "encoder_only"]


def _stratum_enc_dec(specs, precision):
    return [ps for ps in specs
            if ps.arch == "encoder_decoder" and ps.precision == precision]


def _stratum_eager(specs):
    return [ps for ps in specs if ps.attn_kind == "eager"]


def _stratum_extension(specs):
    return [ps for ps in specs if ps.model in _EXTENSION]


def _stratum_spot(specs):
    return [ps for ps in specs if ps.model == "GPT-2" and ps.seq_len == 512
            and ps.attn_kind == "flash"]


# --- the frozen enumeration ---------------------------------------------------

def test_total_points_is_98():
    assert len(_specs()) == 98


def test_strata_partition_the_grid():
    specs = _specs()
    strata = {
        "decoder": _stratum_shared_decoder(specs),
        "encoder": _stratum_shared_encoder(specs),
        "encdec16": _stratum_enc_dec(specs, "fp16"),
        "encdec32": _stratum_enc_dec(specs, "fp32"),
        "eager": _stratum_eager(specs),
        "extension": _stratum_extension(specs),
        "spot": _stratum_spot(specs),
    }
    key_sets = {name: {ps.key() for ps in group} for name, group in strata.items()}
    names = list(key_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (key_sets[a] & key_sets[b]), f"strata overlap: {a} and {b}"
    union = set().union(*key_sets.values())
    assert union == {ps.key() for ps in specs}
    sizes = {name: len(g) for name, g in strata.items()}
    assert sizes == {"decoder": 48, "encoder": 12, "encdec16": 12,
                     "encdec32": 6, "eager": 6, "extension": 10, "spot": 4}


def test_shared_decoder_stratum_48():
    group = _stratum_shared_decoder(_specs())
    assert len(group) == 48
    assert {ps.model for ps in group} == _SHARED_DECODERS
    assert {ps.precision for ps in group} == {"fp16", "fp32"}
    assert sum(1 for ps in group if ps.phase == "prefill") == 24
    decode = [ps for ps in group if ps.phase == "decode"]
    assert len(decode) == 24
    for ps in decode:
        assert ps.tgt_ctx == ps.seq_len       # decode ctx = s (amendment 5a)
        assert ps.decode_tokens == 64 and ps.decode_mode == "growing"
    for ps in group:
        assert ps.batch_size == 1 and ps.weights == "random"


def test_shared_encoder_stratum_12():
    group = _stratum_shared_encoder(_specs())
    assert len(group) == 12
    assert {ps.model for ps in group} == _SHARED_ENCODERS
    assert all(ps.phase == "encode" for ps in group)
    assert {ps.seq_len for ps in group} == _SHARED_SEQS
    assert {ps.precision for ps in group} == {"fp16", "fp32"}


def _enc_dec_cell(ps):
    return (
        ps.model,
        ps.phase,
        ps.seq_len,
        ps.tgt_len if ps.phase == "decoder_prefill" else None,
        ps.tgt_ctx if ps.phase == "decode" else None,
    )


def test_enc_dec_fp16_both_arms_at_anchor():
    group = _stratum_enc_dec(_specs(), "fp16")
    assert len(group) == 12
    expected = set()
    for m in _ENC_DEC:
        expected |= {
            (m, "encode", 1024, None, None),
            (m, "decoder_prefill", 1024, 1024, None),
            (m, "decode", 128, None, 1024),    # source arm, tgt anchored
            (m, "decode", 2048, None, 1024),   # source arm, tgt anchored
            (m, "decode", 1024, None, 128),    # target arm, src anchored
            (m, "decode", 1024, None, 2048),   # target arm, src anchored
        }
    assert {_enc_dec_cell(ps) for ps in group} == expected


def test_enc_dec_fp32_anchor_cells_only():
    group = _stratum_enc_dec(_specs(), "fp32")
    assert len(group) == 6
    expected = set()
    for m in _ENC_DEC:
        expected |= {
            (m, "encode", 1024, None, None),
            (m, "decoder_prefill", 1024, 1024, None),
            (m, "decode", 1024, None, 1024),   # arms coincide; dedup to one
        }
    assert {_enc_dec_cell(ps) for ps in group} == expected


def test_eager_subset_6():
    group = _stratum_eager(_specs())
    assert len(group) == 6
    assert {ps.model for ps in group} == {"DistilGPT2", "GPT-2"}
    assert all(ps.phase == "prefill" and ps.precision == "fp16" for ps in group)
    assert {ps.seq_len for ps in group} == {512, 1024, 2048}


def test_extension_7b_stratum_10():
    group = _stratum_extension(_specs())
    assert len(group) == 10
    assert {ps.model for ps in group} == _EXTENSION
    assert all(ps.precision == "fp16" and ps.weights == "random"
               and ps.attn_kind == "flash" for ps in group)
    for m in _EXTENSION:
        cells = [ps for ps in group if ps.model == m]
        prefill = sorted(ps.seq_len for ps in cells if ps.phase == "prefill")
        decode = sorted(ps.seq_len for ps in cells if ps.phase == "decode")
        assert prefill == [1024, 4096, 8192]
        assert decode == [1024, 4096]
        for ps in cells:
            if ps.phase == "decode":
                assert ps.tgt_ctx == ps.seq_len
                assert ps.decode_tokens == 64 and ps.decode_mode == "growing"


def test_spot_cells_follow_up_b_arms():
    group = _stratum_spot(_specs())
    assert len(group) == 4
    assert all(ps.phase == "prefill" and ps.precision == "fp16"
               and ps.seq_len == 512 for ps in group)
    arms = {ps.weights: ps for ps in group}
    assert set(arms) == {"random", "random_v", "ported", "pretrained"}
    assert arms["ported"].pretrained_id == "gpt2"
    assert arms["pretrained"].pretrained_id == "gpt2"
    assert arms["random"].pretrained_id is None
    assert arms["random_v"].pretrained_id is None


# --- seeds and determinism ----------------------------------------------------

def test_every_point_has_explicit_derived_seed():
    specs = _specs()
    for ps in specs:
        assert isinstance(ps.seed, int)
        assert 0 <= ps.seed < 2 ** 31
        assert f"seed{ps.seed}" in ps.key()   # the seed joins the key
    assert len({ps.seed for ps in specs}) == len(specs)
    assert len({ps.key() for ps in specs}) == len(specs)


def test_expansion_is_deterministic():
    a = [ps.key() for ps in expand_passes(_cfg())]
    b = [ps.key() for ps in expand_passes(_cfg())]
    assert a == b


def test_seed_derivation_properties():
    assert derive_seed(42, "k") == derive_seed(42, "k")
    assert derive_seed(42, "k") != derive_seed(43, "k")
    assert derive_seed(42, "k1") != derive_seed(42, "k2")
    assert 0 <= derive_seed(42, "k") < 2 ** 31


# --- validation and the integrity guard --------------------------------------

def test_minimal_cfg_expands_to_one_seeded_point():
    specs = expand_passes(_minimal_cfg())
    assert len(specs) == 1
    assert specs[0].seed is not None


def test_expected_points_guard_raises_on_mismatch():
    with pytest.raises(ValueError, match="expected_points"):
        expand_passes(_minimal_cfg(expected_points=999))


def test_missing_passes_raises():
    with pytest.raises(ValueError, match="passes"):
        expand_passes({"seed": {"master": 42}})


def test_missing_seed_master_raises():
    cfg = _minimal_cfg()
    del cfg["seed"]
    with pytest.raises(ValueError, match="seed.master"):
        expand_passes(cfg)


def test_pass_without_config_raises():
    cfg = _minimal_cfg()
    del cfg["passes"][0]["config"]
    with pytest.raises(ValueError, match="config"):
        expand_passes(cfg)
