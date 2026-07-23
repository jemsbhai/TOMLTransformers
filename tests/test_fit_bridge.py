"""Bridge test gates (fit_plan.md section 3, plus D1' launch invariants).

Gate (a): emitted keys match energy_model.FEATURES exactly.
Gate (b): hand-checked invariants per architecture class, each grounded in
          the documented counting conventions of architectures/common.py and
          architectures/attention.py (see fit_plan.md section 11, D1').
Gate (c): every latest record in the frozen EXP-002 dataset resolves.

All CPU-only; no torch required.
"""

from pathlib import Path

import pytest

from tomltransformers import to_costs as tc
from tomltransformers.architectures import decoder as dec
from tomltransformers.architectures import encoder_decoder as ed
from tomltransformers.architectures.configs import get as get_config
from tomltransformers.energy_model import FEATURES
from tomltransformers.fit.bridge import (BridgeError, PRECISIONS,
                                         features_for_record,
                                         features_for_spec,
                                         load_latest_records)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "experiments" / "exp_002_size_sweep" / "energy.jsonl"

TO_FEATURES = ("to_mac", "to_nonlinear", "to_sram", "to_hbm")


def _spec(model, arch, phase, **kw):
    base = {
        "model": model, "arch": arch, "phase": phase,
        "precision": "fp16", "attn_kind": "flash", "weights": "random",
        "seq_len": None, "tgt_len": 1, "tgt_ctx": 128,
        "decode_tokens": 64, "decode_mode": "growing",
        "batch_size": 1, "key": "test",
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------------------
# Gate (a): key set
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("spec", [
    _spec("GPT-2", "decoder_only", "prefill", seq_len=128),
    _spec("BERT-base", "encoder_only", "encode", seq_len=128),
    _spec("T5-small", "encoder_decoder", "decoder_prefill", seq_len=256, tgt_len=64),
], ids=["decoder", "encoder", "enc_dec"])
def test_keys_match_energy_model_features(spec):
    assert set(features_for_spec(spec)) == set(FEATURES)


# ------------------------------------------------------------------------------
# Gate (b): hand-checked invariants
# ------------------------------------------------------------------------------
def test_decoder_prefill_launches_sequence_independent_tos_grow():
    a = features_for_spec(_spec("GPT-2", "decoder_only", "prefill", seq_len=512))
    b = features_for_spec(_spec("GPT-2", "decoder_only", "prefill", seq_len=1024))
    # Launch counts are structural (per-GEMM), not sequence-dependent.
    assert a["n_launches"] == b["n_launches"] > 0
    # Every TO feature strictly grows with sequence length.
    for f in TO_FEATURES:
        assert b[f] > a[f], f


def test_decoder_decode_additivity_and_launch_ratio():
    """decode(s, K) == prefill(s) + sum_t decode_step(s+t+1), and the launch
    count is exactly 65x a single forward (decode-step layers have the same
    launch structure as prefill layers; KV-cache ops count zero launches)."""
    cfg = get_config("GPT-2")
    prec = PRECISIONS["fp16"]
    got = features_for_spec(_spec("GPT-2", "decoder_only", "decode", seq_len=512))

    pf = dec.prefill(cfg, 512, device="rtx4090", prec=prec, attn_kind="flash").breakdown
    acc = pf
    for t in range(64):
        acc = acc + dec.decode_step(cfg, 512 + t + 1, device="rtx4090", prec=prec,
                                    attn_kind="flash").breakdown
    want = acc.as_record()
    for f in FEATURES:
        assert got[f] == pytest.approx(want[f], rel=1e-9), f

    pf_launches = pf.as_record()["n_launches"]
    assert got["n_launches"] == 65 * pf_launches


def test_precision_exact_ratios_decoder_prefill():
    """fp32 vs fp16 at fixed shape: memory words double exactly; MAC cost
    scales by the to_costs multiplier; nonlinear op costs and launch counts
    are precision-independent."""
    f16 = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                  seq_len=512, precision="fp16"))
    f32 = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                  seq_len=512, precision="fp32"))
    word_ratio = tc.words_per_element("fp32") / tc.words_per_element("fp16")
    assert word_ratio == 2.0
    assert f32["to_hbm"] / f16["to_hbm"] == pytest.approx(word_ratio, rel=1e-12)
    assert f32["to_sram"] / f16["to_sram"] == pytest.approx(word_ratio, rel=1e-12)
    assert f32["to_mac"] / f16["to_mac"] == pytest.approx(
        tc.mac("fp32") / tc.mac("fp16"), rel=1e-12)
    assert f32["to_nonlinear"] == pytest.approx(f16["to_nonlinear"], rel=1e-12)
    assert f32["n_launches"] == f16["n_launches"]


def test_eager_vs_flash_gpt2():
    """Same arithmetic; eager (standard) adds off-chip score traffic and two
    extra kernel launches per layer (3 vs 1)."""
    cfg = get_config("GPT-2")
    fl = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                 seq_len=2048, attn_kind="flash"))
    eg = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                 seq_len=2048, attn_kind="eager"))
    assert eg["to_mac"] == pytest.approx(fl["to_mac"], rel=1e-12)
    assert eg["to_nonlinear"] == pytest.approx(fl["to_nonlinear"], rel=1e-12)
    assert eg["to_sram"] == pytest.approx(fl["to_sram"], rel=1e-12)
    assert eg["to_hbm"] > fl["to_hbm"]
    assert eg["n_launches"] - fl["n_launches"] == 2 * cfg.n_layers


def test_vit_ignores_seq_len():
    a = features_for_spec(_spec("ViT-B/16", "encoder_only", "encode", seq_len=128))
    b = features_for_spec(_spec("ViT-B/16", "encoder_only", "encode", seq_len=2048))
    assert a == b


def test_enc_dec_decode_additivity():
    """enc-dec decode(src, ctx, K) == decoder_prefill(src, ctx) +
    sum_t decode_step(src, ctx+t+1); cross-cache build lives inside the
    decoder_prefill component, matching the measured execution."""
    cfg = get_config("T5-small")
    prec = PRECISIONS["fp16"]
    got = features_for_spec(_spec("T5-small", "encoder_decoder", "decode",
                                  seq_len=1024, tgt_ctx=128))
    acc = ed.decoder_prefill(cfg, 1024, 128, device="rtx4090", prec=prec,
                             attn_kind="flash")
    for t in range(64):
        acc = acc + ed.decode_step(cfg, 1024, 128 + t + 1, device="rtx4090",
                                   prec=prec, attn_kind="flash")
    want = acc.as_record()
    for f in FEATURES:
        assert got[f] == pytest.approx(want[f], rel=1e-9), f


def test_enc_dec_cross_cache_scales_with_source():
    """The static cross-cache read makes decode off-chip traffic grow with
    the SOURCE length at fixed target context."""
    small = features_for_spec(_spec("T5-small", "encoder_decoder", "decode",
                                    seq_len=512, tgt_ctx=128))
    large = features_for_spec(_spec("T5-small", "encoder_decoder", "decode",
                                    seq_len=2048, tgt_ctx=128))
    assert large["to_hbm"] > small["to_hbm"]


def test_n_fused_steps_zero_everywhere():
    for spec in (
        _spec("DistilGPT2", "decoder_only", "decode", seq_len=128),
        _spec("BERT-large", "encoder_only", "encode", seq_len=512),
        _spec("BART-base", "encoder_decoder", "encode", seq_len=256),
    ):
        assert features_for_spec(spec)["n_fused_steps"] == 0.0


# ------------------------------------------------------------------------------
# Failure paths: fail loudly, never silently
# ------------------------------------------------------------------------------
def test_fixed_step_mode_rejected():
    with pytest.raises(BridgeError, match="growing"):
        features_for_spec(_spec("GPT-2", "decoder_only", "decode",
                                seq_len=512, decode_mode="fixed_step"))


def test_unknown_precision_rejected():
    with pytest.raises(BridgeError, match="precision"):
        features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                seq_len=512, precision="int8"))


def test_text_encoder_requires_seq_len():
    with pytest.raises(BridgeError, match="seq_len"):
        features_for_spec(_spec("BERT-base", "encoder_only", "encode",
                                seq_len=None))


def test_wrong_phase_for_arch_rejected():
    with pytest.raises(BridgeError, match="phase"):
        features_for_spec(_spec("GPT-2", "decoder_only", "encode", seq_len=512))


# ------------------------------------------------------------------------------
# Gate (c): the entire frozen dataset resolves
# ------------------------------------------------------------------------------
def test_all_frozen_records_resolve():
    records = load_latest_records(DATA)
    assert len(records) == 296
    combos = set()
    for rec in records:
        feats = features_for_record(rec)          # raises BridgeError on any failure
        assert set(feats) == set(FEATURES)
        combos.add((rec["spec"]["arch"], rec["spec"]["phase"]))
    assert combos == {
        ("decoder_only", "prefill"), ("decoder_only", "decode"),
        ("encoder_only", "encode"),
        ("encoder_decoder", "encode"), ("encoder_decoder", "decoder_prefill"),
        ("encoder_decoder", "decode"),
    }
