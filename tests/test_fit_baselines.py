"""Baseline and stats tests (fit_plan sections 8 and 12).

Invariants are grounded in the counting conventions and to_costs constants;
synthetic-recovery tests pin the estimators; the registry test locks the
2026-07-24 device correction (RTX 4090 Laptop GPU = GDDR6).
"""

import numpy as np
import pytest

from tomltransformers import to_costs as tc
from tomltransformers.fit.baselines import (BANDWIDTH_BYTES_S,
                                            PEAK_BY_PRECISION,
                                            PEAK_FP32_FLOPS_S,
                                            LayerwiseBaseline,
                                            RooflineBaseline, raw_counts,
                                            roofline_time_s)
from tomltransformers.fit.bridge import features_for_spec
from tomltransformers.fit.stats import ape_pct, holm_adjust, wilcoxon_less


def _spec(model, arch, phase, **kw):
    base = {"model": model, "arch": arch, "phase": phase,
            "precision": "fp16", "attn_kind": "flash", "weights": "random",
            "seq_len": None, "tgt_len": 1, "tgt_ctx": 128,
            "decode_tokens": 64, "decode_mode": "growing",
            "batch_size": 1, "key": "test"}
    base.update(kw)
    return base


def test_registry_laptop_gpu_is_gddr6():
    assert tc.offchip_tier("rtx4090") == "gddr6"
    assert tc.offchip_tier("rtx4090_desktop") == "gddr6x"


def test_raw_counts_precision_invariants():
    f16 = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                  seq_len=512, precision="fp16"))
    f32 = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                  seq_len=512, precision="fp32"))
    r16 = raw_counts(f16, "fp16")
    r32 = raw_counts(f32, "fp32")
    # Op counts are structural: identical across precision.
    assert r32["raw_macs"] == pytest.approx(r16["raw_macs"], rel=1e-9)
    assert r16["flops"] == pytest.approx(2.0 * r16["raw_macs"], rel=1e-12)
    # Word traffic doubles at fp32 exactly (2 words vs 1 per element).
    assert r32["hbm_words"] / r16["hbm_words"] == pytest.approx(2.0, rel=1e-9)
    assert r32["sram_words"] / r16["sram_words"] == pytest.approx(2.0, rel=1e-9)
    assert r16["hbm_bytes"] == pytest.approx(4.0 * r16["hbm_words"], rel=1e-12)


def test_roofline_time_positive_and_monotone_in_s():
    a = features_for_spec(_spec("GPT-2", "decoder_only", "prefill", seq_len=512))
    b = features_for_spec(_spec("GPT-2", "decoder_only", "prefill", seq_len=2048))
    ta = roofline_time_s(a, "fp16")
    tb = roofline_time_s(b, "fp16")
    assert 0 < ta < tb


def test_roofline_constants():
    assert PEAK_FP32_FLOPS_S == pytest.approx(2.0 * 9728 * 2.040e9, rel=1e-12)
    assert PEAK_BY_PRECISION["fp16"] == pytest.approx(2.0 * PEAK_FP32_FLOPS_S,
                                                      rel=1e-12)
    assert BANDWIDTH_BYTES_S == pytest.approx(576.0e9, rel=1e-12)


def test_roofline_fit_recovers_p_avg_absolute_and_relative():
    feats = [features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                     seq_len=s)) for s in (128, 256, 512, 1024)]
    precs = ["fp16"] * len(feats)
    p_true = 87.5
    base = RooflineBaseline()
    ys = p_true * base.times(feats, precs)
    for relative in (False, True):
        m = RooflineBaseline().fit(feats, precs, ys, relative=relative)
        assert m.p_avg_w_ == pytest.approx(p_true, rel=1e-9)
        pred = m.predict(feats, precs)
        assert np.allclose(pred, ys, rtol=1e-9)


def test_layerwise_fit_recovers_synthetic_coefficients():
    specs = [
        _spec("GPT-2", "decoder_only", "prefill", seq_len=s)
        for s in (128, 256, 512, 1024, 2048)
    ] + [
        _spec("GPT-2", "decoder_only", "decode", seq_len=s)
        for s in (128, 512, 1024)
    ] + [
        _spec("BERT-base", "encoder_only", "encode", seq_len=s)
        for s in (128, 512)
    ]
    feats = [features_for_spec(s) for s in specs]
    precs = [s["precision"] for s in specs]
    lb = LayerwiseBaseline()
    A = lb.design(feats, precs)
    coef_true = np.array([3e-12, 2e-10, 5e-9, 1e-4, 0.02])
    ys = A @ coef_true
    m = LayerwiseBaseline().fit(feats, precs, ys)
    pred = m.predict(feats, precs)
    assert np.allclose(pred, ys, rtol=1e-6)


def test_wilcoxon_less_detects_uniform_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(10, 30, size=20)
    winner = base - rng.uniform(1, 5, size=20)
    p = wilcoxon_less(winner, base)
    assert p < 0.05
    assert wilcoxon_less(base, base) == 1.0


def test_holm_adjust_hand_example():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06], rel=1e-12)
    assert holm_adjust([0.2]) == pytest.approx([0.2])


def test_ape_pct():
    out = ape_pct([2.0, 4.0], [1.0, 5.0])
    assert out == pytest.approx([50.0, 25.0])
