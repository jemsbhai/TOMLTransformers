"""A100 roofline constants and device-invariance tests (a100_amendment.md
section 10, decision D7; constants added 2026-08-17 before any A100 baseline
is fitted).

Locks: the A100 constants reproduce the vendor figures by the stated
formulas; the rtx4090 default set is untouched; raw structural counts
recovered through raw_counts are device-invariant (only the off-chip prior
differs between registry entries); and the A100 roofline encodes the 16x
fp32/fp16 datapath gap on a compute-bound cell.
"""

import numpy as np
import pytest

from tomltransformers import to_costs as tc
from tomltransformers.fit.baselines import (A100_BANDWIDTH_BYTES_S,
                                            A100_CORES,
                                            A100_PEAK_BY_PRECISION,
                                            A100_PEAK_FP16_FLOPS_S,
                                            A100_PEAK_FP32_FLOPS_S,
                                            BANDWIDTH_BYTES_S,
                                            PEAK_BY_PRECISION,
                                            TDP_W_BY_DEVICE,
                                            RooflineBaseline, raw_counts,
                                            roofline_constants)
from tomltransformers.fit.bridge import features_for_spec


def _spec(model, arch, phase, **kw):
    base = {"model": model, "arch": arch, "phase": phase,
            "precision": "fp16", "attn_kind": "flash", "weights": "random",
            "seq_len": None, "tgt_len": 1, "tgt_ctx": 128,
            "decode_tokens": 64, "decode_mode": "growing",
            "batch_size": 1, "key": "test"}
    base.update(kw)
    return base


def test_a100_constants_by_the_stated_formulas():
    assert A100_CORES == 108 * 64 == 6912
    assert A100_PEAK_FP32_FLOPS_S == pytest.approx(2.0 * 6912 * 1.410e9, rel=1e-12)
    assert A100_PEAK_FP32_FLOPS_S == pytest.approx(19.5e12, rel=2e-3)     # datasheet
    assert A100_PEAK_FP16_FLOPS_S == pytest.approx(2.0 * 108 * 1024 * 1.410e9, rel=1e-12)
    assert A100_PEAK_FP16_FLOPS_S == pytest.approx(312e12, rel=2e-3)      # datasheet
    assert A100_PEAK_FP16_FLOPS_S / A100_PEAK_FP32_FLOPS_S == pytest.approx(16.0, rel=1e-12)
    assert A100_BANDWIDTH_BYTES_S == pytest.approx(1555.0e9, rel=1e-12)
    assert A100_PEAK_BY_PRECISION == {"fp32": A100_PEAK_FP32_FLOPS_S,
                                      "fp16": A100_PEAK_FP16_FLOPS_S}
    assert TDP_W_BY_DEVICE == {"rtx4090": 150.0, "a100": 400.0}


def test_roofline_constants_lookup_and_4090_default_untouched():
    a = roofline_constants("a100")
    assert a["peak_by_precision"] == A100_PEAK_BY_PRECISION
    assert a["bandwidth_bytes_s"] == A100_BANDWIDTH_BYTES_S
    r = roofline_constants("rtx4090")
    assert r["peak_by_precision"] == PEAK_BY_PRECISION
    assert r["bandwidth_bytes_s"] == BANDWIDTH_BYTES_S
    assert PEAK_BY_PRECISION["fp32"] == pytest.approx(2.0 * 9728 * 2.040e9, rel=1e-12)
    assert BANDWIDTH_BYTES_S == pytest.approx(576.0e9, rel=1e-12)
    with pytest.raises(KeyError):
        roofline_constants("h100")
    # returned dicts are copies: mutating them cannot alter module state
    a["peak_by_precision"]["fp16"] = 1.0
    assert roofline_constants("a100")["peak_by_precision"] == A100_PEAK_BY_PRECISION


def test_raw_counts_are_device_invariant():
    # The same spec featurized under either device registry entry must invert
    # to identical structural counts; only the off-chip prior differs.
    spec = _spec("GPT-2", "decoder_only", "decode", seq_len=1024, precision="fp32")
    f40 = features_for_spec(spec, device="rtx4090")
    fa1 = features_for_spec(spec, device="a100")
    r40 = raw_counts(f40, "fp32", "rtx4090")
    ra1 = raw_counts(fa1, "fp32", "a100")
    for k in ("raw_macs", "flops", "hbm_words", "hbm_bytes", "sram_words"):
        assert ra1[k] == pytest.approx(r40[k], rel=1e-9), k
    assert fa1["to_hbm"] / f40["to_hbm"] == pytest.approx(
        tc.mem_word("hbm2e") / tc.mem_word("gddr6"), rel=1e-9)
    assert fa1["to_mac"] == pytest.approx(f40["to_mac"], rel=1e-12)


def test_a100_roofline_encodes_the_16x_datapath_gap_when_compute_bound():
    f16 = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                  seq_len=2048, precision="fp16"), device="a100")
    f32 = features_for_spec(_spec("GPT-2", "decoder_only", "prefill",
                                  seq_len=2048, precision="fp32"), device="a100")
    base = RooflineBaseline(**roofline_constants("a100"), device="a100")
    t16, t32 = base.times([f16, f32], ["fp16", "fp32"])
    # both precisions compute-bound at s=2048 for GPT-2 on the A100 constants
    rc16 = raw_counts(f16, "fp16", "a100")
    rc32 = raw_counts(f32, "fp32", "a100")
    assert rc16["flops"] / A100_PEAK_FP16_FLOPS_S > rc16["hbm_bytes"] / A100_BANDWIDTH_BYTES_S
    assert rc32["flops"] / A100_PEAK_FP32_FLOPS_S > rc32["hbm_bytes"] / A100_BANDWIDTH_BYTES_S
    assert t32 / t16 == pytest.approx(16.0, rel=1e-9)


def test_a100_roofline_fit_recovers_p_avg():
    feats = [features_for_spec(_spec("BERT-base", "encoder_only", "encode",
                                     seq_len=s), device="a100")
             for s in (128, 1024, 2048)]
    precs = ["fp16"] * len(feats)
    p_true = 250.0
    ys = p_true * RooflineBaseline(**roofline_constants("a100"),
                                   device="a100").times(feats, precs)
    for relative in (False, True):
        m = RooflineBaseline(**roofline_constants("a100"), device="a100").fit(
            feats, precs, ys, relative=relative)
        assert m.p_avg_w_ == pytest.approx(p_true, rel=1e-9)
        assert np.allclose(m.predict(feats, precs), ys, rtol=1e-9)
