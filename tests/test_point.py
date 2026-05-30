"""Tests for single-point measurement (tomltransformers.sweep.point).

Pure tests monkeypatch measure_until_floor to return a synthetic PointResult, so
dispatch / key / per-unit-normalization / record-shaping are exercised on CPU with
no GPU and no torch workload execution. One GPU integration test measures a real
point end-to-end and checks the record is well-formed.
"""

import pytest

from tomltransformers.sweep import point as pt
from tomltransformers.sweep import PointSpec
from tomltransformers.measure import instruments as ins
from tomltransformers.measure.runner import PointResult


def _fake_result(means, *, wall=4.0, cv_exceeded=None, short=False, skip=None):
    """Build a PointResult with given per-instrument mean dynamic energy."""
    r = PointResult(label="fake", ok=(skip is None))
    r.n_repeats = 3
    r.skip_reason = skip
    for k, m in means.items():
        r.energy_j_dyn[k] = [m, m, m]
        r.summary[k] = {"mean": m, "std": 0.0, "cv": 0.0, "median": m, "n": 3}
    r.summary["wall_time_s"] = {"mean": wall, "std": 0.0, "cv": 0.0, "median": wall, "n": 3}
    r.instruments_available = sorted(means.keys())
    r.cv_exceeded = cv_exceeded or []
    r.short_window = short
    r.idle_power_w = 30.0
    r.temps_c = [50.0, 50.0, 50.0]
    r.clocks_mhz = [{"sm": 2000, "mem": 9000}] * 3
    r.thermal = {"settled": True, "temp_c": 50.0}
    return r


def _patch_measure(monkeypatch, result, inner=100):
    monkeypatch.setattr(pt.wl, "measure_until_floor",
                        lambda builder, measure_fn, **kw: (result, inner))


# --- key / identity -----------------------------------------------------------


def test_key_includes_phase_specific_fields():
    dec = PointSpec(model="GPT-2", arch="decoder_only", phase="decode",
                    seq_len=512, tgt_ctx=128, decode_tokens=32, decode_mode="growing")
    k = dec.key()
    assert "decode" in k and "ctx128" in k and "k32" in k and "growing" in k

    pre = PointSpec(model="GPT-2", arch="decoder_only", phase="prefill", seq_len=512)
    assert "prefill" in pre.key() and "ctx" not in pre.key()

    edp = PointSpec(model="T5-base", arch="encoder_decoder", phase="decoder_prefill",
                    seq_len=512, tgt_len=128)
    assert "decoder_prefill" in edp.key() and "tgt128" in edp.key()


# --- per-unit normalization (the divisor + contamination flag) ----------------


def test_forward_phase_per_unit_divides_by_inner_iters(monkeypatch):
    res = _fake_result({"B": 1000.0, "A": 900.0, "C": 1000.0})
    _patch_measure(monkeypatch, res, inner=100)
    ps = PointSpec(model="GPT-2", arch="decoder_only", phase="prefill", seq_len=512)
    rec = pt.measure_single_point(ps)
    # per-forward = per-execution / inner_iters.
    assert rec["per_unit_kind"] == "forward"
    assert rec["per_unit_contaminated"] is False
    assert rec["per_unit_j"]["B"] == pytest.approx(1000.0 / 100)
    assert rec["per_execution_j"]["B"] == pytest.approx(1000.0)


def test_decode_growing_per_unit_divides_by_inner_times_tokens_and_flags(monkeypatch):
    res = _fake_result({"B": 1600.0, "A": 1400.0, "C": 1600.0}, cv_exceeded=["A", "B", "C"])
    _patch_measure(monkeypatch, res, inner=40)
    ps = PointSpec(model="GPT-2", arch="decoder_only", phase="decode",
                   seq_len=256, tgt_ctx=256, decode_tokens=32, decode_mode="growing")
    rec = pt.measure_single_point(ps)
    assert rec["per_unit_kind"] == "decode_token"
    # divisor = inner_iters * decode_tokens = 40 * 32.
    assert rec["per_unit_j"]["B"] == pytest.approx(1600.0 / (40 * 32))
    # MUST be flagged prefill-contaminated.
    assert rec["per_unit_contaminated"] is True
    assert rec["cv_exceeded"] == ["A", "B", "C"]


def test_decode_fixed_step_divides_by_inner_only_and_flags(monkeypatch):
    res = _fake_result({"B": 800.0})
    _patch_measure(monkeypatch, res, inner=50)
    ps = PointSpec(model="GPT-2", arch="decoder_only", phase="decode",
                   seq_len=256, tgt_ctx=256, decode_tokens=32, decode_mode="fixed_step")
    rec = pt.measure_single_point(ps)
    assert rec["per_unit_kind"] == "decode_step"
    assert rec["per_unit_j"]["B"] == pytest.approx(800.0 / 50)
    assert rec["per_unit_contaminated"] is True


# --- record shape -------------------------------------------------------------


def test_record_has_schema_provenance_and_primary(monkeypatch):
    res = _fake_result({"B": 500.0, "A": 460.0, "C": 500.0})
    _patch_measure(monkeypatch, res, inner=200)
    ps = PointSpec(model="BERT-base", arch="encoder_only", phase="encode", seq_len=512)
    rec = pt.measure_single_point(ps)
    assert rec["schema"] == "tomltransformers.sweep.point.v1"
    assert "timestamp" in rec and "git_commit" in rec
    assert rec["primary"] == "B"
    assert rec["ok"] is True
    assert rec["inner_iters"] == 200
    # telemetry carried through.
    assert rec["idle_power_w"] == 30.0
    assert rec["thermal"]["settled"] is True
    assert rec["spec"]["model"] == "BERT-base" and rec["spec"]["phase"] == "encode"


def test_short_window_flag_propagates(monkeypatch):
    res = _fake_result({"B": 10.0}, wall=1.2, short=True)
    _patch_measure(monkeypatch, res, inner=5)
    ps = PointSpec(model="GPT-2", arch="decoder_only", phase="prefill", seq_len=128)
    rec = pt.measure_single_point(ps)
    assert rec["short_window"] is True
    assert rec["wall_time_s_median"] == pytest.approx(1.2)


def test_oom_skip_recorded_not_raised(monkeypatch):
    res = _fake_result({}, skip="OOM")
    _patch_measure(monkeypatch, res, inner=1)
    ps = PointSpec(model="GPT-2-XL", arch="decoder_only", phase="prefill", seq_len=2048)
    rec = pt.measure_single_point(ps)
    assert rec["ok"] is False
    assert rec["oom_skipped"] is True
    assert rec["skip_reason"] == "OOM"


def test_exception_in_build_recorded_not_raised(monkeypatch):
    def boom(builder, measure_fn, **kw):
        raise RuntimeError("CUDA error: out of memory")
    monkeypatch.setattr(pt.wl, "measure_until_floor", boom)
    ps = PointSpec(model="GPT-2", arch="decoder_only", phase="prefill", seq_len=512)
    rec = pt.measure_single_point(ps)
    assert rec["ok"] is False
    assert rec["oom_skipped"] is True
    assert "out of memory" in rec["error"]


# --- dispatch validation ------------------------------------------------------


def test_bad_phase_for_arch_is_recorded_as_error(monkeypatch):
    # encoder_only with a decode phase: _build_workload raises ValueError, which
    # surfaces through measure_until_floor's call to builder -> recorded, not raised.
    def call_builder(builder, measure_fn, **kw):
        builder(1)   # trigger the dispatch error
        return _fake_result({"B": 1.0}), 1
    monkeypatch.setattr(pt.wl, "measure_until_floor", call_builder)
    ps = PointSpec(model="BERT-base", arch="encoder_only", phase="decode", seq_len=512)
    rec = pt.measure_single_point(ps)
    assert rec["ok"] is False
    assert "encode" in rec["error"]


def test_unknown_arch_recorded(monkeypatch):
    def call_builder(builder, measure_fn, **kw):
        builder(1)
        return _fake_result({"B": 1.0}), 1
    monkeypatch.setattr(pt.wl, "measure_until_floor", call_builder)
    ps = PointSpec(model="x", arch="quantum_net", phase="prefill", seq_len=8)
    rec = pt.measure_single_point(ps)
    assert rec["ok"] is False
    assert "unknown arch" in rec["error"]


# --- GPU integration ----------------------------------------------------------


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_single_point_end_to_end():
    """Measure a real decoder prefill point end-to-end; check the record is sane."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # A small-but-real model id so this is a true GPU measurement without download:
    # use a registered config name from the zoo (random weights).
    ps = PointSpec(model="DistilGPT2", arch="decoder_only", phase="prefill",
                   seq_len=256, precision="fp16")
    rec = pt.measure_single_point(ps, target_s=4.0, repeats=3, warmup_iters=5)
    print(f"\n[single-point] ok={rec['ok']} inner={rec.get('inner_iters')} "
          f"per_exec_B={rec.get('per_execution_j', {}).get('B')} "
          f"per_unit_B={rec.get('per_unit_j', {}).get('B')} "
          f"short={rec.get('short_window')} agree={rec.get('agreement')}")
    assert rec["ok"] is True
    assert rec["primary"] == "B"
    assert rec["per_execution_j"]["B"] > 0.0
    assert rec["per_unit_j"]["B"] > 0.0
    assert rec["per_unit_kind"] == "forward"
    assert rec["git_commit"] is not None
