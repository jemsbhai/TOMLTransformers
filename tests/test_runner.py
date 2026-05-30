"""Tests for the controlled measurement runner (tomltransformers.measure.runner).

The control logic (idle subtraction, CV gating, OOM handling, summaries,
agreement) is tested on CPU by injecting a fake measure_once, so these run
anywhere and deterministically. One GPU integration test exercises the real
path and auto-skips without NVML/CUDA.
"""

import pytest

from tomltransformers.measure import runner as rn
from tomltransformers.measure.instruments import MeasurementWindow
from tomltransformers.measure import instruments as ins


# --- fixtures that neutralize hardware + timing for fast, deterministic logic --


@pytest.fixture
def no_hardware(monkeypatch):
    """Stub out telemetry and settling so logic tests need no GPU and no waiting."""
    monkeypatch.setattr(rn, "_read_temp_c", lambda idx: 50.0)
    monkeypatch.setattr(rn, "_read_clocks_mhz", lambda idx: {"sm": 2100.0, "mem": 9000.0})
    monkeypatch.setattr(rn, "wait_for_thermal_settle",
                        lambda *a, **k: {"settled": True, "temp_c": 50.0})
    # Default idle power; individual tests can override.
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 10.0)


def _fake_window_factory(energy_by_call, wall_time=1.0):
    """Return a measure_once stand-in that yields preset energies per call.

    energy_by_call: list of dicts, one per measured repeat, e.g.
        [{"A": 60.0, "B": 50.0}, {"A": 61.0, "B": 50.5}, ...]
    Warmup calls (fn invoked directly, not via measure_once) are unaffected.
    """
    state = {"i": 0}

    def fake_measure_once(fn, **kwargs):
        fn()  # honor the call so warmup/exec side effects still happen
        e = energy_by_call[state["i"] % len(energy_by_call)]
        state["i"] += 1
        w = MeasurementWindow(wall_time_s=wall_time)
        w.energy_j = dict(e)
        w.available = list(e.keys())
        return w

    return fake_measure_once


# --- idle subtraction ---------------------------------------------------------


def test_idle_subtraction_is_uniform_across_instruments(no_hardware, monkeypatch):
    # idle 10 W * 1.0 s = 10 J subtracted from every instrument's raw energy.
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 10.0)
    monkeypatch.setattr(rn, "measure_once",
                        _fake_window_factory([{"A": 60.0, "B": 50.0, "C": 50.0}], wall_time=1.0))

    res = rn.measure_point(lambda: None, "t", repeats=1, warmup_iters=0, thermal_settle=False)
    assert res.ok
    assert res.idle_power_w == 10.0
    assert res.energy_j_raw["A"] == [60.0]
    assert res.energy_j_dyn["A"] == [50.0]      # 60 - 10
    assert res.energy_j_dyn["B"] == [40.0]      # 50 - 10
    assert res.energy_j_dyn["C"] == [40.0]      # 50 - 10


def test_negative_dynamic_energy_is_clamped(no_hardware, monkeypatch):
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 100.0)  # 100 J idle
    monkeypatch.setattr(rn, "measure_once",
                        _fake_window_factory([{"A": 5.0}], wall_time=1.0))
    res = rn.measure_point(lambda: None, "t", repeats=1, warmup_iters=0, thermal_settle=False)
    assert res.energy_j_dyn["A"] == [0.0]
    assert any("clamped to 0" in n for n in res.notes)


def test_missing_idle_power_means_no_subtraction(no_hardware, monkeypatch):
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: None)
    monkeypatch.setattr(rn, "measure_once",
                        _fake_window_factory([{"A": 60.0}], wall_time=1.0))
    res = rn.measure_point(lambda: None, "t", repeats=1, warmup_iters=0, thermal_settle=False)
    assert res.idle_power_w is None
    assert res.energy_j_dyn["A"] == [60.0]      # raw, no subtraction


# --- repeats, summary stats, CV gate ------------------------------------------


def test_summary_and_cv_gate(no_hardware, monkeypatch):
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 0.0)
    # A is stable (low CV), B is noisy (high CV).
    monkeypatch.setattr(rn, "measure_once", _fake_window_factory([
        {"A": 100.0, "B": 50.0},
        {"A": 101.0, "B": 80.0},
        {"A": 99.0, "B": 20.0},
    ], wall_time=1.0))
    res = rn.measure_point(lambda: None, "t", repeats=3, warmup_iters=0,
                           thermal_settle=False, cv_threshold=0.05)
    assert res.n_repeats == 3
    assert res.summary["A"]["n"] == 3
    assert abs(res.summary["A"]["mean"] - 100.0) < 1e-9
    assert "A" not in res.cv_exceeded     # ~1% CV, under gate
    assert "B" in res.cv_exceeded         # huge CV, over gate


def test_counts_repeats_and_warmup_separately(no_hardware, monkeypatch):
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 0.0)
    calls = {"n": 0}

    def work():
        calls["n"] += 1

    monkeypatch.setattr(rn, "measure_once", _fake_window_factory([{"A": 1.0}], wall_time=1.0))
    res = rn.measure_point(work, "t", repeats=5, warmup_iters=10, thermal_settle=False)
    # 10 warmup (direct) + 5 measured (each calls work() once via fake) = 15.
    assert calls["n"] == 15
    assert res.n_repeats == 5


# --- OOM handling -------------------------------------------------------------


def test_oom_during_warmup_is_skipped(no_hardware):
    def work():
        raise RuntimeError("CUDA out of memory. Tried to allocate ...")

    res = rn.measure_point(work, "big", repeats=5, warmup_iters=3, thermal_settle=False)
    assert not res.ok
    assert res.skip_reason == "OOM"
    assert res.n_repeats == 0


def test_oom_during_measurement_is_skipped(no_hardware, monkeypatch):
    def boom(fn, **kwargs):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 0.0)
    monkeypatch.setattr(rn, "measure_once", boom)
    res = rn.measure_point(lambda: None, "big", repeats=5, warmup_iters=0, thermal_settle=False)
    assert res.skip_reason == "OOM"
    assert not res.ok


def test_non_oom_error_propagates(no_hardware, monkeypatch):
    def boom(fn, **kwargs):
        raise ValueError("a real bug, not OOM")
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 0.0)
    monkeypatch.setattr(rn, "measure_once", boom)
    with pytest.raises(ValueError):
        rn.measure_point(lambda: None, "t", repeats=1, warmup_iters=0, thermal_settle=False)


# --- agreement ----------------------------------------------------------------


def test_pairwise_agreement_on_means(no_hardware, monkeypatch):
    monkeypatch.setattr(rn, "_read_idle_power_w", lambda idx, sec, hz=20.0: 0.0)
    monkeypatch.setattr(rn, "measure_once",
                        _fake_window_factory([{"A": 110.0, "B": 100.0}], wall_time=1.0))
    res = rn.measure_point(lambda: None, "t", repeats=1, warmup_iters=0, thermal_settle=False)
    ag = rn.pairwise_agreement(res)
    assert abs(ag["A-B"] - 0.10) < 1e-9       # |110-100|/100


# --- GPU integration (auto-skip without NVML/CUDA) ----------------------------


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_measure_point_end_to_end():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float32)

    # The measured window MUST be long enough for 20 Hz power sampling to collect
    # enough samples to integrate (instrument A). A few hundred ms is too short
    # (~2 samples -> A reads 0). ~600 matmuls is ~1-2 s on a 4090: dozens of
    # samples, and long enough that the energy counter tick is negligible.
    def work():
        c = b
        for _ in range(600):
            c = a @ c
        return c

    res = rn.measure_point(
        work, "gpu-smoke", repeats=3, warmup_iters=10,
        idle_baseline_s=2.0, thermal_settle=True, thermal_window_s=2.0,
    )
    print(f"\n[runner] available={res.instruments_available} "
          f"summary={ {k: round(v['mean'], 3) for k, v in res.summary.items()} } "
          f"idle_W={res.idle_power_w} wall_s={res.summary.get('wall_time_s', {}).get('mean')} "
          f"cv_exceeded={res.cv_exceeded} agreement={rn.pairwise_agreement(res)}")
    assert res.ok
    assert res.n_repeats == 3
    assert "B" in res.instruments_available
    assert res.summary["B"]["mean"] > 0.0
    # A must actually have integrated something on a window this long; A=0 here
    # would mean the sampler is broken, not just a short window.
    assert "A" in res.instruments_available
    assert res.summary["A"]["mean"] > 0.0
    # On a long, thermally-settled, idle-subtracted window, our two instruments
    # should agree far better than the uncontrolled 27% smoke bound. This is the
    # first real check that the controls work. Loose gate (15%) to avoid
    # flakiness; the pre-registered target is 5% median across the full grid.
    ag = rn.pairwise_agreement(res)
    assert ag["A-B"] < 0.15, f"A-vs-B {ag['A-B']:.1%} on a controlled window; notes={res.notes}"
