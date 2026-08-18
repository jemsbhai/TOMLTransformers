"""Tests for the energy-measurement instrument layer (tomltransformers.measure).

The pure-logic tests run anywhere. The GPU tests are skipped automatically when
no NVML/CUDA is present. The Zeus test is designed to PASS whether or not Zeus
works on this machine, because instrument C is an optional cross-check that is
never required; it prints the verdict (run pytest with -s to see it).
"""

import time

import pytest

from tomltransformers.measure import instruments as ins

# The frozen sweep protocol's minimum measurement window (measure_until_floor).
# Agreement between instruments is only meaningful at or above this floor; see
# test_gpu_smoke_A_and_B_agree_roughly and findings.md 2026-08-18.
WINDOW_FLOOR_S = 4.0
AB_SMOKE_BAND = 0.25


# --- pure-logic tests (no GPU needed) -----------------------------------------


def test_availability_probes_return_bool():
    assert isinstance(ins.nvml_available(), bool)
    assert isinstance(ins.zeus_available(), bool)
    assert isinstance(ins.energy_counter_supported(), bool)


def test_measure_once_runs_callable_and_times_it():
    flag = {"ran": False}

    def work():
        flag["ran"] = True
        time.sleep(0.05)

    win = ins.measure_once(work, use_A=False, use_B=False, use_C=False)
    assert flag["ran"]
    assert win.wall_time_s >= 0.05
    assert win.available == []          # no instruments requested
    assert win.energy_j == {}


def test_agreement_empty_with_fewer_than_two_instruments():
    win = ins.MeasurementWindow(wall_time_s=1.0, energy_j={"A": 10.0})
    assert win.agreement() == {}


def test_agreement_is_pairwise_relative_difference():
    win = ins.MeasurementWindow(wall_time_s=1.0, energy_j={"A": 10.0, "B": 11.0, "C": 9.0})
    ag = win.agreement()
    assert set(ag) == {"A-B", "A-C", "B-C"}
    assert abs(ag["A-B"] - (1.0 / 11.0)) < 1e-12       # |10-11|/11
    assert abs(ag["B-C"] - (2.0 / 9.0)) < 1e-12        # |11-9|/9


# --- GPU smoke tests (auto-skip without NVML/CUDA) ----------------------------


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_smoke_A_and_B_agree_roughly():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float32)

    for _ in range(10):               # warmup: autotune + cache fill
        _ = a @ b
    torch.cuda.synchronize()

    # Size the window to the protocol's 4 s floor. BELOW that floor both
    # instruments degrade, so an agreement assertion there tests a regime the
    # sweep never operates in. Diagnostic of 2026-08-18
    # (diagnostics/instrument_a.json): on a settled 3.0 s matmul window A-B is
    # 7.1-8.3% at 50-200 Hz, but on a 0.25 s window it is 36-94%, and B itself
    # swung 33.5-52.1 J on identical work there (one reading implying 210 W on
    # a 175 W part). The earlier unfloored ~0.8 s window sat in that degraded
    # band and tripped a 50% assertion intermittently.
    t0 = time.perf_counter()
    for _ in range(20):
        _ = a @ b
    torch.cuda.synchronize()
    per_iter_s = (time.perf_counter() - t0) / 20
    n_iters = max(50, int(WINDOW_FLOOR_S / per_iter_s) + 1)

    def work():
        c = a
        for _ in range(n_iters):
            c = a @ b
        return c

    win = ins.measure_once(work)      # all instruments; C optional
    print(f"\n[smoke] available={win.available} energy_j={win.energy_j} "
          f"n_samples={win.n_power_samples} wall_s={win.wall_time_s:.2f} "
          f"n_iters={n_iters} notes={win.notes}")

    # The window must actually clear the floor, or the agreement check below is
    # not testing what it claims to test.
    assert win.wall_time_s >= WINDOW_FLOOR_S * 0.8, (
        f"window {win.wall_time_s:.2f} s did not reach the {WINDOW_FLOOR_S} s "
        f"floor; instrument agreement is not meaningful below it")

    # B (the hardware accumulator) is the most reliable single reading on Ada.
    assert "B" in win.available, f"energy counter unavailable; notes={win.notes}"
    assert win.energy_j["B"] > 0.0

    if "A" in win.available:
        assert win.energy_j["A"] > 0.0
        assert win.n_power_samples >= 5

    # Smoke gate on a floored window. The pre-registered gate is a 5% pooled
    # median under thermal control, enforced by the runner, not here; this band
    # sits well above the 7-8% measured on a settled window, so the test catches
    # a broken instrument rather than the normal power-vs-counter offset.
    for pair, val in win.agreement().items():
        assert val < AB_SMOKE_BAND, (
            f"{pair} disagree by {val:.0%} on a {win.wall_time_s:.1f} s "
            f"window; notes={win.notes}")


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_zeus_windows_verdict():
    """Resolves the EXP-002 prerequisite: does Zeus return energy on THIS box?

    Passes either way. Prints the verdict so the harness step can record
    measurement.zeus_windows_check accordingly.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device("cuda:0")
    x = torch.randn(4096, 4096, device=dev, dtype=torch.float32)

    def work():
        y = x
        for _ in range(300):          # long enough to exceed the counter tick
            y = x @ x
        return y

    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()

    win = ins.measure_once(work, use_A=False, use_B=False, use_C=True)
    if "C" in win.available and win.energy_j["C"] > 0.0:
        print(f"\n[EXP-002] Zeus on Windows: OK  (energy={win.energy_j['C']:.3f} J)")
    elif "C" in win.available:
        # Imported and ran, but reported zero: coarse energy-counter tick vs window
        # length, not a Zeus-on-Windows failure. A+B remain the primary path.
        print(f"\n[EXP-002] Zeus on Windows: imports/runs but returned 0 J on this "
              f"window (counter granularity). A+B carry the experiment. notes={win.notes}")
    else:
        print(f"\n[EXP-002] Zeus on Windows: UNAVAILABLE -> A+B carry the "
              f"experiment. notes={win.notes}")
