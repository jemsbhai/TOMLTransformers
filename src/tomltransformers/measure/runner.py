"""Controlled measurement runner for EXP-002.

Wraps the raw `measure_once` instrument primitive with the controls the
pre-registration (configs/exp_002.yaml) requires, turning a single noisy reading
into a defensible per-point measurement:

  - warmup iterations (run but NOT measured), to trigger JIT / cuDNN autotune /
    cache fills before timing begins;
  - thermal settling: wait until GPU temperature is stable within a tolerance
    over a window before measuring (leakage current is exponential in temp);
  - a per-point idle baseline: mean idle power P_idle, subtracted UNIFORMLY from
    every instrument as P_idle * active_wall_time. Instrument A integrates
    (P_active - P_idle) directly; B and C read a total-energy accumulator, so we
    subtract the same P_idle * dt from them. This makes idle subtraction
    IDENTICAL across A/B/C, so any residual A-vs-B gap reflects the active
    measurement method, not three different baseline treatments. The reported
    energy is therefore DYNAMIC energy above idle (the workload's marginal cost),
    matching the prior TOML papers' deltaP * t convention and what the energy
    model fits. Raw totals are retained alongside.
  - repeats with mean / std / CV, and a CV gate flag;
  - actual clocks and temperatures logged per point;
  - OOM caught and recorded as a skip, never crashing the sweep.

This module is workload-agnostic: it measures any zero-argument callable. Real
transformer workloads are constructed elsewhere and passed in.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

from .instruments import measure_once, MeasurementWindow


# --- helpers: device telemetry (best-effort; degrade quietly) -----------------


def _read_idle_power_w(device_index: int, seconds: float, sampling_hz: float = 100.0) -> Optional[float]:
    """Mean idle GPU power (W) over `seconds`, via NVML. None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            interval = 1.0 / sampling_hz
            vals: List[float] = []
            t_end = time.perf_counter() + seconds
            while time.perf_counter() < t_end:
                vals.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                time.sleep(interval)
            return sum(vals) / len(vals) if vals else None
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _read_temp_c(device_index: int) -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _read_clocks_mhz(device_index: int) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"sm": None, "mem": None}
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            out["sm"] = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            out["mem"] = float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return out


def wait_for_thermal_settle(
    device_index: int,
    tolerance_c: float = 1.0,
    window_s: float = 5.0,
    timeout_s: float = 120.0,
    poll_hz: float = 2.0,
) -> Dict[str, object]:
    """Block until GPU temperature stays within `tolerance_c` over `window_s`.

    Returns a record of whether settling was achieved and the final temp. Never
    raises; if NVML is unavailable it returns immediately with settled=None.
    """
    t0 = _read_temp_c(device_index)
    if t0 is None:
        return {"settled": None, "reason": "no temperature telemetry", "temp_c": None}

    interval = 1.0 / poll_hz
    need = max(1, int(window_s * poll_hz))
    recent: List[float] = []
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        t = _read_temp_c(device_index)
        if t is None:
            break
        recent.append(t)
        if len(recent) > need:
            recent.pop(0)
        if len(recent) == need and (max(recent) - min(recent)) <= tolerance_c:
            return {"settled": True, "temp_c": t, "spread_c": max(recent) - min(recent)}
        time.sleep(interval)
    return {
        "settled": False,
        "temp_c": recent[-1] if recent else None,
        "reason": "timeout",
    }


# --- per-point result ---------------------------------------------------------


@dataclass
class PointResult:
    """Aggregated controlled measurement for one workload point."""

    label: str
    ok: bool
    n_repeats: int = 0
    # Dynamic energy above idle (joules), per instrument: {"A":[...], ...}
    energy_j_dyn: Dict[str, List[float]] = field(default_factory=dict)
    energy_j_raw: Dict[str, List[float]] = field(default_factory=dict)
    wall_time_s: List[float] = field(default_factory=list)
    # Summary stats over repeats, per instrument: {"A": {"mean":..,"std":..,"cv":..}}
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cv_exceeded: List[str] = field(default_factory=list)   # instruments over the CV gate
    idle_power_w: Optional[float] = None
    temps_c: List[Optional[float]] = field(default_factory=list)
    clocks_mhz: List[Dict[str, Optional[float]]] = field(default_factory=list)
    thermal: Dict[str, object] = field(default_factory=dict)
    instruments_available: List[str] = field(default_factory=list)
    short_window: bool = False     # True if median measured wall-time < min_window_s
    skip_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def as_record(self) -> dict:
        return asdict(self)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "cv": float("nan"),
                "median": float("nan"), "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = (std / mean) if mean else float("nan")
    return {"mean": mean, "std": std, "cv": cv,
            "median": statistics.median(values), "n": len(values)}


# --- the runner ---------------------------------------------------------------


def measure_point(
    fn: Callable[[], object],
    label: str,
    *,
    repeats: int = 5,
    warmup_iters: int = 50,
    idle_baseline_s: float = 3.0,
    cv_threshold: float = 0.05,
    min_window_s: float = 2.0,
    sampling_hz: float = 100.0,
    device_index: int = 0,
    thermal_settle: bool = True,
    thermal_tolerance_c: float = 1.0,
    thermal_window_s: float = 5.0,
    use_A: bool = True,
    use_B: bool = True,
    use_C: bool = True,
    sync: bool = True,
) -> PointResult:
    """Measure one workload point under controlled conditions.

    `fn` is a zero-argument callable that performs ONE execution of the workload
    (e.g. one prefill forward, or one decode window of K tokens). It is called
    `warmup_iters` times unmeasured, then `repeats` times measured.

    `min_window_s` is the window-length floor from the instrument-A diagnostic:
    on sub-second windows even the hardware energy counter is self-inconsistent,
    so a measured execution should last at least this long. This is a SOFT guard:
    if the median measured wall-time is under the floor, `res.short_window` is set
    True and a note is added, but the point is still measured and returned. It is
    the sweep's job to size each workload (loop the per-execution work) so small
    models clear the floor; this guard catches the cases where it failed to.

    OOM (torch.cuda.OutOfMemoryError or a RuntimeError mentioning 'out of memory')
    during warmup or measurement is caught and recorded as a skip; the sweep
    continues.
    """
    res = PointResult(label=label, ok=False)

    def _is_oom(exc: BaseException) -> bool:
        name = type(exc).__name__
        if name == "OutOfMemoryError":
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def _empty_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- warmup (unmeasured) ----
    try:
        for _ in range(warmup_iters):
            fn()
        if sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
    except BaseException as exc:  # noqa: BLE001
        if _is_oom(exc):
            res.skip_reason = "OOM"
            res.notes.append(f"OOM during warmup: {exc!r}")
            _empty_cache()
            return res
        raise

    # ---- thermal settle (after warmup, so we settle at working temperature) ----
    if thermal_settle:
        res.thermal = wait_for_thermal_settle(
            device_index, tolerance_c=thermal_tolerance_c, window_s=thermal_window_s
        )

    # ---- idle baseline (subtracted uniformly from A/B/C as P_idle * dt) ----
    res.idle_power_w = _read_idle_power_w(device_index, idle_baseline_s, sampling_hz)

    # ---- measured repeats ----
    for _ in range(repeats):
        res.temps_c.append(_read_temp_c(device_index))
        res.clocks_mhz.append(_read_clocks_mhz(device_index))
        try:
            win: MeasurementWindow = measure_once(
                fn,
                use_A=use_A, use_B=use_B, use_C=use_C,
                sampling_hz=sampling_hz, device_index=device_index, sync=sync,
            )
        except BaseException as exc:  # noqa: BLE001
            if _is_oom(exc):
                res.skip_reason = "OOM"
                res.notes.append(f"OOM during measurement: {exc!r}")
                _empty_cache()
                return res
            raise

        res.wall_time_s.append(win.wall_time_s)
        if win.notes:
            res.notes.extend(win.notes)
        idle_e = (res.idle_power_w * win.wall_time_s) if res.idle_power_w else 0.0
        for key, e_raw in win.energy_j.items():
            res.energy_j_raw.setdefault(key, []).append(e_raw)
            # Dynamic energy cannot go negative; clamp at 0 and note if it would.
            e_dyn = e_raw - idle_e
            if e_dyn < 0:
                res.notes.append(f"{key}: dynamic energy < 0 (raw {e_raw:.3f} J, "
                                 f"idle {idle_e:.3f} J); clamped to 0")
                e_dyn = 0.0
            res.energy_j_dyn.setdefault(key, []).append(e_dyn)

    res.n_repeats = len(res.wall_time_s)
    res.instruments_available = sorted(res.energy_j_dyn.keys())

    # ---- window-length floor (soft guard; diagnostic 2026-05-29) ----
    # Sub-floor windows put even the hardware counter in its unreliable regime.
    # Flag on the MEDIAN wall-time so a single transient repeat does not trip it.
    if res.wall_time_s:
        median_wall = statistics.median(res.wall_time_s)
        if median_wall < min_window_s:
            res.short_window = True
            res.notes.append(
                f"SHORT WINDOW: median wall {median_wall:.3f} s < min_window_s "
                f"{min_window_s:.1f} s; energy (incl. hardware counter B) is in the "
                f"unreliable sub-window regime. Loop the workload more per execution."
            )

    # ---- summaries + CV gate (on dynamic energy, the reported quantity) ----
    for key, vals in res.energy_j_dyn.items():
        s = _summarize(vals)
        res.summary[key] = s
        if s["n"] > 1 and s["cv"] == s["cv"] and s["cv"] > cv_threshold:
            res.cv_exceeded.append(key)
    res.summary["wall_time_s"] = _summarize(res.wall_time_s)

    res.ok = res.n_repeats > 0 and len(res.instruments_available) > 0
    return res


def pairwise_agreement(res: PointResult) -> Dict[str, float]:
    """Pairwise relative abs difference of MEAN dynamic energy across instruments.

    |mean_i - mean_j| / mean_j. Empty if fewer than two instruments measured.
    """
    out: Dict[str, float] = {}
    keys = [k for k in ("A", "B", "C") if k in res.summary]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            mb = res.summary[b]["mean"]
            if mb and mb == mb:  # nonzero, not NaN
                out[f"{a}-{b}"] = abs(res.summary[a]["mean"] - res.summary[b]["mean"]) / mb
    return out
