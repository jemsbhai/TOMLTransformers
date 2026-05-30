"""Energy-measurement instruments for EXP-002.

Three independent instruments measure the energy of a SINGLE GPU-workload
execution, concentrically (one execution, measured three ways), so their
readings are directly comparable for agreement analysis:

  A  PowerIntegrator  our 20 Hz nvmlDeviceGetPowerUsage sampling, integrated
                      over time (the method used by the prior TOML papers).
  B  EnergyCounter    our direct read of nvmlDeviceGetTotalEnergyConsumption,
                      the on-die energy accumulator (Volta+; the RTX 4090 is Ada).
  C  ZeusInstrument   Zeus ZeusMonitor, an independent peer-reviewed implementation.

A and B are primary and always attempted; C is an independent cross-check and is
NEVER required (the experiment stands on A and B alone if Zeus is unavailable).

Scope of THIS layer: a single `measure_once` runs the workload exactly once and
reports raw energy over that window. Warmup, repeats, thermal settling, locked
clocks, and idle-baseline subtraction are the runner's responsibility, not this
layer's. Energy is always in joules, time in seconds.

NVML unit reminders: nvmlDeviceGetPowerUsage returns milliwatts;
nvmlDeviceGetTotalEnergyConsumption returns millijoules (cumulative since driver
load). Zeus Measurement.total_energy is already joules.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

# --- availability probes (import-safe on machines with no GPU / no Zeus) ------


def nvml_available() -> bool:
    """True if NVML can be initialized (driver present)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except Exception:
        return False


def zeus_available() -> bool:
    """True if Zeus is importable. Does NOT guarantee it can read energy here."""
    try:
        from zeus.monitor import ZeusMonitor  # noqa: F401
        return True
    except Exception:
        return False


def energy_counter_supported(index: int = 0) -> bool:
    """True if nvmlDeviceGetTotalEnergyConsumption works on this device (Volta+)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(index)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)  # raises if unsupported
            return True
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


# --- result container ---------------------------------------------------------


@dataclass
class MeasurementWindow:
    """Energy of one workload execution as seen by each available instrument."""

    wall_time_s: float
    energy_j: Dict[str, float] = field(default_factory=dict)   # keys in {"A","B","C"}
    time_s: Dict[str, float] = field(default_factory=dict)     # per-instrument span
    n_power_samples: int = 0                                   # A's sample count
    available: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def agreement(self) -> Dict[str, float]:
        """Pairwise relative absolute difference |E_i - E_j| / E_j.

        Only over instruments that produced an energy reading. Empty if fewer
        than two are present.
        """
        out: Dict[str, float] = {}
        keys = [k for k in ("A", "B", "C") if k in self.energy_j]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                denom = self.energy_j[b]
                if denom:
                    out[f"{a}-{b}"] = abs(self.energy_j[a] - self.energy_j[b]) / denom
        return out


# --- instrument A: background power sampler -----------------------------------


class _PowerSampler(threading.Thread):
    """Polls instantaneous GPU power on a background thread at a fixed rate."""

    def __init__(self, handle, hz: float):
        super().__init__(daemon=True)
        self._handle = handle
        self._interval = 1.0 / hz
        self._stop_evt = threading.Event()
        self.samples: List[tuple] = []  # (t_perf_s, power_w)

    def run(self) -> None:
        import pynvml
        while not self._stop_evt.is_set():
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            except Exception:
                break
            self.samples.append((time.perf_counter(), mw / 1000.0))
            self._stop_evt.wait(self._interval)

    def stop(self) -> None:
        self._stop_evt.set()


# --- the measurement primitive ------------------------------------------------


def measure_once(
    fn: Callable[[], object],
    *,
    use_A: bool = True,
    use_B: bool = True,
    use_C: bool = True,
    sampling_hz: float = 20.0,
    device_index: int = 0,
    sync: bool = True,
) -> MeasurementWindow:
    """Run ``fn`` exactly once, measuring its GPU energy with every available
    instrument concentrically.

    A CUDA synchronize is issued after ``fn`` returns (before reading any end
    value) so the GPU work is actually complete; otherwise asynchronous kernels
    would be charged to the wrong window. Set ``sync=False`` only if ``fn``
    already synchronizes internally.
    """

    def _cuda_sync() -> None:
        if not sync:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    win = MeasurementWindow(wall_time_s=0.0)

    # Bring up NVML for A and/or B.
    pynvml = None
    handle = None
    if use_A or use_B:
        try:
            import pynvml as _p
            _p.nvmlInit()
            handle = _p.nvmlDeviceGetHandleByIndex(device_index)
            pynvml = _p
        except Exception as exc:  # noqa: BLE001
            win.notes.append(f"NVML unavailable: {exc!r}")
            pynvml, handle = None, None

    # Instrument C: Zeus (optional, independent cross-check).
    zmon = None
    if use_C:
        try:
            from zeus.monitor import ZeusMonitor
            # approx_instant_energy lets Zeus approximate a window shorter than the
            # GPU energy-counter update period as (instant power x duration),
            # instead of returning zero. Needed for short windows (e.g. a single
            # decode step); harmless for long ones.
            try:
                zmon = ZeusMonitor(gpu_indices=[device_index], approx_instant_energy=True)
            except TypeError:
                # Older Zeus without the kwarg.
                zmon = ZeusMonitor(gpu_indices=[device_index])
        except Exception as exc:  # noqa: BLE001
            win.notes.append(f"Zeus unavailable: {exc!r}")
            zmon = None

    # Instrument B: read the energy accumulator at the start.
    e0_b = None
    if pynvml is not None and use_B:
        try:
            e0_b = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
        except Exception as exc:  # noqa: BLE001
            win.notes.append(f"energy counter unsupported: {exc!r}")
            e0_b = None

    # Instrument A: start the background power sampler.
    sampler = None
    if pynvml is not None and use_A:
        sampler = _PowerSampler(handle, sampling_hz)
        sampler.start()

    # Instrument C: open the measurement window.
    if zmon is not None:
        try:
            zmon.begin_window("w")
        except Exception as exc:  # noqa: BLE001
            win.notes.append(f"Zeus begin_window failed: {exc!r}")
            zmon = None

    # ---- run the workload exactly once ----
    t0 = time.perf_counter()
    fn()
    _cuda_sync()
    t1 = time.perf_counter()
    win.wall_time_s = t1 - t0

    # End reads, as close together as possible: B (cheap counter) first.
    if e0_b is not None:
        try:
            e1_b = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
            win.energy_j["B"] = (e1_b - e0_b) / 1000.0
            win.time_s["B"] = win.wall_time_s
            win.available.append("B")
        except Exception as exc:  # noqa: BLE001
            win.notes.append(f"energy counter read failed: {exc!r}")

    if sampler is not None:
        sampler.stop()
        sampler.join(timeout=1.0)
        s = sampler.samples
        win.n_power_samples = len(s)
        if len(s) >= 2:
            import numpy as np
            ts = np.asarray([p[0] for p in s], dtype=float)
            ws = np.asarray([p[1] for p in s], dtype=float)
            integrate = getattr(np, "trapezoid", None) or np.trapz
            win.energy_j["A"] = float(integrate(ws, ts))
            win.time_s["A"] = float(ts[-1] - ts[0])
            win.available.append("A")
        else:
            win.notes.append(f"A: too few power samples ({len(s)}) to integrate")

    if zmon is not None:
        try:
            m = zmon.end_window("w")
            win.energy_j["C"] = float(m.total_energy)
            win.time_s["C"] = float(getattr(m, "time", win.wall_time_s))
            win.available.append("C")
        except Exception as exc:  # noqa: BLE001
            win.notes.append(f"Zeus end_window failed: {exc!r}")

    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return win
