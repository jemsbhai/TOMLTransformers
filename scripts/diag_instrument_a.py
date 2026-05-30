"""DIAGNOSTIC (throwaway, NOT part of the package, NOT imported by anything).

Purpose: resolve the EXP-002 instrument-A question recorded in findings.md
(2026-05-29): instrument A (sampled-power integration) reads ~14% LOW vs the
B/C hardware energy counter on a long controlled window. Decide whether this is
(i) fixable sampler/integration error, after which A corroborates B, or
(ii) a genuine property of nvmlDeviceGetPowerUsage, in which case B (the hardware
counter) becomes the PRIMARY reported instrument and A is a sanity check.

It investigates five things on ONE long, thermally-settled window per workload:
  1. Raw (timestamp, power) samples from an in-process sampler: count, and the
     actual inter-sample spacing vs the nominal 1/hz (starvation shows here).
  2. B (nvmlDeviceGetTotalEnergyConsumption) and C (Zeus) as ground truth.
  3. A recomputed THREE ways from the same samples: trapezoid, rectangle
     (left-Riemann), and mean-power*duration. Disagreement among these =>
     integration is sample-placement-sensitive. Agreement among them but all
     below B => genuine bias, not integration.
  4. Coverage: fraction of the true workload window [t0,t1] actually spanned by
     [first_sample, last_sample]. If A integrates over a shorter span than the
     window, it reads low for a precise, fixable reason.
  5. A sampling-rate sweep (20/50/100/200 Hz) and an EXTERNAL nvidia-smi
     power logger (separate process, no GIL contention). If the gap shrinks with
     rate, or the external logger integrates to B while the in-process sampler
     does not, the cause is in-process sampling (GIL/latency), not NVML.

Workloads: synthetic sustained matmul AND a real GPT-2 prefill forward.
NOTE: the GPT-2 path downloads gpt2 (~0.5 GB) via transformers on first run.

Run:  python scripts/diag_instrument_a.py
Output: prints a report and writes experiments/exp_002_size_sweep/diagnostics/instrument_a.json
This script is exploratory; it has no tests and must not be imported by the package.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

DEVICE_INDEX = 0
OUT_DIR = os.path.join("experiments", "exp_002_size_sweep", "diagnostics")


# ---------------------------------------------------------------------------
# In-process sampler that records RAW samples (timestamp_s, power_w).
# ---------------------------------------------------------------------------
class RawSampler(threading.Thread):
    def __init__(self, handle, hz: float):
        super().__init__(daemon=True)
        import pynvml
        self._pynvml = pynvml
        self._handle = handle
        self._interval = 1.0 / hz
        self._stop_evt = threading.Event()
        self.samples: List[tuple] = []

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            except Exception:
                break
            self.samples.append((time.perf_counter(), mw / 1000.0))
            self._stop_evt.wait(self._interval)

    def stop(self) -> None:
        self._stop_evt.set()


# ---------------------------------------------------------------------------
# Three ways to integrate the same (t, w) samples into energy (J).
# ---------------------------------------------------------------------------
def integrate_three_ways(ts: np.ndarray, ws: np.ndarray) -> Dict[str, float]:
    if len(ts) < 2:
        return {"trapezoid": float("nan"), "rectangle": float("nan"),
                "mean_x_duration": float("nan")}
    trap = getattr(np, "trapezoid", None) or np.trapz
    dur = float(ts[-1] - ts[0])
    dt = np.diff(ts)
    return {
        "trapezoid": float(trap(ws, ts)),
        "rectangle": float(np.sum(ws[:-1] * dt)),          # left-Riemann
        "mean_x_duration": float(np.mean(ws) * dur),
    }


def spacing_stats(ts: np.ndarray, nominal_interval: float) -> Dict[str, float]:
    if len(ts) < 2:
        return {"n": int(len(ts))}
    dt = np.diff(ts)
    return {
        "n": int(len(ts)),
        "nominal_interval_ms": round(nominal_interval * 1000, 3),
        "mean_gap_ms": round(float(np.mean(dt)) * 1000, 3),
        "median_gap_ms": round(float(np.median(dt)) * 1000, 3),
        "max_gap_ms": round(float(np.max(dt)) * 1000, 3),
        "p95_gap_ms": round(float(np.percentile(dt, 95)) * 1000, 3),
    }


# ---------------------------------------------------------------------------
# External nvidia-smi power logger (separate process; no GIL contention).
# ---------------------------------------------------------------------------
class NvidiaSmiLogger:
    def __init__(self, loop_ms: int = 20):
        self._loop_ms = loop_ms
        self._fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="")
        self._path = self._fh.name
        self._fh.close()
        self._proc: Optional[subprocess.Popen] = None
        self._t_start: Optional[float] = None

    def start(self) -> bool:
        try:
            out = open(self._path, "w")
            # power.draw in W; timestamp from nvidia-smi itself.
            self._proc = subprocess.Popen(
                ["nvidia-smi",
                 f"--query-gpu=timestamp,power.draw",
                 "--format=csv,noheader,nounits",
                 f"--loop-ms={self._loop_ms}"],
                stdout=out, stderr=subprocess.DEVNULL,
            )
            self._t_start = time.perf_counter()
            return True
        except Exception:
            return False

    def stop_and_integrate(self) -> Dict[str, object]:
        if self._proc is None:
            return {"available": False, "reason": "nvidia-smi not started"}
        time.sleep(0.05)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        # Parse CSV: columns timestamp, power. Use row index * loop_ms for time
        # base (nvidia-smi timestamp parsing is locale-fragile); spacing is fixed.
        ws: List[float] = []
        try:
            with open(self._path, "r") as fh:
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        try:
                            ws.append(float(row[1]))
                        except ValueError:
                            pass
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": f"parse failed: {exc!r}"}
        finally:
            try:
                os.unlink(self._path)
            except Exception:
                pass
        if len(ws) < 2:
            return {"available": False, "reason": f"too few rows ({len(ws)})"}
        dt = self._loop_ms / 1000.0
        wv = np.asarray(ws, dtype=float)
        # trapezoid on a uniform grid:
        energy = float((np.sum(wv) - 0.5 * (wv[0] + wv[-1])) * dt)
        return {
            "available": True,
            "n_rows": int(len(ws)),
            "loop_ms": self._loop_ms,
            "mean_power_w": round(float(np.mean(wv)), 2),
            "energy_j": round(energy, 3),
        }


# ---------------------------------------------------------------------------
# Core: run ONE window of `fn` while measuring A (raw), B, C, and external smi.
# ---------------------------------------------------------------------------
def measure_window(fn: Callable[[], object], hz: float, with_smi: bool) -> Dict[str, object]:
    import pynvml
    import torch

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(DEVICE_INDEX)

    # Zeus (C)
    zmon = None
    try:
        from zeus.monitor import ZeusMonitor
        try:
            zmon = ZeusMonitor(gpu_indices=[DEVICE_INDEX], approx_instant_energy=True)
        except TypeError:
            zmon = ZeusMonitor(gpu_indices=[DEVICE_INDEX])
    except Exception:
        zmon = None

    smi = NvidiaSmiLogger(loop_ms=int(1000 / hz)) if with_smi else None
    smi_started = smi.start() if smi else False
    if smi_started:
        time.sleep(0.2)  # let nvidia-smi spin up before the window

    e0_b = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
    sampler = RawSampler(handle, hz)
    sampler.start()
    if zmon is not None:
        try:
            zmon.begin_window("w")
        except Exception:
            zmon = None

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    e1_b = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
    sampler.stop()
    sampler.join(timeout=2.0)

    energy_b = (e1_b - e0_b) / 1000.0
    energy_c = None
    if zmon is not None:
        try:
            energy_c = float(zmon.end_window("w").total_energy)
        except Exception:
            energy_c = None

    smi_result = smi.stop_and_integrate() if smi_started else {"available": False}

    s = sampler.samples
    ts = np.asarray([p[0] for p in s], dtype=float)
    ws = np.asarray([p[1] for p in s], dtype=float)

    wall = t1 - t0
    methods = integrate_three_ways(ts, ws)
    # Coverage: how much of [t0,t1] the samples actually span.
    coverage = None
    first_gap_ms = last_gap_ms = None
    if len(ts) >= 2:
        coverage = round((float(ts[-1]) - float(ts[0])) / wall, 4)
        first_gap_ms = round((float(ts[0]) - t0) * 1000, 2)   # delay before 1st sample
        last_gap_ms = round((t1 - float(ts[-1])) * 1000, 2)   # gap after last sample

    # A "corrected" estimate that uses mean power over the FULL wall window,
    # i.e. assumes the uncovered head/tail ran at mean power too.
    a_full_window = float(np.mean(ws) * wall) if len(ws) >= 1 else None

    pynvml.nvmlShutdown()

    def rel(x):
        return None if (x is None or not energy_b) else round(abs(x - energy_b) / energy_b, 4)

    return {
        "hz": hz,
        "wall_s": round(wall, 4),
        "energy_B_j": round(energy_b, 3),
        "energy_C_j": (round(energy_c, 3) if energy_c is not None else None),
        "A_methods_j": {k: (round(v, 3) if v == v else None) for k, v in methods.items()},
        "A_full_window_j": (round(a_full_window, 3) if a_full_window else None),
        "vs_B": {
            "trapezoid": rel(methods["trapezoid"]),
            "rectangle": rel(methods["rectangle"]),
            "mean_x_duration": rel(methods["mean_x_duration"]),
            "A_full_window": rel(a_full_window),
            "C": rel(energy_c),
            "smi": rel(smi_result.get("energy_j")) if smi_result.get("available") else None,
        },
        "spacing": spacing_stats(ts, 1.0 / hz),
        "coverage_frac": coverage,
        "first_sample_delay_ms": first_gap_ms,
        "last_sample_gap_ms": last_gap_ms,
        "smi": smi_result,
    }


# ---------------------------------------------------------------------------
# Workloads.
# ---------------------------------------------------------------------------
def make_matmul(n: int = 4096, iters: int = 600):
    import torch
    dev = torch.device("cuda:0")
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)

    def fn():
        c = b
        for _ in range(iters):
            c = a @ c
        return c

    # warmup
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    return fn


def _gpt2_already_cached() -> bool:
    """True if gpt2 is already in the HF cache before this run (so we KEEP it)."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == "gpt2" and repo.repo_type == "model":
                return True
    except Exception:
        pass
    return False


def _delete_gpt2_from_cache() -> Optional[str]:
    """Delete only the gpt2 model repo from the HF cache. Returns freed-size note."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        revs = set()
        for repo in info.repos:
            if repo.repo_id == "gpt2" and repo.repo_type == "model":
                revs |= {rev.commit_hash for rev in repo.revisions}
        if not revs:
            return "nothing to delete (gpt2 not in cache)"
        strategy = info.delete_revisions(*revs)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        return f"deleted gpt2 from HF cache, freed ~{freed}"
    except Exception as exc:  # noqa: BLE001
        return f"cache cleanup failed (delete manually if needed): {exc!r}"


def make_gpt2_prefill(seq_len: int = 1024, repeats: int = 40):
    """Real GPT-2 forward. Downloads gpt2 (~0.5 GB) on first run."""
    import torch
    from transformers import GPT2LMHeadModel
    dev = torch.device("cuda:0")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).half().eval()
    ids = torch.randint(0, 50257, (1, seq_len), device=dev)

    @torch.no_grad()
    def fn():
        for _ in range(repeats):   # loop so the window is long enough to sample
            model(ids)

    for _ in range(5):
        with torch.no_grad():
            model(ids)
    torch.cuda.synchronize()
    return fn


def settle(seconds: float = 6.0, tol: float = 1.0):
    """Crude thermal settle: wait until temp stable within tol over a short window."""
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(DEVICE_INDEX)
    recent: List[float] = []
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        t = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        recent.append(t)
        if len(recent) > int(seconds * 2):
            recent.pop(0)
        if len(recent) == int(seconds * 2) and (max(recent) - min(recent)) <= tol:
            pynvml.nvmlShutdown()
            return t
        time.sleep(0.5)
    pynvml.nvmlShutdown()
    return recent[-1] if recent else None


def run_for_workload(name: str, builder: Callable[[], Callable], with_smi: bool) -> Dict[str, object]:
    print(f"\n=== workload: {name} ===")
    fn = builder()
    print(f"  settling thermally...")
    temp = settle()
    print(f"  settled near {temp} C")
    rate_sweep = []
    for hz in (20, 50, 100, 200):
        print(f"  measuring at {hz} Hz ...", flush=True)
        r = measure_window(fn, hz=hz, with_smi=with_smi)
        rate_sweep.append(r)
        a_trap = r["A_methods_j"]["trapezoid"]
        print(f"    B={r['energy_B_j']} J  A(trap)={a_trap} J  "
              f"vs_B(trap)={r['vs_B']['trapezoid']}  "
              f"coverage={r['coverage_frac']}  "
              f"1st_delay={r['first_sample_delay_ms']}ms  "
              f"smi_vs_B={r['vs_B'].get('smi')}  n={r['spacing'].get('n')}")
    return {"workload": name, "rate_sweep": rate_sweep}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available; this diagnostic needs the GPU.")
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"torch unavailable: {exc!r}")
        sys.exit(1)

    with_smi = True
    results = {"device_index": DEVICE_INDEX, "workloads": []}

    results["workloads"].append(run_for_workload("matmul_4096", make_matmul, with_smi))

    # GPT-2: record whether it was cached BEFORE this run; only delete what we fetch.
    gpt2_preexisting = _gpt2_already_cached()
    if gpt2_preexisting:
        print("\n[storage] gpt2 already in HF cache; it will be KEPT (not this run's download).")
    else:
        print("\n[storage] gpt2 not cached; this run will download it and DELETE it on exit.")
    try:
        results["workloads"].append(
            run_for_workload("gpt2_prefill_s1024", make_gpt2_prefill, with_smi))
    except Exception as exc:  # noqa: BLE001
        print(f"  GPT-2 workload failed/skipped: {exc!r}")
        results["workloads"].append({"workload": "gpt2_prefill_s1024", "error": repr(exc)})
    finally:
        # Clean up only if WE downloaded it (and only the gpt2 repo). Runs even on crash.
        if not gpt2_preexisting:
            note = _delete_gpt2_from_cache()
            print(f"[storage] {note}")

    path = os.path.join(OUT_DIR, "instrument_a.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {path}")
    print("\nInterpretation guide:")
    print("  - If vs_B(trapezoid) shrinks toward 0 as Hz rises -> undersampling; raise A's rate.")
    print("  - If coverage_frac << 1 or first_sample_delay_ms is large -> A integrates a short")
    print("    span; fix by anchoring integration to [t0,t1] (use A_full_window vs_B as the check).")
    print("  - If trapezoid/rectangle/mean_x_duration disagree -> placement-sensitive integration.")
    print("  - If they AGREE but all sit ~14% below B at every Hz, and smi_vs_B is ALSO ~14% ->")
    print("    genuine NVML power-vs-counter bias; make B primary, A a sanity check.")
    print("  - If smi_vs_B ~ 0 but in-process A is ~14% low -> GIL/in-process sampling is the cause.")


if __name__ == "__main__":
    main()
