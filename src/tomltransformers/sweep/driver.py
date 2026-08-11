"""Sweep driver: pre-flight, expand the grid, and measure every point with
crash-safe, resumable JSONL output.

Ties the pieces together:
  preflight() (hard git gate)  ->  expand_grid()  ->  for each point not already
  done: measure_single_point() with regime-aware repeats  ->  append one JSONL
  record immediately.

Resumability (your choice): a point is SKIPPED only if its latest JSONL record is
ok=True. Points previously recorded as failed, OOM, or short_window are RE-RUN
(a transient failure or an undersized window should not be permanent). Records are
append-only (crash-safe); on read we keep the LAST record per key (last-write-wins),
so a successful re-run supersedes an earlier failure without rewriting the file.

Regime-aware repeats (your choice): decode and decoder_prefill were measured
noisier than the steady forward phases (findings 2026-05-29), so they get more
physical replicates.

Multi-pass configs (2026-08-10, Step 6 of the A100 amendment): a config that
carries a `passes` list (see sweep/grid_passes.py) is expanded by
expand_passes(), which owns weights, anchors, and attention-compare PER PASS
and derives every point's explicit init seed; the per-call expander kwargs
(enc_dec_anchor, include_attention_compare, weights) apply only to classic
single-pass configs. Dispatch lives in build_points_from_config() so it is
unit-testable without any measurement.

This module performs measurement only through measure_single_point; it adds no
new energy logic, just orchestration, provenance, and resumability.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .provenance import preflight, PreflightError
from .point import PointSpec, measure_single_point
from .grid import expand_grid, load_config
from .grid_passes import expand_passes


# Regime-aware physical replicates per point.
_REPEATS_NOISY = 10        # decode, decoder_prefill
_REPEATS_STEADY = 5        # prefill, encode
_NOISY_PHASES = {"decode", "decoder_prefill"}


def _repeats_for(phase: str) -> int:
    return _REPEATS_NOISY if phase in _NOISY_PHASES else _REPEATS_STEADY


def build_points_from_config(
    cfg: dict,
    *,
    enc_dec_anchor: int = 1024,
    include_attention_compare: bool = True,
    weights: str = "random",
) -> List[PointSpec]:
    """Dispatch a loaded config dict to the right grid expander.

    Multi-pass configs (a `passes` list; see sweep/grid_passes.py) are expanded
    by expand_passes(), which reads weights, anchors, attention-compare, and
    pretrained_id PER PASS from the config itself and assigns every point its
    derived init seed; the kwargs here are ignored for them. Classic
    single-pass configs go through expand_grid() with the kwargs, exactly as
    before (frozen-4090 behavior unchanged).
    """
    if "passes" in cfg:
        return expand_passes(cfg)
    return expand_grid(
        cfg, enc_dec_anchor=enc_dec_anchor,
        include_attention_compare=include_attention_compare, weights=weights,
    )


def read_done_keys(results_path: str) -> set:
    """Return the set of keys whose LATEST record is ok=True AND not short_window.

    Last-write-wins: later records for the same key override earlier ones, so a
    successful re-run supersedes a prior failure. A point counts as done only when
    its latest record both succeeded (ok=True) and cleared the window floor
    (short_window False): a short window leaves even the hardware counter in its
    unreliable regime, so it is treated as needs-retry, not final. Malformed lines
    are skipped.
    """
    if not os.path.isfile(results_path):
        return set()
    latest: Dict[str, bool] = {}
    with open(results_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = rec.get("spec", {}).get("key")
            if key is None:
                continue
            # done iff ok AND the window floor was cleared.
            latest[key] = bool(rec.get("ok", False)) and not bool(rec.get("short_window", False))
    return {k for k, done in latest.items() if done}


def _append_record(results_path: str, record: dict) -> None:
    """Append one record as a JSON line and flush+fsync (crash-safe)."""
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class SweepProgress:
    total: int = 0
    measured: int = 0
    skipped_done: int = 0
    ok: int = 0
    failed: int = 0
    oom: int = 0
    short_window: int = 0
    stopped_early: bool = False           # True if a max_hours budget halted the run
    keys_failed: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total": self.total, "measured": self.measured,
            "skipped_done": self.skipped_done, "ok": self.ok,
            "failed": self.failed, "oom": self.oom,
            "short_window": self.short_window, "stopped_early": self.stopped_early,
            "keys_failed": self.keys_failed,
        }


def run_sweep(
    *,
    config_path: Optional[str] = None,
    points: Optional[List[PointSpec]] = None,
    run_dir: str,
    results_filename: str = "energy.jsonl",
    allow_dirty: bool = False,
    allow_untracked_paths: Optional[List[str]] = None,
    enc_dec_anchor: int = 1024,
    include_attention_compare: bool = True,
    weights: str = "random",
    target_s: float = 4.0,
    min_window_s: float = 4.0,
    sampling_hz: float = 100.0,
    max_hours: Optional[float] = None,
    log: bool = True,
    progress_callback=None,
) -> SweepProgress:
    """Run the EXP-002 sweep (or any explicit PointSpec list) with resumable output.

    Exactly one of config_path or points must be given:
      - config_path: load the config and build the grid via
        build_points_from_config (multi-pass configs dispatch to
        expand_passes; classic configs to expand_grid).
      - points: an explicit list of PointSpecs (the expander's output, or custom).

    Pre-flight runs first and HARD-REFUSES a dirty git tree unless allow_dirty
    (records the override). allow_untracked_paths exempts ONLY untracked copies
    of this run's own outputs (see provenance.preflight); tracked modifications
    still refuse, preserving the between-chunk harvest-commit ritual.
    environment.json and the frozen config copy are written into run_dir.
    Results are appended to run_dir/results_filename as JSONL.

    max_hours: if set, the sweep stops cleanly BEFORE starting a point once this
    wall-clock budget is exceeded (the in-progress point always finishes and is
    recorded). This supports running the long sweep in nightly chunks: re-run the
    same command and resume skips the done points. None = no budget (run to end).

    Returns a SweepProgress summary. Never raises on a single point's failure: the
    failure is recorded and the sweep continues. PreflightError (dirty tree) is the
    one fatal condition, raised before any measurement.
    """
    if (config_path is None) == (points is None):
        raise ValueError("provide exactly one of config_path or points")

    results_path = os.path.join(run_dir, results_filename)

    # ---- pre-flight: provenance + hard git gate (before any measurement) ----
    pf = preflight(run_dir, config_path=config_path, allow_dirty=allow_dirty,
                   allow_untracked_paths=allow_untracked_paths)
    if log:
        print(f"[preflight] run_dir={run_dir} git_commit={pf.git_commit} "
              f"dirty={pf.git_dirty} overridden={pf.overridden}")
        for w in pf.warnings:
            print(f"[preflight] WARNING: {w}")

    # ---- build the grid ----
    if points is None:
        cfg = load_config(config_path)
        points = build_points_from_config(
            cfg, enc_dec_anchor=enc_dec_anchor,
            include_attention_compare=include_attention_compare, weights=weights,
        )
        if log and "passes" in cfg:
            print(f"[sweep] multi-pass config: {len(points)} seeded points via "
                  f"expand_passes (per-pass weights/anchors; expander kwargs ignored)")

    prog = SweepProgress(total=len(points))
    done = read_done_keys(results_path)
    if log:
        print(f"[sweep] {len(points)} points; {len(done)} already done (ok=True); "
              f"results -> {results_path}")

    t_start = time.time()
    budget_s = (max_hours * 3600.0) if max_hours else None
    for i, ps in enumerate(points, 1):
        key = ps.key()
        if key in done:
            prog.skipped_done += 1
            continue

        # Wall-clock budget: stop BEFORE starting a new point once exceeded, so
        # the in-progress point is never interrupted. Nightly-chunk support.
        if budget_s is not None and (time.time() - t_start) >= budget_s:
            prog.stopped_early = True
            if log:
                remaining = sum(1 for p in points[i - 1:] if p.key() not in done)
                print(f"[sweep] max_hours budget ({max_hours}h) reached; stopping "
                      f"cleanly with ~{remaining} point(s) left. Re-run to resume.",
                      flush=True)
            break

        repeats = _repeats_for(ps.phase)
        if log:
            print(f"[sweep] ({i}/{len(points)}) measuring {key} (repeats={repeats}) ...",
                  flush=True)

        rec = measure_single_point(
            ps, target_s=target_s, repeats=repeats,
            min_window_s=min_window_s, sampling_hz=sampling_hz,
        )
        _append_record(results_path, rec)
        prog.measured += 1

        if rec.get("ok"):
            prog.ok += 1
            if rec.get("short_window"):
                prog.short_window += 1
        else:
            prog.failed += 1
            prog.keys_failed.append(key)
            if rec.get("oom_skipped"):
                prog.oom += 1

        if log:
            b = rec.get("per_execution_j", {}).get("B")
            bu = rec.get("per_unit_j", {}).get("B")
            status = ("OK" if rec.get("ok") else
                      ("OOM" if rec.get("oom_skipped") else "FAIL"))
            extra = " SHORT" if rec.get("short_window") else ""
            bstr = f"B/exec={b:.2f}J B/unit={bu:.4g}J" if b is not None else ""
            print(f"[sweep]   -> {status}{extra} {bstr} "
                  f"(elapsed {time.time() - t_start:.0f}s)", flush=True)

        if progress_callback is not None:
            progress_callback(prog, rec)

    if log:
        print(f"[sweep] done. {prog.as_dict()}")
    # Write a small summary file alongside results.
    try:
        with open(os.path.join(run_dir, "sweep_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"progress": prog.as_dict(),
                       "git_commit": pf.git_commit,
                       "elapsed_s": time.time() - t_start}, fh, indent=2)
    except Exception:
        pass
    return prog
