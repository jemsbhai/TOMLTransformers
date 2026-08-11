#!/usr/bin/env python
"""Run the EXP-002 A100 cross-platform sweep (Steps 6-7) on Lambda.

Grid: configs/exp_002_a100.yaml (FROZEN; 98 points; multi-pass expansion with
per-point derived init seeds). Pre-registration:
experiments/exp_002_size_sweep/a100_amendment.md (APPROVED 2026-08-10).
Complete the amendment's section 13 smoke checklist BEFORE the first measured
chunk.

Usage (Lambda instance, Linux, from the repo root):

    python scripts/run_exp002_a100.py --max-hours 3.5
    python scripts/run_exp002_a100.py --allow-dirty   # gate override (recorded)

Resumable identically to the 4090 sweep: re-running continues where it left
off (a point is done iff its latest record is ok=True and not short_window;
failures, OOMs, and short windows are re-run). Output is appended to
experiments/exp_002_size_sweep/a100/energy.jsonl, one JSON record per line.

Provenance gate: HARD-REFUSES a dirty tree, with one exemption -- UNTRACKED
copies of this run's own outputs under experiments/exp_002_size_sweep/a100/
(so a crash before the first harvest commit does not block the resume).
Tracked modifications always refuse: the between-chunk ritual is unchanged:

    run chunk -> validate -> commit harvest (data + reports + docs) -> push
    -> terminate instance -> relaunch -> resume

The instance disk does NOT survive termination; nothing is safe until pushed.

Validate after each chunk with the path-parametrized validator:

    python scripts/validate_exp002.py \\
        --data experiments/exp_002_size_sweep/a100/energy.jsonl \\
        --summary experiments/exp_002_size_sweep/a100/sweep_summary.json \\
        --json experiments/exp_002_size_sweep/a100/validation_report.json \\
        --txt experiments/exp_002_size_sweep/a100/validation_report.txt

Note on validator attention bands: the WARN-only bands (idle power, A-B by
phase, CV) are 4090-observed priors, not pass/fail criteria. A100 bands are
set from observed smoke values in a dated code change committed before the
measured chunks; until then, idle-power warnings on the A100 (SXM idles far
above the laptop band) are expected and benign.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a plain script: ensure src/ is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tomltransformers.sweep import run_sweep, PreflightError  # noqa: E402

# Untracked-only exemption for this run's own outputs (crash-resume before the
# first harvest commit). Repo-relative "dir/" prefix; tracked changes still
# refuse. See provenance.preflight and the module docstring above.
_A100_OUTPUT_PREFIX = "experiments/exp_002_size_sweep/a100/"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the EXP-002 A100 cross-platform sweep (frozen 98-point grid).")
    ap.add_argument("--config", default=os.path.join(_HERE, "..", "configs",
                                                     "exp_002_a100.yaml"),
                    help="path to the frozen A100 multi-pass grid config")
    ap.add_argument("--run-dir", default=os.path.join(_HERE, "..", "experiments",
                                                      "exp_002_size_sweep", "a100"),
                    help="directory for results + provenance")
    ap.add_argument("--results-filename", default="energy.jsonl")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="override the clean-git provenance gate (records the override)")
    ap.add_argument("--target-s", type=float, default=4.0,
                    help="target per-execution window length (seconds)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after ~N hours (chunked sessions; resume by re-running)")
    args = ap.parse_args()

    try:
        prog = run_sweep(
            config_path=os.path.abspath(args.config),
            run_dir=os.path.abspath(args.run_dir),
            results_filename=args.results_filename,
            allow_dirty=args.allow_dirty,
            allow_untracked_paths=[_A100_OUTPUT_PREFIX],
            target_s=args.target_s,
            max_hours=args.max_hours,
        )
    except PreflightError as exc:
        print(f"\nPRE-FLIGHT FAILED:\n{exc}\n", file=sys.stderr)
        print("Commit your work (or pass --allow-dirty to override) and re-run.",
              file=sys.stderr)
        return 2

    print("\n=== EXP-002 A100 sweep summary ===")
    for k, v in prog.as_dict().items():
        print(f"  {k}: {v}")
    if prog.failed:
        print(f"\n{prog.failed} point(s) failed/skipped; re-run to retry "
              f"(ok points are not repeated).")
    print("\nNext: validate (see module docstring), commit the harvest, push, "
          "and only then terminate the instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
