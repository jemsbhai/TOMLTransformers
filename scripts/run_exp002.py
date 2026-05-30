"""Run the EXP-002 transformer-energy sweep.

Usage (PowerShell):
    python scripts/run_exp002.py
    python scripts/run_exp002.py --allow-dirty           # override the git gate
    python scripts/run_exp002.py --no-attention-compare  # skip the eager/flash sub-sweep
    python scripts/run_exp002.py --run-dir experiments/exp_002_size_sweep

The sweep is RESUMABLE: re-running continues from where it left off (points whose
latest record is ok=True are skipped; failures and short-window points are re-run).
Output is appended to <run_dir>/energy.jsonl as one JSON record per line.

Pre-flight HARD-REFUSES a dirty git tree unless --allow-dirty is passed, so every
datapoint traces to a commit.
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the EXP-002 transformer-energy sweep.")
    ap.add_argument("--config", default=os.path.join(_HERE, "..", "configs", "exp_002.yaml"),
                    help="path to the frozen EXP-002 config")
    ap.add_argument("--run-dir", default=os.path.join(_HERE, "..", "experiments",
                                                       "exp_002_size_sweep"),
                    help="directory for results + provenance")
    ap.add_argument("--results-filename", default="energy.jsonl")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="override the clean-git provenance gate (records the override)")
    ap.add_argument("--no-attention-compare", action="store_true",
                    help="skip the eager-vs-flash attention sub-sweep")
    ap.add_argument("--enc-dec-anchor", type=int, default=1024,
                    help="held-constant dimension for enc-dec source/target sub-sweeps")
    ap.add_argument("--target-s", type=float, default=4.0,
                    help="target per-execution window length (seconds)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after ~N hours (for nightly chunks; resume by re-running)")
    args = ap.parse_args()

    try:
        prog = run_sweep(
            config_path=os.path.abspath(args.config),
            run_dir=os.path.abspath(args.run_dir),
            results_filename=args.results_filename,
            allow_dirty=args.allow_dirty,
            enc_dec_anchor=args.enc_dec_anchor,
            include_attention_compare=not args.no_attention_compare,
            target_s=args.target_s,
            max_hours=args.max_hours,
        )
    except PreflightError as exc:
        print(f"\nPRE-FLIGHT FAILED:\n{exc}\n", file=sys.stderr)
        print("Commit your work (or pass --allow-dirty to override) and re-run.",
              file=sys.stderr)
        return 2

    print("\n=== EXP-002 sweep summary ===")
    for k, v in prog.as_dict().items():
        print(f"  {k}: {v}")
    if prog.failed:
        print(f"\n{prog.failed} point(s) failed/skipped; re-run to retry "
              f"(ok points are not repeated).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
