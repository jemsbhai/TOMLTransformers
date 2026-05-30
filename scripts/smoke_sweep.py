"""THROWAWAY smoke test for the EXP-002 sweep driver on real hardware.

NOT part of the experiment. Runs a tiny 3-point explicit PointSpec list through
the real driver (real measure_single_point, real GPU) to confirm, before the
multi-hour full sweep, that:
  1. the driver's happy path works on a fast forward point (DistilGPT2 prefill);
  2. the noisy decode regime works (10 repeats, per-token contamination flag);
  3. an OOM-candidate (GPT-2-XL fp32 eager s=2048) is caught gracefully
     (recorded as a skip, sweep continues) rather than crashing.

Writes to a SCRATCH dir (experiments/_smoke), NOT the real experiment dir, so it
never pollutes real results. Run it twice to confirm resume skips ok points.

Usage (PowerShell):
    python scripts/smoke_sweep.py
    python scripts/smoke_sweep.py            # 2nd run: should skip the ok points
    python scripts/smoke_sweep.py --allow-dirty
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tomltransformers.sweep import run_sweep, PointSpec, PreflightError  # noqa: E402


def smoke_points():
    return [
        # 1. fast forward point: proven in the single-point test; happy path.
        PointSpec(model="DistilGPT2", arch="decoder_only", phase="prefill",
                  seq_len=256, precision="fp16", attn_kind="flash"),
        # 2. decode (noisy regime): 10 repeats, per-token contamination flag.
        PointSpec(model="DistilGPT2", arch="decoder_only", phase="decode",
                  seq_len=512, tgt_ctx=512, decode_tokens=64, precision="fp16",
                  attn_kind="flash", decode_mode="growing"),
        # 3. OOM candidate: GPT-2-XL fp32 eager attention at s=2048.
        #    ~6 GB weights + a materialized s=2048 score matrix on a 16 GB card.
        #    If it fits, fine; if it OOMs, we confirm skip-and-log works.
        PointSpec(model="GPT-2-XL", arch="decoder_only", phase="prefill",
                  seq_len=2048, precision="fp32", attn_kind="eager"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Throwaway smoke test for the sweep driver.")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--run-dir", default=os.path.join(_HERE, "..", "experiments", "_smoke"))
    args = ap.parse_args()

    try:
        prog = run_sweep(
            points=smoke_points(),
            run_dir=os.path.abspath(args.run_dir),
            results_filename="smoke.jsonl",
            allow_dirty=args.allow_dirty,
            target_s=4.0,
        )
    except PreflightError as exc:
        print(f"\nPRE-FLIGHT FAILED:\n{exc}\n", file=sys.stderr)
        return 2

    print("\n=== smoke summary ===")
    for k, v in prog.as_dict().items():
        print(f"  {k}: {v}")
    print(f"\nInspect: {os.path.join(os.path.abspath(args.run_dir), 'smoke.jsonl')}")
    print("Run again to confirm resume skips the ok points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
