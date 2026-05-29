"""Capture a reproducibility environment snapshot to a JSON file.

Usage:
    python scripts/snapshot_env.py [output_path]

Default output path: environment.json. Cross-platform; records git commit and
dirty state, Python and OS, key package versions, and GPU info (via nvidia-smi
if available). Run this at the start of every experiment and commit the result
into that experiment's directory.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def snapshot() -> dict:
    git_status = _run(["git", "status", "--porcelain"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": (bool(git_status) if git_status is not None else None),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]),
        "cuda_via_nvcc": _run(["nvcc", "--version"]),
        "packages": {p: _pkg(p) for p in
                     ("numpy", "scipy", "torch", "transformers", "pynvml", "pytest")},
    }


def main(out_path: str) -> None:
    snap = snapshot()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    print(json.dumps(snap, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "environment.json")
