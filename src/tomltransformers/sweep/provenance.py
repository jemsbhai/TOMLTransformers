"""Pre-flight provenance and the clean-git gate, run BEFORE any measurement.

The lab-runner standard: if a datapoint cannot be traced to an exact commit and a
recorded environment, it did not happen. So before the sweep measures anything,
preflight():

  1. refuses to proceed on a dirty git tree (uncommitted changes), unless an
     explicit override is passed -- so every result traces to a real commit;
  2. writes an environment snapshot (git commit, OS, Python, package versions,
     GPU) into the run directory;
  3. freezes a verbatim copy of the experiment config into the run directory, so
     the exact parameters that produced the data are stored beside it.

Untracked-output allowlist (2026-08-10, Step 6 of the A100 amendment): a run
that crashes before its first harvest commit leaves its own freshly created
output files UNTRACKED, which would block the resume launch. Callers may pass
`allow_untracked_paths` (exact repo-relative paths, or directory prefixes
ending in "/"): porcelain entries that are UNTRACKED ("??") and fall under the
allowlist are excluded from the dirty verdict and recorded as a warning. This
mirrors the representativeness harness's own-outputs exemption
(scripts/run_exp002_representativeness.py, GATE NOTE 2026-08-10). Tracked
modifications are NEVER exempted: appended-but-uncommitted data still refuses,
which is exactly the between-chunk harvest-commit ritual. The environment
snapshot's git_dirty field remains the RAW (unfiltered) state.

This module deliberately does no measurement and imports no torch: it is cheap,
testable, and safe to call repeatedly.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple


class PreflightError(RuntimeError):
    """Raised when the pre-flight provenance gate fails (e.g. dirty git tree)."""


def _run(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def git_status_porcelain(cwd: Optional[str] = None) -> Optional[str]:
    """Return `git status --porcelain` output, or None if git is unavailable.

    Empty string means a clean tree; non-empty means uncommitted changes.
    """
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, cwd=cwd,
        ).strip()
    except Exception:
        return None


def _split_allowed_untracked(
    status: str,
    allow_untracked_paths: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Split porcelain lines into (offending_lines, ignored_untracked_paths).

    A line is ignored iff it is an UNTRACKED entry ("??") whose normalized
    path (backslashes -> slashes, surrounding quotes stripped) either equals
    an allowlisted path exactly or falls under an allowlisted directory prefix
    (an entry ending in "/"). Git also reports a wholly untracked directory as
    a single "dir/" entry; the prefix match covers that form. Every other
    line, including any tracked modification under the allowlist, is
    offending.
    """
    exact: set = set()
    prefixes: List[str] = []
    for entry in allow_untracked_paths:
        norm = str(entry).replace("\\", "/")
        if norm.endswith("/"):
            prefixes.append(norm)
        else:
            exact.add(norm)

    offending: List[str] = []
    ignored: List[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if line.startswith("??"):
            path = line[2:].strip().strip('"').replace("\\", "/")
            if path in exact or any(path.startswith(p) for p in prefixes):
                ignored.append(path)
                continue
        offending.append(line)
    return offending, ignored


def environment_snapshot() -> dict:
    """Capture a reproducibility snapshot. Mirrors scripts/snapshot_env.py."""
    status = git_status_porcelain()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": (bool(status) if status is not None else None),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]),
        "cuda_via_nvcc": _run(["nvcc", "--version"]),
        "packages": {p: _pkg(p) for p in
                     ("numpy", "scipy", "torch", "transformers", "pynvml",
                      "zeus", "pyyaml", "pytest")},
    }


@dataclass
class PreflightResult:
    run_dir: str
    environment_path: str
    frozen_config_path: Optional[str]
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    overridden: bool = False
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def preflight(
    run_dir: str,
    *,
    config_path: Optional[str] = None,
    allow_dirty: bool = False,
    env_filename: str = "environment.json",
    allow_untracked_paths: Optional[Iterable[str]] = None,
) -> PreflightResult:
    """Run the pre-flight gate and write provenance into `run_dir`.

    Args:
      run_dir: directory to write provenance into (created if absent).
      config_path: experiment config to freeze verbatim into run_dir. If None,
        no config is frozen (a warning is recorded).
      allow_dirty: if False (default), raise PreflightError when the git tree has
        uncommitted changes. If True, proceed but record overridden=True and a
        warning -- the explicit escape hatch.
      env_filename: name of the snapshot file written into run_dir.
      allow_untracked_paths: repo-relative paths (or "dir/" prefixes) whose
        UNTRACKED porcelain entries are excluded from the dirty verdict; used
        for a run's own not-yet-committed outputs (crash-resume before the
        first harvest commit). Tracked modifications are never excluded. The
        exclusions are recorded as a warning; the environment snapshot keeps
        the raw git state.

    Returns a PreflightResult. Raises PreflightError on a dirty tree (unless
    allow_dirty), or if git state cannot be determined and allow_dirty is False
    (we refuse to measure when provenance is unknowable).
    """
    warnings: list = []

    status = git_status_porcelain()
    if status is None:
        # git unavailable / not a repo: provenance cannot be established.
        if not allow_dirty:
            raise PreflightError(
                "cannot determine git state (git unavailable or not a repository); "
                "refusing to measure without provenance. Pass allow_dirty=True to override."
            )
        warnings.append("git state unknown; proceeding under allow_dirty override")
        dirty = None
    else:
        offending_text = status
        if status and allow_untracked_paths:
            offending, ignored = _split_allowed_untracked(status, allow_untracked_paths)
            offending_text = "\n".join(offending)
            if ignored:
                warnings.append(
                    "ignored {} untracked run-output path(s) under the allowlist: {}".format(
                        len(ignored), ignored))
        dirty = bool(offending_text)
        if dirty and not allow_dirty:
            raise PreflightError(
                "git tree is dirty (uncommitted changes); refusing to measure so "
                "every datapoint traces to a commit. Commit your work, or pass "
                "allow_dirty=True to override.\n--- git status --porcelain ---\n"
                + offending_text
            )
        if dirty and allow_dirty:
            warnings.append("git tree dirty; proceeding under allow_dirty override")

    os.makedirs(run_dir, exist_ok=True)

    snap = environment_snapshot()
    env_path = os.path.join(run_dir, env_filename)
    with open(env_path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)

    frozen_path: Optional[str] = None
    if config_path is not None:
        if not os.path.isfile(config_path):
            raise PreflightError(f"config_path does not exist: {config_path}")
        frozen_path = os.path.join(run_dir, "frozen_" + os.path.basename(config_path))
        shutil.copy2(config_path, frozen_path)   # verbatim copy, preserve mtime
    else:
        warnings.append("no config_path provided; nothing frozen")

    return PreflightResult(
        run_dir=run_dir,
        environment_path=env_path,
        frozen_config_path=frozen_path,
        git_commit=snap.get("git_commit"),
        git_dirty=dirty,
        overridden=bool(allow_dirty),
        warnings=warnings,
    )
