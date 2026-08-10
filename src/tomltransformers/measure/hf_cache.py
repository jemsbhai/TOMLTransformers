"""Ephemeral Hugging Face cache management (promoted from the
diag_instrument_a.py pattern per the standing disk constraint).

Records which of the requested repos already exist in the HF cache BEFORE a
run, and on exit deletes only the repos this run caused to appear, never
pre-existing ones. Cleanup runs even on crash (context-manager finally).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List


def cached_model_repos() -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id for r in scan_cache_dir().repos if r.repo_type == "model"}
    except Exception:
        return set()


def delete_model_repo(repo_id: str) -> str:
    """Delete one model repo (all revisions) from the HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        revs = set()
        for repo in info.repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                revs |= {rev.commit_hash for rev in repo.revisions}
        if not revs:
            return f"{repo_id}: nothing to delete (not in cache)"
        strategy = info.delete_revisions(*revs)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        return f"{repo_id}: deleted from HF cache, freed ~{freed}"
    except Exception as exc:  # noqa: BLE001
        return f"{repo_id}: cache cleanup FAILED (delete manually): {exc!r}"


@contextmanager
def ephemeral_hf_repos(repo_ids: Iterable[str], log=print):
    """Context manager: repos in `repo_ids` that were NOT cached before entry
    are deleted from the HF cache on exit (success or crash). Pre-existing
    repos are always kept."""
    wanted = list(repo_ids)
    before = cached_model_repos()
    keep = [r for r in wanted if r in before]
    ours = [r for r in wanted if r not in before]
    for r in keep:
        log(f"[hf-cache] {r}: already cached; will be KEPT")
    for r in ours:
        log(f"[hf-cache] {r}: not cached; will be downloaded and DELETED on exit")
    try:
        yield
    finally:
        notes: List[str] = [delete_model_repo(r) for r in ours]
        for n in notes:
            log(f"[hf-cache] {n}")
