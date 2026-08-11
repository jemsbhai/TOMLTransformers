"""Tests for the preflight untracked-output allowlist (Step 6, A100 amendment).

Same approach as test_preflight.py: git state is controlled by monkeypatching
git_status_porcelain, so the tests are deterministic and never touch the real
repository. The allowlist exempts ONLY untracked ("??") entries under the
allowed paths; tracked modifications always refuse (the between-chunk
harvest-commit ritual is preserved).
"""

import pytest

import tomltransformers.sweep.provenance as pf
from tomltransformers.sweep import PreflightError
from tomltransformers.sweep.provenance import _split_allowed_untracked

_A100 = "experiments/exp_002_size_sweep/a100/"


def _pre(tmp_path, **kw):
    return pf.preflight(str(tmp_path / "run"), config_path=None, **kw)


def test_untracked_output_under_prefix_is_exempt(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain",
                        lambda cwd=None: "?? experiments/exp_002_size_sweep/a100/energy.jsonl")
    res = _pre(tmp_path, allow_untracked_paths=[_A100])
    assert res.git_dirty is False
    assert res.overridden is False
    assert any("allowlist" in w for w in res.warnings)


def test_untracked_directory_entry_is_exempt(tmp_path, monkeypatch):
    # A wholly untracked directory appears as a single "dir/" porcelain entry.
    monkeypatch.setattr(pf, "git_status_porcelain",
                        lambda cwd=None: "?? experiments/exp_002_size_sweep/a100/")
    res = _pre(tmp_path, allow_untracked_paths=[_A100])
    assert res.git_dirty is False


def test_exact_path_entries_are_exempt(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: "?? out/report.txt")
    res = _pre(tmp_path, allow_untracked_paths=["out/report.txt"])
    assert res.git_dirty is False


def test_backslash_paths_are_normalized(tmp_path, monkeypatch):
    # Defensive parity with the representativeness harness: a quoted porcelain
    # entry with backslash separators normalizes to the slash form.
    line = '?? "experiments\\exp_002_size_sweep\\a100\\energy.jsonl"'
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: line)
    res = _pre(tmp_path, allow_untracked_paths=[_A100])
    assert res.git_dirty is False


def test_untracked_outside_allowlist_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: "?? stray.py")
    with pytest.raises(PreflightError, match="dirty"):
        _pre(tmp_path, allow_untracked_paths=[_A100])


def test_tracked_modification_under_allowlist_still_refuses(tmp_path, monkeypatch):
    # Appended-but-uncommitted data must force the harvest commit.
    monkeypatch.setattr(pf, "git_status_porcelain",
                        lambda cwd=None: " M experiments/exp_002_size_sweep/a100/energy.jsonl")
    with pytest.raises(PreflightError, match="dirty"):
        _pre(tmp_path, allow_untracked_paths=[_A100])


def test_mixed_dirt_refuses_and_reports_only_offenders(tmp_path, monkeypatch):
    status = ("?? experiments/exp_002_size_sweep/a100/energy.jsonl\n"
              " M src/tomltransformers/sweep/driver.py")
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: status)
    with pytest.raises(PreflightError) as ei:
        _pre(tmp_path, allow_untracked_paths=[_A100])
    msg = str(ei.value)
    assert "driver.py" in msg
    assert "energy.jsonl" not in msg


def test_allow_dirty_override_still_works_with_allowlist(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain",
                        lambda cwd=None: " M src/foo.py\n?? experiments/exp_002_size_sweep/a100/x")
    res = _pre(tmp_path, allow_dirty=True, allow_untracked_paths=[_A100])
    assert res.git_dirty is True
    assert res.overridden is True


def test_no_allowlist_behavior_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain",
                        lambda cwd=None: "?? experiments/exp_002_size_sweep/a100/energy.jsonl")
    with pytest.raises(PreflightError, match="dirty"):
        _pre(tmp_path)


def test_split_helper_partitions_lines():
    status = ("?? experiments/exp_002_size_sweep/a100/energy.jsonl\n"
              "?? stray.py\n"
              " M src/mod.py")
    offending, ignored = _split_allowed_untracked(status, [_A100])
    assert ignored == ["experiments/exp_002_size_sweep/a100/energy.jsonl"]
    assert len(offending) == 2
    assert any("stray.py" in line for line in offending)
    assert any("src/mod.py" in line for line in offending)
