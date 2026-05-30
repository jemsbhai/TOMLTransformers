"""Tests for the sweep pre-flight provenance gate (tomltransformers.sweep.provenance).

Pure / CPU only: no torch, no GPU. We control git state by monkeypatching
git_status_porcelain so the tests are deterministic and never touch the real
repository state.
"""

import json
import os

import pytest

import tomltransformers.sweep.provenance as pf
from tomltransformers.sweep import PreflightError


def _write_cfg(tmp_path, text="grid: [a, b]\nseed: 42\n"):
    p = tmp_path / "exp.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_clean_tree_writes_env_and_freezes_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: "")  # clean
    cfg = _write_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    res = pf.preflight(run_dir, config_path=cfg)

    assert res.git_dirty is False
    assert res.overridden is False
    # environment snapshot written and parseable.
    assert os.path.isfile(res.environment_path)
    with open(res.environment_path, encoding="utf-8") as fh:
        snap = json.load(fh)
    assert "packages" in snap and "git_commit" in snap
    # config frozen verbatim under a frozen_ name.
    assert res.frozen_config_path and os.path.isfile(res.frozen_config_path)
    assert os.path.basename(res.frozen_config_path).startswith("frozen_")
    assert open(res.frozen_config_path, encoding="utf-8").read() == open(cfg, encoding="utf-8").read()


def test_dirty_tree_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain",
                        lambda cwd=None: " M src/foo.py\n?? bar.py")
    cfg = _write_cfg(tmp_path)
    with pytest.raises(PreflightError, match="dirty"):
        pf.preflight(str(tmp_path / "run"), config_path=cfg)


def test_dirty_tree_allowed_with_override(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: " M src/foo.py")
    cfg = _write_cfg(tmp_path)
    res = pf.preflight(str(tmp_path / "run"), config_path=cfg, allow_dirty=True)
    assert res.git_dirty is True
    assert res.overridden is True
    assert any("dirty" in w for w in res.warnings)
    assert os.path.isfile(res.environment_path)


def test_unknown_git_refuses_without_override(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: None)  # git unavailable
    cfg = _write_cfg(tmp_path)
    with pytest.raises(PreflightError, match="cannot determine git state"):
        pf.preflight(str(tmp_path / "run"), config_path=cfg)


def test_unknown_git_allowed_with_override(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: None)
    cfg = _write_cfg(tmp_path)
    res = pf.preflight(str(tmp_path / "run"), config_path=cfg, allow_dirty=True)
    assert res.git_dirty is None
    assert any("git state unknown" in w for w in res.warnings)


def test_missing_config_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: "")
    with pytest.raises(PreflightError, match="config_path does not exist"):
        pf.preflight(str(tmp_path / "run"), config_path=str(tmp_path / "nope.yaml"))


def test_no_config_warns_but_proceeds(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: "")
    res = pf.preflight(str(tmp_path / "run"), config_path=None)
    assert res.frozen_config_path is None
    assert any("nothing frozen" in w for w in res.warnings)
    assert os.path.isfile(res.environment_path)


def test_run_dir_created_if_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "git_status_porcelain", lambda cwd=None: "")
    run_dir = tmp_path / "deep" / "nested" / "run"
    res = pf.preflight(str(run_dir), config_path=None)
    assert os.path.isdir(res.run_dir)


def test_environment_snapshot_has_expected_keys():
    snap = pf.environment_snapshot()
    for key in ("timestamp", "git_commit", "git_dirty", "python_version",
                "platform", "packages"):
        assert key in snap
    # package dict includes the libraries the experiment depends on.
    for p in ("torch", "transformers", "numpy", "scipy"):
        assert p in snap["packages"]
