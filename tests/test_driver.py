"""Tests for the sweep driver (tomltransformers.sweep.driver).

Pure / CPU only: measure_single_point and preflight are monkeypatched, so the
driver's orchestration is tested without GPU or real measurement. We verify:
resume semantics (skip only ok=True; re-run failures/short windows via
last-write-wins), regime-aware repeats, crash-safe JSONL append, and that a dirty
git tree is refused unless overridden.
"""

import json
import os

import pytest

from tomltransformers.sweep import driver as dv
from tomltransformers.sweep import PointSpec, PreflightError
from tomltransformers.sweep import provenance as prov


def _clean_git(monkeypatch):
    monkeypatch.setattr(prov, "git_status_porcelain", lambda cwd=None: "")


def _fake_record(ps: PointSpec, *, ok=True, oom=False, short=False, b=100.0):
    return {
        "schema": "tomltransformers.sweep.point.v1",
        "spec": {"key": ps.key(), "model": ps.model, "phase": ps.phase},
        "ok": ok, "oom_skipped": oom, "short_window": short,
        "per_execution_j": {"B": b} if ok else {},
        "per_unit_j": {"B": b / 10} if ok else {},
    }


def _patch_measure(monkeypatch, recorder=None, **record_kw):
    """Patch measure_single_point; optionally record the repeats it was called with."""
    def fake(ps, **kw):
        if recorder is not None:
            recorder.append((ps.key(), kw.get("repeats")))
        return _fake_record(ps, **record_kw)
    monkeypatch.setattr(dv, "measure_single_point", fake)


_PTS = [
    PointSpec(model="GPT-2", arch="decoder_only", phase="prefill", seq_len=128),
    PointSpec(model="GPT-2", arch="decoder_only", phase="decode", seq_len=128,
              tgt_ctx=128, decode_tokens=64),
]


def test_runs_all_points_and_appends_jsonl(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    _patch_measure(monkeypatch, ok=True)
    run_dir = str(tmp_path / "run")
    prog = dv.run_sweep(points=list(_PTS), run_dir=run_dir, log=False)
    assert prog.total == 2 and prog.measured == 2 and prog.ok == 2
    # JSONL has one line per point.
    results = os.path.join(run_dir, "energy.jsonl")
    lines = [l for l in open(results, encoding="utf-8").read().splitlines() if l]
    assert len(lines) == 2
    for l in lines:
        json.loads(l)  # each line is valid JSON


def test_regime_aware_repeats(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    rec = []
    _patch_measure(monkeypatch, recorder=rec, ok=True)
    dv.run_sweep(points=list(_PTS), run_dir=str(tmp_path / "run"), log=False)
    repeats = dict(rec)
    # prefill -> 5, decode -> 10.
    assert repeats[_PTS[0].key()] == 5
    assert repeats[_PTS[1].key()] == 10


def test_resume_skips_only_ok_true(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    run_dir = str(tmp_path / "run")
    results = os.path.join(run_dir, "energy.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    # Pre-seed: point 0 done OK, point 1 recorded as FAILED.
    with open(results, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_fake_record(_PTS[0], ok=True)) + "\n")
        fh.write(json.dumps(_fake_record(_PTS[1], ok=False)) + "\n")

    measured = []
    _patch_measure(monkeypatch, recorder=measured, ok=True)
    prog = dv.run_sweep(points=list(_PTS), run_dir=run_dir, log=False)

    # Only the previously-failed point 1 is re-measured; point 0 (ok) is skipped.
    assert prog.skipped_done == 1
    assert prog.measured == 1
    assert [k for k, _ in measured] == [_PTS[1].key()]


def test_last_write_wins_supersedes_failure(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    run_dir = str(tmp_path / "run")
    results = os.path.join(run_dir, "energy.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    # point 0 first failed, then later succeeded (two records, same key).
    with open(results, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_fake_record(_PTS[0], ok=False)) + "\n")
        fh.write(json.dumps(_fake_record(_PTS[0], ok=True)) + "\n")
    done = dv.read_done_keys(results)
    # latest record is ok -> counts as done.
    assert _PTS[0].key() in done


def test_ok_but_short_window_is_rerun(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    run_dir = str(tmp_path / "run")
    results = os.path.join(run_dir, "energy.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    # ok=True BUT short_window: a short window leaves even the hardware counter in
    # its unreliable regime, so it is treated as needs-retry, NOT done.
    with open(results, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_fake_record(_PTS[0], ok=True, short=True)) + "\n")
    done = dv.read_done_keys(results)
    assert _PTS[0].key() not in done   # ok=True + short_window -> re-run, not done

    # and the driver actually re-measures it.
    measured = []
    _patch_measure(monkeypatch, recorder=measured, ok=True, short=False)
    prog = dv.run_sweep(points=[_PTS[0]], run_dir=run_dir, log=False)
    assert prog.measured == 1 and [k for k, _ in measured] == [_PTS[0].key()]


def test_oom_counted_and_recorded(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    _patch_measure(monkeypatch, ok=False, oom=True)
    prog = dv.run_sweep(points=[_PTS[0]], run_dir=str(tmp_path / "run"), log=False)
    assert prog.failed == 1 and prog.oom == 1


def test_dirty_git_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "git_status_porcelain", lambda cwd=None: " M x.py")
    _patch_measure(monkeypatch, ok=True)
    with pytest.raises(PreflightError, match="dirty"):
        dv.run_sweep(points=[_PTS[0]], run_dir=str(tmp_path / "run"), log=False)


def test_dirty_git_allowed_with_override(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "git_status_porcelain", lambda cwd=None: " M x.py")
    _patch_measure(monkeypatch, ok=True)
    prog = dv.run_sweep(points=[_PTS[0]], run_dir=str(tmp_path / "run"),
                        allow_dirty=True, log=False)
    assert prog.ok == 1


def test_requires_exactly_one_of_config_or_points(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        dv.run_sweep(run_dir=str(tmp_path / "run"), log=False)  # neither
    with pytest.raises(ValueError, match="exactly one"):
        dv.run_sweep(config_path="x.yaml", points=[_PTS[0]],
                     run_dir=str(tmp_path / "run"), log=False)  # both


def test_summary_file_written(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    _patch_measure(monkeypatch, ok=True)
    run_dir = str(tmp_path / "run")
    dv.run_sweep(points=[_PTS[0]], run_dir=run_dir, log=False)
    summ = os.path.join(run_dir, "sweep_summary.json")
    assert os.path.isfile(summ)
    data = json.load(open(summ, encoding="utf-8"))
    assert "progress" in data and data["progress"]["ok"] == 1
