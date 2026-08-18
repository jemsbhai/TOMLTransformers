"""Locks fit/a100_strata.py to the frozen A100 enumeration and dataset.

The strata replay must reproduce expand_passes' enumeration exactly (same
keys, same first-occurrence dedup) and pin the amendment's per-stratum
sizes: shared 84 (66 + 12 + 6), extension 10, spot 4. Pure CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tomltransformers.fit.a100_calibration import (T2_CALIBRATION_KEYS,
                                                   T2_SENSITIVITY_ALTERNATES,
                                                   seedless_key)
from tomltransformers.fit.a100_strata import (key_to_stratum,
                                              partition_records,
                                              strata_by_pass,
                                              stratum_of_pass)
from tomltransformers.fit.bridge import load_latest_records
from tomltransformers.sweep.grid import load_config
from tomltransformers.sweep.grid_passes import expand_passes

_ROOT = Path(__file__).resolve().parents[1]
_A100_YAML = _ROOT / "configs" / "exp_002_a100.yaml"
_A100_DATA = _ROOT / "experiments" / "exp_002_size_sweep" / "a100" / "energy.jsonl"


def _cfg():
    return load_config(str(_A100_YAML))


def test_pass_sizes_match_the_amendment_strata():
    sizes = {name: len(keys) for name, keys in strata_by_pass(_cfg()).items()}
    assert sizes == {
        "shared_main": 66,
        "shared_enc_dec_fp16": 12,
        "shared_enc_dec_fp32_anchor": 6,
        "extension_7b": 10,
        "spot_random": 1,
        "spot_random_v": 1,
        "spot_ported": 1,
        "spot_pretrained_hf": 1,
    }


def test_replay_reproduces_expand_passes_exactly():
    cfg = _cfg()
    replay = [k for keys in strata_by_pass(cfg).values() for k in keys]
    frozen = [seedless_key(ps.key()) for ps in expand_passes(cfg)]
    assert replay == frozen               # same keys, same order, no dups
    assert len(set(replay)) == 98


def test_key_to_stratum_partitions_98_into_84_10_4():
    k2s = key_to_stratum(_cfg())
    assert len(k2s) == 98
    counts = {s: sum(1 for v in k2s.values() if v == s) for s in ("shared", "extension", "spot")}
    assert counts == {"shared": 84, "extension": 10, "spot": 4}


def test_stratum_of_pass_prefixes():
    assert stratum_of_pass("shared_main") == "shared"
    assert stratum_of_pass("extension_7b") == "extension"
    assert stratum_of_pass("spot_ported") == "spot"
    with pytest.raises(ValueError):
        stratum_of_pass("holdout_x")


def test_calibration_keys_are_all_shared():
    k2s = key_to_stratum(_cfg())
    for k in T2_CALIBRATION_KEYS + T2_SENSITIVITY_ALTERNATES:
        assert k2s[k] == "shared", k


def test_partition_of_the_measured_dataset():
    records = load_latest_records(_A100_DATA)
    parts = partition_records(records, _cfg())
    assert {s: len(v) for s, v in parts.items()} == {"shared": 84, "extension": 10, "spot": 4}
    assert all(r["ok"] for part in parts.values() for r in part)
    ext_models = {r["spec"]["model"] for r in parts["extension"]}
    assert ext_models == {"LLaMA-7B", "Mistral-7B"}
    spot_arms = {r["spec"]["weights"] for r in parts["spot"]}
    assert spot_arms == {"random", "random_v", "ported", "pretrained"}
    assert all(r["spec"]["weights"] == "random" for r in parts["shared"])
    assert all(r["spec"]["precision"] == "fp16" for r in parts["extension"] + parts["spot"])


def test_partition_rejects_a_key_outside_the_enumeration():
    records = load_latest_records(_A100_DATA)
    bad = dict(records[0])
    bad["spec"] = dict(bad["spec"], key="decoder_only|GPT-2|prefill|fp16|flash|random|s777|b1|seed1")
    with pytest.raises(ValueError, match="not in the frozen A100 enumeration"):
        partition_records([bad], _cfg())
