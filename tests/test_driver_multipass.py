"""Tests for the driver's multi-pass grid dispatch (Step 6, A100 amendment).

build_points_from_config is the driver's grid-build path factored out so the
dispatch is testable without any measurement: a config carrying a `passes`
list goes to expand_passes (per-pass weights/anchors, derived seeds); a
classic single-pass config goes to expand_grid with the caller's kwargs,
byte-identical to the frozen-4090 behavior. Pure CPU (no torch, no GPU).
"""

from __future__ import annotations

from pathlib import Path

from tomltransformers.sweep.driver import build_points_from_config
from tomltransformers.sweep.grid import load_config

_ROOT = Path(__file__).resolve().parents[1]
_A100_YAML = _ROOT / "configs" / "exp_002_a100.yaml"


def _single_pass_cfg():
    return {
        "models": {"decoder_only": ["GPT-2"]},
        "precisions": ["fp16"],
        "batch_size": 1,
        "workloads": {"prefill": {"seq_lens": [128]}},
    }


def _multi_pass_cfg(pass_weights="random"):
    return {
        "seed": {"master": 42},
        "passes": [
            {
                "name": "tiny",
                "weights": pass_weights,
                "include_attention_compare": False,
                "config": _single_pass_cfg(),
            }
        ],
    }


def test_single_pass_config_uses_expand_grid_unchanged():
    points = build_points_from_config(_single_pass_cfg())
    assert len(points) == 1
    ps = points[0]
    assert ps.model == "GPT-2" and ps.phase == "prefill" and ps.seq_len == 128
    assert ps.seed is None                 # classic path assigns no seeds
    assert "seed" not in ps.key()          # frozen-4090 keys unchanged


def test_single_pass_config_respects_caller_kwargs():
    points = build_points_from_config(_single_pass_cfg(), weights="random_v")
    assert points[0].weights == "random_v"


def test_multi_pass_config_dispatches_to_expand_passes():
    points = build_points_from_config(_multi_pass_cfg())
    assert len(points) == 1
    ps = points[0]
    assert isinstance(ps.seed, int)        # derived seed assigned
    assert f"seed{ps.seed}" in ps.key()


def test_multi_pass_ignores_caller_kwargs_in_favor_of_per_pass_settings():
    points = build_points_from_config(_multi_pass_cfg(pass_weights="random"),
                                      weights="ported")
    assert points[0].weights == "random"   # the pass, not the kwarg, decides


def test_frozen_a100_yaml_builds_98_points_through_the_driver_path():
    cfg = load_config(str(_A100_YAML))
    points = build_points_from_config(cfg)
    assert len(points) == 98
    assert all(ps.seed is not None for ps in points)
