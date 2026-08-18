"""Tests for the validate_exp002.py CLI surface and pairing helper.

The validator lives in scripts/ (not a package), so it is loaded by file
path. Loading executes only module-level definitions; main() is behind
the __main__ guard and never runs here.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_exp002.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_exp002", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load_validator()


def _spec(**overrides):
    """A realistic A100-style spec carrying every production schema field."""
    base = {
        "model": "GPT-2",
        "arch": "decoder_only",
        "phase": "prefill",
        "precision": "fp16",
        "attn_kind": "flash",
        "weights": "random",
        "seq_len": 512,
        "tgt_len": 1,
        "tgt_ctx": 512,
        "decode_tokens": 64,
        "decode_mode": "growing",
        "batch_size": 1,
        "seed": 1617754261,
        "key": "decoder_only|GPT-2|prefill|fp16|flash|random|s512|b1|seed1617754261",
    }
    base.update(overrides)
    return base


class TestShapePairKey:
    def test_pairs_across_precision_and_seed(self):
        # The A100 failure mode: per-point seeds are derived from the
        # precision-inclusive key, so the fp32 twin of a shape carries a
        # different seed and a different key. The pair key must match.
        fp16 = _spec()
        fp32 = _spec(
            precision="fp32",
            seed=903214577,
            key="decoder_only|GPT-2|prefill|fp32|flash|random|s512|b1|seed903214577",
        )
        assert V.shape_pair_key(fp16) == V.shape_pair_key(fp32)

    def test_different_shape_does_not_pair(self):
        a = _spec()
        b = _spec(
            seq_len=1024,
            tgt_ctx=1024,
            key="decoder_only|GPT-2|prefill|fp16|flash|random|s1024|b1|seed17",
        )
        assert V.shape_pair_key(a) != V.shape_pair_key(b)

    def test_different_model_does_not_pair(self):
        a = _spec()
        b = _spec(
            model="GPT-2-medium",
            key="decoder_only|GPT-2-medium|prefill|fp16|flash|random|s512|b1|seed23",
        )
        assert V.shape_pair_key(a) != V.shape_pair_key(b)

    def test_seedless_specs_match_legacy_pairing(self):
        # 4090-style specs carry no seed field. The pair key must equal the
        # pre-update tuple (precision and key excluded only), proving the
        # seed exclusion is a no-op on the default data.
        s = _spec()
        del s["seed"]
        s["key"] = "decoder_only|GPT-2|prefill|fp16|flash|random|s512|b1"
        legacy = tuple(sorted((k, v) for k, v in s.items()
                              if k not in ("precision", "key")))
        assert V.shape_pair_key(s) == legacy

    def test_seedless_specs_pair_across_precision(self):
        s16 = _spec()
        del s16["seed"]
        s16["key"] = "decoder_only|GPT-2|prefill|fp16|flash|random|s512|b1"
        s32 = dict(
            s16,
            precision="fp32",
            key="decoder_only|GPT-2|prefill|fp32|flash|random|s512|b1",
        )
        assert V.shape_pair_key(s16) == V.shape_pair_key(s32)


class TestIdleBandArg:
    def test_default_is_4090_band(self):
        args = V.build_arg_parser().parse_args([])
        assert args.idle_band == [V.IDLE_W_LOW, V.IDLE_W_HIGH]
        assert args.idle_band == [1.0, 15.0]

    def test_override_parses_two_floats(self):
        args = V.build_arg_parser().parse_args(["--idle-band", "40", "90"])
        assert args.idle_band == [40.0, 90.0]

    def test_other_defaults_unchanged(self):
        args = V.build_arg_parser().parse_args([])
        assert args.data == V.DEFAULT_DATA
        assert args.summary == V.DEFAULT_SUMMARY
        assert args.json_out == V.DEFAULT_JSON
        assert args.txt_out == V.DEFAULT_TXT
