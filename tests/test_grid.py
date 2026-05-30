"""Tests for the grid expander (tomltransformers.sweep.grid).

Pure / CPU only. A small inline config exercises the enumeration logic; a second
test loads the REAL frozen configs/exp_002.yaml to catch schema drift between the
expander and the pre-registration. The enc-dec two-sub-sweep (source-varies vs
target-varies) is checked explicitly because it is the architecture's key claim.
"""

import os

import pytest

from tomltransformers.sweep import grid as gr
from tomltransformers.sweep import PointSpec


# A compact config mirroring the real schema, with names that resolve in the
# config registry (so _arch_of / _is_vision use the real registry).
_CFG = {
    "models": {
        "decoder_only": ["GPT-2"],
        "encoder_only": ["BERT-base", "ViT-B/16"],
        "encoder_decoder": ["T5-base"],
    },
    "precisions": ["fp16", "fp32"],
    "batch_size": 1,
    "workloads": {
        "prefill": {"seq_lens": [128, 512]},
        "decode": {"context_lens": [128, 512], "tokens_per_window": 64},
    },
    "attention_compare": {
        "models": ["GPT-2"], "kinds": ["eager", "flash"],
        "seq_lens": [512, 1024], "precision": "fp16",
    },
}


def _by(specs, **kw):
    out = specs
    for k, v in kw.items():
        out = [s for s in out if getattr(s, k) == v]
    return out


def test_decoder_prefill_and_decode_counts():
    g = gr.expand_grid(_CFG, include_attention_compare=False)
    dec = _by(g, arch="decoder_only")
    # 1 model x 2 precisions x 2 prefill seqs = 4 prefill points.
    assert len(_by(dec, phase="prefill")) == 4
    # 1 model x 2 precisions x 2 decode ctxs = 4 decode points.
    assert len(_by(dec, phase="decode")) == 4
    # decode points carry tgt_ctx == seq_len (== context) and K.
    for s in _by(dec, phase="decode"):
        assert s.tgt_ctx == s.seq_len and s.decode_tokens == 64
        assert s.decode_mode == "growing"


def test_vision_encoder_dedups_to_one_per_precision():
    g = gr.expand_grid(_CFG, include_attention_compare=False)
    vit = _by(g, model="ViT-B/16", phase="encode")
    # ViT ignores seq_len -> one encode point per precision (2), not per seq_len.
    assert len(vit) == 2
    assert all(s.seq_len is None for s in vit)

    bert = _by(g, model="BERT-base", phase="encode")
    # text encoder sweeps seq_lens: 2 precisions x 2 seqs = 4.
    assert len(bert) == 4
    assert all(s.seq_len in (128, 512) for s in bert)


def test_enc_dec_has_all_three_phases():
    g = gr.expand_grid(_CFG, include_attention_compare=False)
    ed = _by(g, arch="encoder_decoder")
    assert len(_by(ed, phase="encode")) == 4          # 2 prec x 2 src
    assert len(_by(ed, phase="decoder_prefill")) == 4  # 2 prec x 2 src
    # decode: TWO sub-sweeps x 2 prec x 2 ctxs = 8.
    assert len(_by(ed, phase="decode")) == 8


def test_enc_dec_decode_two_sub_sweeps_isolate_source_and_target():
    """The key enc-dec design: one sub-sweep varies SOURCE with target fixed at
    the anchor; the other varies TARGET with source fixed at the anchor."""
    g = gr.expand_grid(_CFG, include_attention_compare=False, enc_dec_anchor=1024)
    dec = _by(g, arch="encoder_decoder", phase="decode", precision="fp16")
    assert len(dec) == 4   # 2 (source sweep) + 2 (target sweep)

    # source sweep: tgt_ctx pinned at anchor (1024), seq_len varies over ctxs.
    source_sweep = [s for s in dec if s.tgt_ctx == 1024]
    assert {s.seq_len for s in source_sweep} == {128, 512}

    # target sweep: seq_len pinned at anchor (1024), tgt_ctx varies over ctxs.
    target_sweep = [s for s in dec if s.seq_len == 1024]
    assert {s.tgt_ctx for s in target_sweep} == {128, 512}

    # the two sweeps are distinct points (no overlap collapsing them).
    assert len({s.key() for s in dec}) == 4


def test_decoder_prefill_target_anchored():
    g = gr.expand_grid(_CFG, include_attention_compare=False, enc_dec_anchor=1024)
    dp = _by(g, arch="encoder_decoder", phase="decoder_prefill")
    assert all(s.tgt_len == 1024 for s in dp)


def test_attention_compare_eager_and_flash():
    g = gr.expand_grid(_CFG, include_attention_compare=True)
    ac = [s for s in g if s.model == "GPT-2" and s.attn_kind == "eager"]
    # eager prefill points: 2 seqs (512, 1024) at fp16.
    assert len(ac) == 2
    assert all(s.phase == "prefill" and s.precision == "fp16" for s in ac)
    # flash variant exists for the same seqs (some may dedup with the main grid).
    flash_ac = [s for s in g if s.model == "GPT-2" and s.attn_kind == "flash"
                and s.seq_len in (512, 1024) and s.precision == "fp16"]
    assert len(flash_ac) >= 2


def test_all_keys_unique():
    g = gr.expand_grid(_CFG, include_attention_compare=True)
    keys = [s.key() for s in g]
    assert len(keys) == len(set(keys)), "grid must not contain duplicate keys"


def test_anchor_override_changes_fixed_dimension():
    g512 = gr.expand_grid(_CFG, include_attention_compare=False, enc_dec_anchor=512)
    dp = _by(g512, arch="encoder_decoder", phase="decoder_prefill")
    assert all(s.tgt_len == 512 for s in dp)


def test_empty_config_yields_empty_grid():
    assert gr.expand_grid({}, include_attention_compare=True) == []


# --- against the REAL frozen config (schema-drift guard) ----------------------


def _frozen_path():
    here = os.path.dirname(__file__)
    return os.path.join(here, "..", "configs", "exp_002.yaml")


@pytest.mark.skipif(not os.path.isfile(_frozen_path()), reason="frozen config not found")
def test_real_frozen_config_expands_without_error():
    pytest.importorskip("yaml")
    cfg = gr.load_config(_frozen_path())
    g = gr.expand_grid(cfg)
    assert len(g) > 0
    # all 14 models represented.
    models = {s.model for s in g}
    for m in ("DistilGPT2", "GPT-2", "GPT-2-XL", "BERT-base", "ViT-L/16",
              "T5-small", "BART-large"):
        assert m in models, f"{m} missing from expanded grid"
    # keys unique across the whole real grid.
    keys = [s.key() for s in g]
    assert len(keys) == len(set(keys))
    # every spec has a resolvable phase for its arch (smoke).
    for s in g:
        assert s.phase in ("prefill", "decode", "encode", "decoder_prefill")
