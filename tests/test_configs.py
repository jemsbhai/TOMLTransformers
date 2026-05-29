"""Tests for the model zoo (tomltransformers.architectures.configs)."""

import pytest

from tomltransformers.architectures import configs as cf


def test_registry_complete():
    for name in ("LLaMA-7B", "Mistral-7B", "GPT-2", "BERT-base", "ViT-B/16",
                 "T5-base", "BART-base", "LLaMA-3-8B", "BERT-large",
                 "DistilGPT2", "GPT-2-medium", "GPT-2-large", "GPT-2-XL",
                 "DistilBERT", "ViT-L/16", "T5-small", "BART-large"):
        assert name in cf.MODELS
    assert cf.get("GPT-2").name == "GPT-2"
    with pytest.raises(KeyError):
        cf.get("nope")


def test_resolved_properties():
    assert cf.LLAMA_7B.d_head == 128
    assert cf.LLAMA_7B.kv_heads == 32
    assert cf.MISTRAL_7B.kv_heads == 8        # GQA
    assert cf.GPT2.d_head == 64
    assert cf.LLAMA_7B.is_gated and not cf.GPT2.is_gated
    assert cf.LLAMA_7B.is_causal and cf.LLAMA_7B.has_kv_cache
    assert not cf.BERT_BASE.is_causal and not cf.BERT_BASE.has_kv_cache
    assert cf.T5_BASE.has_kv_cache   # decoder stack decodes autoregressively


def test_param_counts_match_known_totals():
    # Within ~12% of published parameter counts.
    def close(actual, expected, tol=0.12):
        return abs(actual - expected) / expected < tol
    assert close(cf.LLAMA_7B.param_count, 6.74e9)
    assert close(cf.MISTRAL_7B.param_count, 7.24e9)
    assert close(cf.GPT2.param_count, 124e6)
    assert close(cf.BERT_BASE.param_count, 110e6)
    assert close(cf.VIT_B16.param_count, 86e6)
    # EXP-002 zoo expansion (published totals).
    assert close(cf.DISTILGPT2.param_count, 82e6)
    assert close(cf.GPT2_MEDIUM.param_count, 355e6)
    assert close(cf.GPT2_LARGE.param_count, 774e6)
    assert close(cf.GPT2_XL.param_count, 1558e6)
    assert close(cf.DISTILBERT.param_count, 66e6)
    assert close(cf.VIT_L16.param_count, 304e6)
    assert close(cf.T5_SMALL.param_count, 60.5e6)
    assert close(cf.BART_LARGE.param_count, 406e6)


def test_validation_rejects_bad_configs():
    with pytest.raises(ValueError):
        cf.TransformerConfig(name="bad", arch="seq2seq", d_model=768, d_ff=3072,
                             n_heads=12, vocab_size=100, n_layers=12)
    with pytest.raises(ValueError):  # d_model not divisible by heads, no head_dim
        cf.TransformerConfig(name="bad", arch="decoder_only", d_model=100, d_ff=400,
                             n_heads=12, vocab_size=100, n_layers=2)
    with pytest.raises(ValueError):  # enc-dec without layer split
        cf.TransformerConfig(name="bad", arch="encoder_decoder", d_model=768, d_ff=3072,
                             n_heads=12, head_dim=64, vocab_size=100)


def test_encoder_decoder_has_more_params_than_single_stack_equiv():
    # T5-base (12+12 layers + cross-attn) should exceed a 12-layer encoder of same width.
    assert cf.T5_BASE.param_count > cf.BERT_BASE.param_count
