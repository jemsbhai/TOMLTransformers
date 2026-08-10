"""Follow-up B correctness gate: port pretrained GPT-2 weights into our stack
and prove exact logit equivalence BEFORE any energy is measured.

These tests build a tiny GPT2LMHeadModel IN MEMORY from a GPT2Config (random
weights, no network, CPU-only), port it via workloads/port_gpt2.py, and
assert full-position logit agreement plus the mapping details. The identical
verify_port() gate runs against the real downloaded gpt2 inside the harness
at measurement time.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel

from tomltransformers.workloads.port_gpt2 import (port_gpt2_into_decoder,
                                                  verify_port)


def _tiny_hf(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=97, n_positions=64, n_embd=32, n_layer=2,
                     n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg).eval()


def test_ported_logits_match_hf_exactly():
    hf = _tiny_hf()
    dm = port_gpt2_into_decoder(hf, device="cpu", dtype_str="fp32")
    diff = verify_port(dm, hf, seq_len=16, atol=1e-4)
    assert diff <= 1e-4


def test_verify_port_catches_a_corrupted_weight():
    hf = _tiny_hf()
    dm = port_gpt2_into_decoder(hf, device="cpu", dtype_str="fp32")
    with torch.no_grad():
        dm.layers[1]["up"].weight[0, 0] += 1.0   # corrupt one element
    with pytest.raises(AssertionError, match="disagree"):
        verify_port(dm, hf, seq_len=16, atol=1e-4)


def test_mapping_transposes_conv1d_and_copies_biases():
    hf = _tiny_hf()
    dm = port_gpt2_into_decoder(hf, device="cpu", dtype_str="fp32")
    sd = hf.state_dict()
    assert torch.allclose(dm.layers[0]["qkv"].weight,
                          sd["transformer.h.0.attn.c_attn.weight"].t())
    assert torch.allclose(dm.layers[0]["qkv"].bias,
                          sd["transformer.h.0.attn.c_attn.bias"])
    assert torch.allclose(dm.layers[0]["down"].weight,
                          sd["transformer.h.0.mlp.c_proj.weight"].t())
    assert torch.allclose(dm.embed.weight, sd["transformer.wte.weight"])
    assert torch.allclose(dm.wpe.weight, sd["transformer.wpe.weight"])
    assert torch.allclose(dm.lm_head.weight, sd["lm_head.weight"])


def test_variant_flags_do_not_change_default_stack():
    """The Follow-up B flags must leave the sweep's default stack untouched:
    no bias tensors, no wpe module, identical parameter names."""
    from tomltransformers.architectures.configs import get as get_config
    from tomltransformers.workloads.decoder import _DecoderModel

    cfg = get_config("DistilGPT2")
    dm = _DecoderModel(cfg, "fp32", "cpu")
    assert dm.wpe is None
    assert dm.layers[0]["qkv"].bias is None
    assert dm.lm_head.bias is None
    names = {n for n, _ in dm.module.named_parameters()}
    assert not any("wpe" in n for n in names)


def test_decode_step_positions_flow_through_wpe():
    """decode_step with pos_offset must consume the position embedding at the
    right index (only when use_wpe): shifting the offset changes the logits."""
    hf = _tiny_hf()
    dm = port_gpt2_into_decoder(hf, device="cpu", dtype_str="fp32")
    ids = torch.randint(0, 97, (1, 8))
    tok = torch.randint(0, 97, (1, 1))
    with torch.no_grad():
        c1 = dm.prefill_into_cache(ids, attn_kind="eager")
        a = dm.decode_step(tok, c1, attn_kind="eager", pos_offset=8)
        c2 = dm.prefill_into_cache(ids, attn_kind="eager")
        b = dm.decode_step(tok, c2, attn_kind="eager", pos_offset=20)
    assert not torch.allclose(a, b)
