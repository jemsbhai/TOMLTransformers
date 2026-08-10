"""Port pretrained GPT-2 weights from Hugging Face into OUR _DecoderModel.

This is the Follow-up B isolation tool (findings.md 2026-08-10): identical
weight VALUES, our implementation, so the pretrained-vs-random comparison is
implementation-free. The pre-stated predictions live in findings.md.

Mapping (GPT2LMHeadModel -> _DecoderModel with use_bias=True, use_wpe=True):
  transformer.wte.weight            -> embed.weight          (direct)
  transformer.wpe.weight            -> wpe.weight            (direct)
  h[i].ln_1.{weight,bias}           -> layers[i].norm1       (direct)
  h[i].attn.c_attn.{weight,bias}    -> layers[i].qkv         (Conv1D: weight.T)
  h[i].attn.c_proj.{weight,bias}    -> layers[i].out         (Conv1D: weight.T)
  h[i].ln_2.{weight,bias}           -> layers[i].norm2       (direct)
  h[i].mlp.c_fc.{weight,bias}       -> layers[i].up          (Conv1D: weight.T)
  h[i].mlp.c_proj.{weight,bias}     -> layers[i].down        (Conv1D: weight.T)
  transformer.ln_f.{weight,bias}    -> final_norm            (direct)
  lm_head.weight                    -> lm_head.weight        (direct; tied to wte)

HF Conv1D stores weight as [in, out]; nn.Linear stores [out, in]; hence the
transposes. GPT-2 packs c_attn outputs as [q | k | v], which matches our
fused-qkv slicing order exactly (MHA: q_dim == kv_dim == d_model).

ACTIVATION ALIGNMENT: HF GPT-2 uses activation_function="gelu_new", the tanh
approximation; our default "gelu" is the exact erf form. The porter overrides
the ported model's activation to F.gelu(approximate="tanh") when the HF
config uses a tanh-family GELU. Op count and energy are identical; this is a
numerics detail required for the exact-equivalence gate.

CORRECTNESS GATE: verify_port() asserts full logit agreement between the
ported stack and the HF model on a probe input, in fp32, before any energy
is measured. The unit tests exercise the same gate on a tiny in-memory
GPT2Config model (no download, CPU-only).

(2026-08-10 fix: the derived TransformerConfig uses the actual constructor
fields n_kv_heads / head_dim; kv_heads and d_head are resolved properties.)
"""

from __future__ import annotations

from typing import Optional


def _torch():
    import torch
    return torch


_TANH_GELU_NAMES = {"gelu_new", "gelu_pytorch_tanh", "gelu_fast"}


def port_gpt2_into_decoder(hf_model, cfg=None, *, device: str = "cpu",
                           dtype_str: str = "fp32"):
    """Copy all weights from a GPT2LMHeadModel into a fresh _DecoderModel
    (use_bias=True, use_wpe=True). Returns the ported _DecoderModel.

    cfg defaults to a TransformerConfig derived from the HF config so tiny
    test models port too; pass architectures.configs.GPT2 for the real one.
    """
    torch = _torch()
    import torch.nn.functional as F
    from ..architectures.configs import TransformerConfig
    from .decoder import _DecoderModel

    hc = hf_model.config
    if cfg is None:
        cfg = TransformerConfig(
            name=f"ported:{getattr(hc, 'name_or_path', 'gpt2')}",
            arch="decoder_only",
            d_model=hc.n_embd,
            d_ff=4 * hc.n_embd,
            n_heads=hc.n_head,
            vocab_size=hc.vocab_size,
            n_layers=hc.n_layer,
            n_kv_heads=hc.n_head,
            head_dim=hc.n_embd // hc.n_head,
            activation="gelu",
            norm_type="layernorm",
            ffn_type="standard",
            tie_embeddings=True,
            max_position=hc.n_positions,
        )
    assert cfg.n_layers == hc.n_layer and cfg.d_model == hc.n_embd, \
        f"config mismatch: ours {cfg.n_layers}x{cfg.d_model} vs HF {hc.n_layer}x{hc.n_embd}"
    assert not cfg.is_gated, "GPT-2 porting requires a standard (non-gated) FFN"

    dm = _DecoderModel(cfg, dtype_str, device, use_bias=True, use_wpe=True,
                       max_positions=hc.n_positions)

    # Activation alignment (numerics only; identical op count and energy).
    if getattr(hc, "activation_function", "gelu_new") in _TANH_GELU_NAMES:
        dm._act = lambda x: F.gelu(x, approximate="tanh")

    sd = hf_model.state_dict()

    def cp(dst, src_name: str, transpose: bool = False):
        src = sd[src_name]
        w = src.t() if transpose else src
        if dst.shape != w.shape:
            raise ValueError(f"shape mismatch {src_name}: dst {tuple(dst.shape)} "
                             f"vs src {tuple(w.shape)}")
        with torch.no_grad():
            dst.copy_(w.to(dst.dtype))

    cp(dm.embed.weight, "transformer.wte.weight")
    cp(dm.wpe.weight, "transformer.wpe.weight")
    for i, layer in enumerate(dm.layers):
        p = f"transformer.h.{i}."
        cp(layer["norm1"].weight, p + "ln_1.weight")
        cp(layer["norm1"].bias, p + "ln_1.bias")
        cp(layer["qkv"].weight, p + "attn.c_attn.weight", transpose=True)
        cp(layer["qkv"].bias, p + "attn.c_attn.bias")
        cp(layer["out"].weight, p + "attn.c_proj.weight", transpose=True)
        cp(layer["out"].bias, p + "attn.c_proj.bias")
        cp(layer["norm2"].weight, p + "ln_2.weight")
        cp(layer["norm2"].bias, p + "ln_2.bias")
        cp(layer["up"].weight, p + "mlp.c_fc.weight", transpose=True)
        cp(layer["up"].bias, p + "mlp.c_fc.bias")
        cp(layer["down"].weight, p + "mlp.c_proj.weight", transpose=True)
        cp(layer["down"].bias, p + "mlp.c_proj.bias")
    cp(dm.final_norm.weight, "transformer.ln_f.weight")
    cp(dm.final_norm.bias, "transformer.ln_f.bias")
    cp(dm.lm_head.weight, "lm_head.weight")
    return dm


def verify_port(dm, hf_model, *, seq_len: int = 16, atol: float = 1e-3,
                seed: int = 0) -> float:
    """Assert full-position logit agreement (fp32 recommended) between the
    ported stack and the HF model on one probe input. Returns max |diff|.
    Raises AssertionError on disagreement: NEVER measure energy on an
    unverified port."""
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = hf_model.config.vocab_size
    ids = torch.randint(0, vocab, (1, seq_len), generator=g).to(
        next(iter(hf_model.parameters())).device)
    with torch.no_grad():
        ref = hf_model(ids).logits
        x = dm._embed(ids)
        got = dm._attn(x, attn_kind="eager", last_token_only=False)
    diff = (got.float() - ref.float()).abs().max().item()
    assert diff <= atol, (
        f"ported logits disagree with HF: max|diff|={diff:.3e} > atol={atol}")
    return diff


def load_and_port_gpt2(hf_id: str, *, precision: str, device: str,
                       verify_atol: float = 1e-3):
    """Load HF GPT-2 in fp32, port, VERIFY in fp32, then cast the ported
    model to the target precision and free the HF model. Returns the ported
    _DecoderModel ready to measure."""
    torch = _torch()
    from transformers import AutoModelForCausalLM
    from ..architectures.configs import get as get_config

    hf = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    hf = hf.to(device).eval()
    cfg = get_config("GPT-2") if hf.config.n_layer == 12 and hf.config.n_embd == 768 else None
    dm = port_gpt2_into_decoder(hf, cfg, device=device, dtype_str="fp32")
    diff = verify_port(dm, hf, atol=verify_atol)
    del hf
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if precision == "fp16":
        dm.module = dm.module.half()
        dm.dtype = torch.float16
    return dm, diff
