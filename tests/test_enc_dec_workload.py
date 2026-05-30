"""Tests for the encoder-decoder runnable workloads (tomltransformers.workloads).

CPU structural tests check the runnable model mirrors encoder_decoder.py: the
decoder layer has THREE norms (self/cross/ffn) and SEPARATE cross-attention K/V
projections; the cross-attention cache is STATIC (built once from the encoder
output, unchanged by decode steps) while the self-attention cache GROWS with the
target; all three phases (encode / decoder_prefill / decode) build and execute.
One GPU integration test runs a decoder_prefill through the controlled runner.
"""

import pytest

from tomltransformers.architectures.configs import TransformerConfig
from tomltransformers.measure import instruments as ins
from tomltransformers import workloads as wl


def _tiny(gated=False, n_kv=None, n_heads=4):
    """A tiny but shape-valid encoder-decoder config (T5/BART-like)."""
    return TransformerConfig(
        name="tiny-ed", arch="encoder_decoder",
        n_encoder_layers=2, n_decoder_layers=2,
        d_model=32, d_ff=64, n_heads=n_heads,
        n_kv_heads=(n_kv if n_kv is not None else n_heads),
        head_dim=8, vocab_size=128,
        activation=("silu" if gated else "gelu"),
        norm_type=("rmsnorm" if gated else "layernorm"),
        ffn_type=("gated" if gated else "standard"),
        max_position=64,
    )


# --- spec / interface ---------------------------------------------------------


def test_spec_and_label():
    w = wl.build_enc_dec_workload(_tiny(), phase="encode", src_len=16, precision="fp32")
    assert w.spec.arch == "encoder_decoder"
    assert w.spec.phase == "encode"
    assert w.spec.seq_len == 16
    assert "tiny-ed" in w.spec.label() and "encode" in w.spec.label()
    w.free()


def test_workload_satisfies_protocol():
    w = wl.build_enc_dec_workload(_tiny(), phase="encode", src_len=8, precision="fp32")
    assert isinstance(w, wl.Workload)
    w.free()


def test_rejects_non_enc_dec_config():
    enc = TransformerConfig(name="e", arch="encoder_only", n_layers=2, d_model=32,
                            d_ff=64, n_heads=4, head_dim=8, vocab_size=64)
    with pytest.raises(ValueError, match="not encoder_decoder"):
        wl.build_enc_dec_workload(enc, phase="encode", src_len=8)


def test_rejects_bad_phase_and_precision():
    with pytest.raises(ValueError, match="phase"):
        wl.build_enc_dec_workload(_tiny(), phase="generate", src_len=8)
    with pytest.raises(ValueError, match="precision"):
        wl.build_enc_dec_workload(_tiny(), phase="encode", src_len=8, precision="fp8")


def test_all_three_phases_build():
    for phase in ("encode", "decoder_prefill", "decode"):
        w = wl.build_enc_dec_workload(_tiny(), phase=phase, src_len=8, tgt_len=4,
                                      tgt_ctx=6, decode_tokens=3, precision="fp32")
        assert w.spec.phase == phase
        w.free()


# --- structural fidelity to architectures/encoder_decoder.py (CPU) ------------


def test_decoder_layer_has_three_norms_and_cross_projections():
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    m = _EncDecModel(_tiny(), "float32", "cpu")
    layer = m.dec_layers[0]
    # three norms: self-attn, cross-attn, ffn.
    assert "sa_norm" in layer and "ca_norm" in layer and "ffn_norm" in layer
    # separate cross-attention Q, K, V projections.
    assert "ca_q" in layer and "ca_k" in layer and "ca_v" in layer and "ca_out" in layer
    # self-attention fused QKV.
    assert "sa_qkv" in layer and "sa_out" in layer


def test_encoder_layer_has_two_norms():
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    m = _EncDecModel(_tiny(), "float32", "cpu")
    layer = m.enc_layers[0]
    assert "norm1" in layer and "norm2" in layer
    assert "qkv" in layer and "out" in layer


def test_layer_counts_match_config():
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    cfg = _tiny()
    m = _EncDecModel(cfg, "float32", "cpu")
    assert len(m.enc_layers) == cfg.n_encoder_layers
    assert len(m.dec_layers) == cfg.n_decoder_layers


def test_cross_kv_projection_widths_are_gqa_aware():
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    # 4 q heads, 2 kv heads, head_dim 8: ca_q -> 4*8=32, ca_k/ca_v -> 2*8=16.
    m = _EncDecModel(_tiny(n_heads=4, n_kv=2), "float32", "cpu")
    layer = m.dec_layers[0]
    assert layer["ca_q"].out_features == 4 * 8
    assert layer["ca_k"].out_features == 2 * 8
    assert layer["ca_v"].out_features == 2 * 8


# --- the thesis-critical property: cross-cache static, self-cache grows -------


def test_cross_cache_is_static_self_cache_grows():
    """Cross-attention K/V are projected ONCE from the encoder output and stay at
    src_len; self-attention K/V grow one position per decode step. This asymmetry
    (constant cross-cache vs growing self-cache) is the structural heart of the
    enc-dec decode memory pattern the thesis distinguishes."""
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    cfg = _tiny()
    m = _EncDecModel(cfg, "float32", "cpu")
    src_len, tgt_ctx = 10, 5
    src = torch.randint(0, cfg.vocab_size, (1, src_len))
    enc_out = m.encode(src, attn_kind="eager")

    cross = m.build_cross_cache(enc_out)
    # cross cache holds src_len positions per layer (seq axis = dim 2).
    ck0, cv0 = cross[0]
    assert ck0.shape[2] == src_len and cv0.shape[2] == src_len

    # establish self-cache with a prefill of tgt_ctx target tokens.
    sc = m.new_self_cache()
    prompt = torch.randint(0, cfg.vocab_size, (1, tgt_ctx))
    m.decoder_prefill(prompt, cross, attn_kind="eager", self_cache=sc)
    sk0, sv0 = sc[0]
    assert sk0.shape[2] == tgt_ctx   # self-cache at the prompt length

    # one decode step: self-cache grows by 1, cross-cache UNCHANGED.
    tok = torch.randint(0, cfg.vocab_size, (1, 1))
    m.decode_step(tok, cross, sc, attn_kind="eager")
    sk1, sv1 = sc[0]
    assert sk1.shape[2] == tgt_ctx + 1, "self-cache must grow by one per step"
    ck_after, cv_after = cross[0]
    assert ck_after.shape[2] == src_len, "cross-cache must stay at src_len (static)"
    assert cv_after.shape[2] == src_len

    # a second step: self-cache 2 longer; cross still static.
    m.decode_step(tok, cross, sc, attn_kind="eager")
    assert sc[0][0].shape[2] == tgt_ctx + 2
    assert cross[0][0].shape[2] == src_len


def test_decode_step_returns_single_token_logits():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    cfg = _tiny()
    m = _EncDecModel(cfg, "float32", "cpu")
    src = torch.randint(0, cfg.vocab_size, (1, 8))
    enc_out = m.encode(src, attn_kind="eager")
    cross = m.build_cross_cache(enc_out)
    sc = m.new_self_cache()
    m.decoder_prefill(torch.randint(0, cfg.vocab_size, (1, 4)), cross,
                      attn_kind="eager", self_cache=sc)
    logits = m.decode_step(torch.randint(0, cfg.vocab_size, (1, 1)), cross, sc,
                           attn_kind="eager")
    assert tuple(logits.shape) == (1, 1, cfg.vocab_size)


def test_encode_output_shape():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder_decoder import _EncDecModel
    cfg = _tiny()
    m = _EncDecModel(cfg, "float32", "cpu")
    src = torch.randint(0, cfg.vocab_size, (1, 12))
    enc_out = m.encode(src, attn_kind="flash")
    assert tuple(enc_out.shape) == (1, 12, cfg.d_model)


# --- CPU execution of all phases ----------------------------------------------


def test_all_phase_run_callables_execute_on_cpu():
    for phase in ("encode", "decoder_prefill", "decode"):
        w = wl.build_enc_dec_workload(_tiny(), phase=phase, src_len=8, tgt_len=4,
                                      tgt_ctx=6, decode_tokens=3, precision="fp32",
                                      inner_iters=2)
        w.run()
        w.free()


def test_decode_fixed_step_mode_executes():
    w = wl.build_enc_dec_workload(_tiny(), phase="decode", src_len=8, tgt_ctx=6,
                                  decode_tokens=4, decode_mode="fixed_step",
                                  precision="fp32", inner_iters=2)
    w.run()
    w.free()


def test_gated_enc_dec_executes():
    w = wl.build_enc_dec_workload(_tiny(gated=True, n_heads=4, n_kv=2),
                                  phase="decode", src_len=8, tgt_ctx=6,
                                  decode_tokens=2, precision="fp32", inner_iters=1)
    w.run()
    w.free()


# --- GPU integration ----------------------------------------------------------


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_decoder_prefill_through_runner():
    """A T5-base-shaped decoder_prefill runs through the controlled runner, clears
    the window floor, and the hardware counter reads positive energy. (Encoder
    runs once in setup; the measured phase is the decoder prefill with cross-KV
    projection and caching.)"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tomltransformers.measure import runner as rn

    cfg = _ed_big()

    def builder(inner):
        return wl.build_enc_dec_workload(cfg, phase="decoder_prefill", src_len=512,
                                         tgt_len=128, precision="fp16", inner_iters=inner)

    def measure_fn(work):
        return rn.measure_point(
            work.run, work.spec.label(), repeats=3, warmup_iters=5,
            idle_baseline_s=1.0, thermal_settle=True, thermal_window_s=2.0,
            min_window_s=4.0,
        )

    res, inner = wl.measure_until_floor(builder, measure_fn, target_s=4.0)
    ag = rn.pairwise_agreement(res)
    print(f"\n[ed-decoder_prefill] inner_iters={inner} available={res.instruments_available} "
          f"wall_s={res.summary.get('wall_time_s', {}).get('median'):.2f} "
          f"short_window={res.short_window} cv_exceeded={res.cv_exceeded} "
          f"summary={ {k: round(v['mean'], 2) for k, v in res.summary.items()} } "
          f"agreement={ {k: round(v, 4) for k, v in ag.items()} }")
    assert res.ok
    assert "B" in res.instruments_available and res.summary["B"]["mean"] > 0.0
    assert not res.short_window, "decoder_prefill failed to clear the 4s window floor"


def _ed_big():
    """A T5-base-shaped encoder-decoder: real GPU workload, no download."""
    return TransformerConfig(
        name="probe-ed", arch="encoder_decoder",
        n_encoder_layers=12, n_decoder_layers=12,
        d_model=768, d_ff=3072, n_heads=12, head_dim=64, vocab_size=32128,
        activation="relu", norm_type="rmsnorm", ffn_type="standard",
        tie_embeddings=True, relative_position_bias=True,
    )
