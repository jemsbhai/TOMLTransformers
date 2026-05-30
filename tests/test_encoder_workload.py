"""Tests for the encoder-only runnable workloads (tomltransformers.workloads).

Structural tests run on CPU with tiny configs (no GPU, no downloads): they check
the runnable model mirrors the counted encoder structure (layer count, the right
projections per layer, GQA-aware QKV width), the modality-dependent embedding
(token gather for text vs patch projection for vision), the ViT fixed sequence
length (num_patches + 1), and output shapes with/without the classifier head. One
GPU integration test runs a real BERT-shaped encode through the controlled runner
and checks the window floor and B energy at the 100 Hz default.
"""

import pytest

from tomltransformers.architectures.configs import TransformerConfig
from tomltransformers.measure import instruments as ins
from tomltransformers import workloads as wl


def _tiny_text(gated=False, n_kv=None, n_heads=4):
    """A tiny but shape-valid text (BERT-like) encoder for fast CPU tests."""
    return TransformerConfig(
        name="tiny-enc", arch="encoder_only", n_layers=2, d_model=32, d_ff=64,
        n_heads=n_heads, n_kv_heads=(n_kv if n_kv is not None else n_heads),
        head_dim=8, vocab_size=128,
        activation=("silu" if gated else "gelu"),
        norm_type=("rmsnorm" if gated else "layernorm"),
        ffn_type=("gated" if gated else "standard"),
        max_position=64,
    )


def _tiny_vision(num_patches=9, num_classes=10):
    """A tiny but shape-valid vision (ViT-like) encoder."""
    return TransformerConfig(
        name="tiny-vit", arch="encoder_only", n_layers=2, d_model=32, d_ff=64,
        n_heads=4, head_dim=8, vocab_size=0,
        activation="gelu", norm_type="layernorm", ffn_type="standard",
        is_vision=True, num_patches=num_patches, num_classes=num_classes,
    )


# --- spec / interface ---------------------------------------------------------


def test_spec_and_label_text():
    w = wl.build_encoder_workload(_tiny_text(), seq_len=16, precision="fp32")
    assert w.spec.model_name == "tiny-enc"
    assert w.spec.phase == "encode"
    assert w.spec.seq_len == 16
    lbl = w.spec.label()
    assert "tiny-enc" in lbl and "encode" in lbl and "s16" in lbl
    w.free()


def test_workload_satisfies_protocol():
    w = wl.build_encoder_workload(_tiny_text(), seq_len=8, precision="fp32")
    assert isinstance(w, wl.Workload)
    w.free()


def test_rejects_non_encoder_config():
    dec = TransformerConfig(
        name="d", arch="decoder_only", n_layers=2, d_model=32, d_ff=64,
        n_heads=4, head_dim=8, vocab_size=64,
    )
    with pytest.raises(ValueError, match="not encoder_only"):
        wl.build_encoder_workload(dec, seq_len=8)


def test_text_encoder_requires_seq_len():
    with pytest.raises(ValueError, match="seq_len is required"):
        wl.build_encoder_workload(_tiny_text(), seq_len=None)


def test_rejects_bad_precision():
    with pytest.raises(ValueError, match="precision"):
        wl.build_encoder_workload(_tiny_text(), seq_len=8, precision="fp8")


# --- vision sequence-length handling ------------------------------------------


def test_vision_seq_len_is_fixed_num_patches_plus_one():
    # seq_len is IGNORED for vision; s = num_patches + 1 (class token).
    w = wl.build_encoder_workload(_tiny_vision(num_patches=9), seq_len=99999,
                                  precision="fp32")
    assert w.spec.seq_len == 10   # 9 patches + 1 class token
    assert w.spec.extra["is_vision"] is True
    w.free()


def test_vision_does_not_require_seq_len():
    # vision ignores seq_len, so None is fine.
    w = wl.build_encoder_workload(_tiny_vision(), seq_len=None, precision="fp32")
    assert w.spec.seq_len == _tiny_vision().num_patches + 1
    w.free()


# --- structural fidelity to architectures/encoder.py (CPU) --------------------


def test_text_model_uses_token_embedding_no_patch_proj():
    from tomltransformers.workloads.encoder import _EncoderModel
    em = _EncoderModel(_tiny_text(), "float32", "cpu")
    assert em.embed is not None        # token-embedding gather
    assert em.patch_proj is None       # no patch projection for text
    assert len(em.layers) == 2


def test_vision_model_uses_patch_projection_no_token_embedding():
    from tomltransformers.workloads.encoder import _EncoderModel, PATCH_DIM
    em = _EncoderModel(_tiny_vision(), "float32", "cpu")
    assert em.patch_proj is not None   # patch projection (a GEMM)
    assert em.embed is None            # no token embedding for vision
    # patch projection maps PATCH_DIM -> d_model.
    assert em.patch_proj.in_features == PATCH_DIM
    assert em.patch_proj.out_features == _tiny_vision().d_model


def test_layer_structure_standard_and_gated():
    from tomltransformers.workloads.encoder import _EncoderModel
    std = _EncoderModel(_tiny_text(gated=False), "float32", "cpu").layers[0]
    assert "gate" not in std and "up" in std and "down" in std
    assert set(["norm1", "norm2", "qkv", "out", "up", "down"]).issubset(set(std.keys()))
    gat = _EncoderModel(_tiny_text(gated=True), "float32", "cpu").layers[0]
    assert "gate" in gat and "up" in gat and "down" in gat


def test_qkv_width_is_gqa_aware():
    from tomltransformers.workloads.encoder import _EncoderModel
    # 4 q heads, 2 kv heads, head_dim 8 -> qkv out = 4*8 + 2*(2*8) = 64.
    em = _EncoderModel(_tiny_text(n_heads=4, n_kv=2), "float32", "cpu")
    assert em.layers[0]["qkv"].out_features == 4 * 8 + 2 * (2 * 8)


def test_head_out_features_uses_num_classes_for_vision():
    from tomltransformers.workloads.encoder import _EncoderModel
    em = _EncoderModel(_tiny_vision(num_classes=10), "float32", "cpu")
    assert em.head.out_features == 10
    # text encoder with num_classes=0 -> head maps to d_model.
    emt = _EncoderModel(_tiny_text(), "float32", "cpu")
    assert emt.head.out_features == _tiny_text().d_model


# --- forward shapes (CPU) -----------------------------------------------------


def test_text_forward_head_shape():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder import _EncoderModel
    cfg = _tiny_text()
    em = _EncoderModel(cfg, "float32", "cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = em.forward(ids, attn_kind="eager", include_head=True)
    # head on the single pooled token -> (B, 1, head_out); head_out == d_model here.
    assert tuple(out.shape) == (1, 1, cfg.d_model)


def test_text_forward_no_head_returns_hidden_states():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder import _EncoderModel
    cfg = _tiny_text()
    em = _EncoderModel(cfg, "float32", "cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    out = em.forward(ids, attn_kind="flash", include_head=False)
    assert tuple(out.shape) == (1, 12, cfg.d_model)


def test_vision_forward_head_shape():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder import _EncoderModel, PATCH_DIM
    cfg = _tiny_vision(num_patches=9, num_classes=10)
    em = _EncoderModel(cfg, "float32", "cpu")
    s = cfg.num_patches + 1
    patches = torch.randn(1, s, PATCH_DIM)
    out = em.forward(patches, attn_kind="eager", include_head=True)
    assert tuple(out.shape) == (1, 1, 10)   # num_classes logits on pooled token


def test_forward_runs_gated_and_gqa():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.encoder import _EncoderModel
    cfg = _tiny_text(gated=True, n_heads=4, n_kv=2)
    em = _EncoderModel(cfg, "float32", "cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    out = em.forward(ids, attn_kind="eager", include_head=False)
    assert tuple(out.shape) == (1, 10, cfg.d_model)


def test_run_callable_executes_text_and_vision_on_cpu():
    wt = wl.build_encoder_workload(_tiny_text(), seq_len=8, precision="fp32",
                                   inner_iters=2)
    wt.run()
    wt.free()
    wv = wl.build_encoder_workload(_tiny_vision(), precision="fp32", inner_iters=2)
    wv.run()
    wv.free()


# --- GPU integration: real encode through the controlled runner ---------------


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_encode_through_runner():
    """A BERT-shaped bidirectional encode runs through the controlled runner,
    clears the window floor, and the hardware counter reads positive energy."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tomltransformers.measure import runner as rn

    cfg = _enc_big()

    def builder(inner):
        return wl.build_encoder_workload(cfg, seq_len=512, precision="fp16",
                                         inner_iters=inner)

    def measure_fn(work):
        return rn.measure_point(
            work.run, work.spec.label(), repeats=3, warmup_iters=5,
            idle_baseline_s=1.0, thermal_settle=True, thermal_window_s=2.0,
            min_window_s=4.0,
        )

    res, inner = wl.measure_until_floor(builder, measure_fn, target_s=4.0)
    ag = rn.pairwise_agreement(res)
    print(f"\n[encode] inner_iters={inner} available={res.instruments_available} "
          f"wall_s={res.summary.get('wall_time_s', {}).get('median'):.2f} "
          f"short_window={res.short_window} cv_exceeded={res.cv_exceeded} "
          f"summary={ {k: round(v['mean'], 2) for k, v in res.summary.items()} } "
          f"agreement={ {k: round(v, 4) for k, v in ag.items()} }")
    assert res.ok
    assert "B" in res.instruments_available and res.summary["B"]["mean"] > 0.0
    assert not res.short_window, "encode failed to clear the 4s window floor"


def _enc_big():
    """A BERT-large-shaped encoder: real GPU workload, no download."""
    return TransformerConfig(
        name="probe-enc", arch="encoder_only", n_layers=24, d_model=1024, d_ff=4096,
        n_heads=16, head_dim=64, vocab_size=30522,
        activation="gelu", norm_type="layernorm", ffn_type="standard",
        max_position=512,
    )
