"""Tests for the decoder-only runnable workloads (tomltransformers.workloads).

Structural tests run on CPU with a tiny config (no GPU, no downloads): they check
the runnable model mirrors the counted decoder structure (layer count, the right
projections per layer, GQA-aware QKV width, output shape), the decode KV-cache
grows one position per step (the memory-bound signature), and that the spec/label
are correct. GPU integration tests run a real prefill and a real (growing-cache)
decode through the controlled runner and check the window floor and A-vs-B
agreement at the 100 Hz default.
"""

import pytest

from tomltransformers.architectures.configs import TransformerConfig
from tomltransformers.measure import instruments as ins
from tomltransformers import workloads as wl


def _tiny(arch="decoder_only", gated=False, n_kv=None, n_heads=4):
    """A tiny but shape-valid decoder config for fast CPU tests."""
    return TransformerConfig(
        name="tiny-dec", arch=arch, n_layers=2, d_model=32, d_ff=64,
        n_heads=n_heads, n_kv_heads=(n_kv if n_kv is not None else n_heads),
        head_dim=8, vocab_size=128,
        activation=("silu" if gated else "gelu"),
        norm_type=("rmsnorm" if gated else "layernorm"),
        ffn_type=("gated" if gated else "standard"),
        max_position=64,
    )


# --- spec / interface ---------------------------------------------------------


def test_spec_and_label():
    w = wl.build_decoder_workload(_tiny(), phase="prefill", seq_len=16,
                                  precision="fp32", weights="random")
    assert w.spec.model_name == "tiny-dec"
    assert w.spec.phase == "prefill"
    assert w.spec.precision == "fp32"
    lbl = w.spec.label()
    assert "tiny-dec" in lbl and "prefill" in lbl and "s16" in lbl and "rand" in lbl
    w.free()


def test_workload_satisfies_protocol():
    w = wl.build_decoder_workload(_tiny(), phase="prefill", seq_len=8, precision="fp32")
    assert isinstance(w, wl.Workload)     # runtime_checkable Protocol
    w.free()


def test_rejects_non_decoder_config():
    enc = _tiny(arch="encoder_only")
    with pytest.raises(ValueError, match="not decoder_only"):
        wl.build_decoder_workload(enc, phase="prefill", seq_len=8)


def test_rejects_bad_precision_and_phase():
    with pytest.raises(ValueError, match="precision"):
        wl.build_decoder_workload(_tiny(), phase="prefill", seq_len=8, precision="fp8")
    with pytest.raises(ValueError, match="phase"):
        wl.build_decoder_workload(_tiny(), phase="generate", seq_len=8)


def test_decode_builds_for_both_modes():
    # decode is implemented now; both modes build and carry mode in the spec.
    for mode in ("growing", "fixed_step"):
        w = wl.build_decoder_workload(_tiny(), phase="decode", seq_len=8,
                                      precision="fp32", decode_tokens=4,
                                      decode_mode=mode)
        assert w.spec.phase == "decode"
        assert w.spec.extra["decode_mode"] == mode
        assert w.spec.extra["decode_tokens"] == 4
        w.free()


def test_decode_rejects_bad_mode():
    with pytest.raises(ValueError, match="decode_mode"):
        wl.build_decoder_workload(_tiny(), phase="decode", seq_len=8,
                                  decode_mode="sampling")


# --- structural fidelity to architectures/decoder.py (CPU) --------------------


def test_model_structure_standard_ffn():
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(gated=False)
    dm = _DecoderModel(cfg, "float32", "cpu")
    assert len(dm.layers) == cfg.n_layers
    layer = dm.layers[0]
    # standard FFN: up + down, NO gate.
    assert "gate" not in layer
    assert "up" in layer and "down" in layer
    # two norms, qkv, out.
    assert set(["norm1", "norm2", "qkv", "out", "up", "down"]).issubset(set(layer.keys()))


def test_model_structure_gated_ffn():
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(gated=True)
    dm = _DecoderModel(cfg, "float32", "cpu")
    layer = dm.layers[0]
    # gated FFN: gate + up + down.
    assert "gate" in layer and "up" in layer and "down" in layer


def test_qkv_width_is_gqa_aware():
    from tomltransformers.workloads.decoder import _DecoderModel
    # 4 query heads, 2 kv heads, head_dim 8 -> qkv out = 4*8 + 2*(2*8) = 32+32 = 64.
    cfg = _tiny(n_heads=4, n_kv=2)
    dm = _DecoderModel(cfg, "float32", "cpu")
    qkv = dm.layers[0]["qkv"]
    assert qkv.out_features == 4 * 8 + 2 * (2 * 8)


def test_forward_output_shape_last_token():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(gated=False)
    dm = _DecoderModel(cfg, "float32", "cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    x = dm.embed(ids)
    out = dm._attn(x, attn_kind="eager", last_token_only=True)
    # lm_head on the last token only: (B, 1, vocab).
    assert tuple(out.shape) == (1, 1, cfg.vocab_size)


def test_forward_runs_with_gated_and_gqa():
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(gated=True, n_heads=4, n_kv=2)
    dm = _DecoderModel(cfg, "float32", "cpu")
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    x = dm.embed(ids)
    out = dm._attn(x, attn_kind="eager", last_token_only=False)
    assert tuple(out.shape) == (1, 12, cfg.vocab_size)


def test_run_callable_executes_on_cpu():
    # inner_iters loop should call the forward without error on CPU.
    w = wl.build_decoder_workload(_tiny(), phase="prefill", seq_len=8,
                                  precision="fp32", inner_iters=3)
    w.run()
    w.free()


# --- decode KV-cache mechanics (CPU) ------------------------------------------


def test_kv_cache_grows_one_position_per_step():
    """The defining decode property: each step appends exactly one (K,V) position,
    so the cached context length grows by 1 per step. This is the memory-bound
    signature the thesis rests on (cache READ scales with context)."""
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(gated=False)
    dm = _DecoderModel(cfg, "float32", "cpu")
    ctx = torch.randint(0, cfg.vocab_size, (1, 10))
    cache = dm.prefill_into_cache(ctx, attn_kind="eager")
    # after prefill, each layer's cache holds the 10 context positions.
    k0, v0 = cache[0]
    assert k0.shape[2] == 10 and v0.shape[2] == 10
    # one decode step -> context length 11 in every layer.
    tok = torch.randint(0, cfg.vocab_size, (1, 1))
    dm.decode_step(tok, cache, attn_kind="eager")
    for li in range(cfg.n_layers):
        k, v = cache[li]
        assert k.shape[2] == 11 and v.shape[2] == 11, f"layer {li} cache did not grow"
    # a second step -> 12.
    dm.decode_step(tok, cache, attn_kind="eager")
    k, v = cache[0]
    assert k.shape[2] == 12 and v.shape[2] == 12


def test_decode_step_processes_single_token():
    """decode_step computes 1-token QKV and returns logits for exactly one token."""
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(gated=False)
    dm = _DecoderModel(cfg, "float32", "cpu")
    ctx = torch.randint(0, cfg.vocab_size, (1, 6))
    cache = dm.prefill_into_cache(ctx, attn_kind="eager")
    tok = torch.randint(0, cfg.vocab_size, (1, 1))
    logits = dm.decode_step(tok, cache, attn_kind="eager")
    assert tuple(logits.shape) == (1, 1, cfg.vocab_size)


def test_kv_cache_gqa_shapes():
    """Cached K/V have n_kv_heads (not n_heads) under GQA."""
    torch = pytest.importorskip("torch")
    from tomltransformers.workloads.decoder import _DecoderModel
    cfg = _tiny(n_heads=4, n_kv=2)
    dm = _DecoderModel(cfg, "float32", "cpu")
    ctx = torch.randint(0, cfg.vocab_size, (1, 5))
    cache = dm.prefill_into_cache(ctx, attn_kind="eager")
    k, v = cache[0]
    # (B, n_kv_heads, seq, head_dim)
    assert k.shape[1] == 2 and v.shape[1] == 2
    assert k.shape[3] == cfg.d_head


def test_decode_run_callables_execute_on_cpu():
    """Both decode modes' run() execute end-to-end on CPU (random-init path)."""
    for mode in ("growing", "fixed_step"):
        w = wl.build_decoder_workload(_tiny(), phase="decode", seq_len=6,
                                      precision="fp32", decode_tokens=3,
                                      decode_mode=mode, inner_iters=2)
        w.run()
        w.free()


# --- GPU integration: real prefill through the controlled runner --------------


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_prefill_through_runner_agreement():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tomltransformers.measure import runner as rn

    cfg = _tiny_big()  # large enough that a looped prefill is a real GPU load

    def builder(inner):
        return wl.build_decoder_workload(cfg, phase="prefill", seq_len=512,
                                         precision="fp16", inner_iters=inner)

    def measure_fn(work):
        return rn.measure_point(
            work.run, work.spec.label(), repeats=3, warmup_iters=5,
            idle_baseline_s=1.0, thermal_settle=True, thermal_window_s=2.0,
            min_window_s=4.0,
        )

    # measure-until-floor: rescale inner_iters from REAL measured wall-time and
    # retry until the 4 s window is cleared (replaces fragile predict-by-timing).
    res, inner = wl.measure_until_floor(builder, measure_fn, target_s=4.0)
    ag = rn.pairwise_agreement(res)
    print(f"\n[workload] inner_iters={inner} available={res.instruments_available} "
          f"wall_s={res.summary.get('wall_time_s', {}).get('median'):.2f} "
          f"short_window={res.short_window} cv_exceeded={res.cv_exceeded} "
          f"summary={ {k: round(v['mean'], 2) for k, v in res.summary.items()} } "
          f"agreement={ {k: round(v, 4) for k, v in ag.items()} }")
    assert res.ok
    assert "B" in res.instruments_available and res.summary["B"]["mean"] > 0.0
    assert not res.short_window, "measure_until_floor failed to clear the 4s window floor"
    # The ~5% confirmation: A vs B on a real transformer prefill at 100 Hz,
    # controlled path. Loose 12% gate to avoid laptop-GPU flakiness; the actual
    # number is in the printout.
    if "A" in res.summary:
        assert ag["A-B"] < 0.12, f"A-vs-B {ag['A-B']:.1%}; notes={res.notes}"


@pytest.mark.skipif(not ins.nvml_available(), reason="no NVML / GPU")
def test_gpu_decode_growing_through_runner():
    """Real decode (growing KV cache) runs through the controlled runner, clears
    the window floor, and the hardware counter reads positive energy. This is the
    memory-bound regime; we are validating the workload executes and measures, not
    yet calibrating its energy."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tomltransformers.measure import runner as rn

    cfg = _tiny_big()

    def builder(inner):
        return wl.build_decoder_workload(cfg, phase="decode", seq_len=256,
                                         precision="fp16", decode_tokens=32,
                                         decode_mode="growing", inner_iters=inner)

    def measure_fn(work):
        return rn.measure_point(
            work.run, work.spec.label(), repeats=3, warmup_iters=3,
            idle_baseline_s=1.0, thermal_settle=True, thermal_window_s=2.0,
            min_window_s=4.0,
        )

    res, inner = wl.measure_until_floor(builder, measure_fn, target_s=4.0)
    ag = rn.pairwise_agreement(res)
    print(f"\n[decode-growing] inner_iters={inner} available={res.instruments_available} "
          f"wall_s={res.summary.get('wall_time_s', {}).get('median'):.2f} "
          f"short_window={res.short_window} cv_exceeded={res.cv_exceeded} "
          f"summary={ {k: round(v['mean'], 2) for k, v in res.summary.items()} } "
          f"agreement={ {k: round(v, 4) for k, v in ag.items()} }")
    assert res.ok
    assert "B" in res.instruments_available and res.summary["B"]["mean"] > 0.0
    assert not res.short_window, "decode failed to clear the 4s window floor"


def _tiny_big():
    """A mid-size decoder: heavy enough to be a real GPU workload, no download."""
    return TransformerConfig(
        name="probe-dec", arch="decoder_only", n_layers=12, d_model=1024, d_ff=4096,
        n_heads=16, n_kv_heads=16, head_dim=64, vocab_size=50257,
        activation="gelu", norm_type="layernorm", ffn_type="standard",
        max_position=1024,
    )
