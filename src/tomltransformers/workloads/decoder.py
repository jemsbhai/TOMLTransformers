"""Decoder-only runnable workloads (shape-faithful to architectures/decoder.py).

Builds a real torch module whose per-layer structure matches what the TO-counting
front-end counts, so measured energy and predicted TO counts describe the same
computation:

  pre-norm -> QKV projection -> attention core (causal, SDPA) -> output
  projection -> post-attn norm -> FFN (gated SwiGLU or standard), x n_layers,
  then a final norm and the lm_head on the last token.

Weights are random-init by default (energy depends on op shapes / data movement,
not values). A pretrained path is provided for spot checks.

Attention kind maps to the SDPA backend: 'flash' selects the flash/efficient
kernels (no materialized score matrix); 'eager' forces the math kernel (which
materializes the s x s scores), matching the standard-attention TO accounting.
"""

from __future__ import annotations

from typing import Optional

from .protocol import CallableWorkload, WorkloadSpec
from ..architectures.configs import TransformerConfig, get as get_config


_DTYPE = {"fp16": "float16", "fp32": "float32"}


def _torch():
    import torch
    return torch


class _DecoderModel:
    """Lazily-built torch.nn module mirroring the counted decoder structure.

    Kept as a thin wrapper so we can construct it from our TransformerConfig
    without importing torch at module import time (the package must import on
    machines without torch).
    """

    def __init__(self, cfg: TransformerConfig, dtype_str: str, device: str):
        torch = _torch()
        import torch.nn as nn

        self.cfg = cfg
        # Accept either a precision key ("fp16"/"fp32") or a torch dtype name
        # ("float16"/"float32"); normalize to the torch dtype.
        torch_dtype_name = _DTYPE.get(dtype_str, dtype_str)
        if not hasattr(torch, torch_dtype_name):
            raise ValueError(
                f"dtype '{dtype_str}' is neither a precision key {list(_DTYPE)} "
                f"nor a torch dtype name"
            )
        self.dtype = getattr(torch, torch_dtype_name)
        self.device = device
        d = cfg.d_model
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.kv_heads
        self.head_dim = cfg.d_head
        self.is_gated = cfg.is_gated

        Norm = nn.LayerNorm  # RMSNorm vs LayerNorm differ negligibly in energy;
        # both are counted via op(norm_type). LayerNorm is the available primitive.

        def lin(i, o):
            return nn.Linear(i, o, bias=False)

        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            layer = nn.ModuleDict()
            layer["norm1"] = Norm(d)
            # Fused QKV: q is n_heads*head_dim, k/v are n_kv_heads*head_dim (GQA-aware).
            q_dim = self.n_heads * self.head_dim
            kv_dim = self.n_kv_heads * self.head_dim
            layer["qkv"] = lin(d, q_dim + 2 * kv_dim)
            layer["out"] = lin(q_dim, d)
            layer["norm2"] = Norm(d)
            if cfg.is_gated:
                layer["gate"] = lin(d, cfg.d_ff)
                layer["up"] = lin(d, cfg.d_ff)
                layer["down"] = lin(cfg.d_ff, d)
            else:
                layer["up"] = lin(d, cfg.d_ff)
                layer["down"] = lin(cfg.d_ff, d)
            self.layers.append(layer)
        self.final_norm = Norm(d)
        self.lm_head = lin(d, cfg.vocab_size)

        self.module = nn.Module()
        self.module.embed = self.embed
        self.module.layers = self.layers
        self.module.final_norm = self.final_norm
        self.module.lm_head = self.lm_head
        self.module = self.module.to(device=device, dtype=self.dtype).eval()

        self._act = self._activation_fn(cfg.activation)

    @staticmethod
    def _activation_fn(kind: str):
        torch = _torch()
        import torch.nn.functional as F
        return {
            "gelu": F.gelu,
            "silu": F.silu,
            "relu": F.relu,
        }.get(kind, F.gelu)

    def _attn(self, x, attn_kind: str, last_token_only: bool):
        torch = _torch()
        import torch.nn.functional as F
        B, S, _ = x.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        out_results = []
        for layer in self.layers:
            h = layer["norm1"](x)
            qkv = layer["qkv"](h)
            q_dim = nh * hd
            kv_dim = nkv * hd
            q = qkv[..., :q_dim].view(B, S, nh, hd).transpose(1, 2)
            k = qkv[..., q_dim:q_dim + kv_dim].view(B, S, nkv, hd).transpose(1, 2)
            v = qkv[..., q_dim + kv_dim:].view(B, S, nkv, hd).transpose(1, 2)
            # GQA: expand kv heads to query heads if needed.
            if nkv != nh:
                rep = nh // nkv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            # SDPA backend selection: flash/efficient vs math (materialized scores).
            if attn_kind == "eager":
                backends = [torch.nn.attention.SDPBackend.MATH]
            else:
                backends = [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                ]
            with torch.nn.attention.sdpa_kernel(backends):
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, S, q_dim)
            x = x + layer["out"](a)
            h2 = layer["norm2"](x)
            if self.is_gated:
                ff = layer["down"](self._act(layer["gate"](h2)) * layer["up"](h2))
            else:
                ff = layer["down"](self._act(layer["up"](h2)))
            x = x + ff
        x = self.final_norm(x)
        if last_token_only:
            x = x[:, -1:, :]
        return self.lm_head(x)


def calibrate_inner_iters(
    spec_builder,
    target_s: float = 4.0,
    probe_iters: int = 4,
    warmup: int = 8,
    max_iters: int = 1000000,
) -> int:
    """Initial GUESS for inner_iters so one execution lasts ~target_s.

    This is only a starting estimate. It is deliberately well-warmed (so it does
    not over-warm relative to the real run and underestimate iters), but timing
    extrapolation is fragile near launch-bound regimes, so callers that need a
    guarantee should use `measure_until_floor`, which corrects from the REAL
    measured wall-time. Returns 1 if torch/CUDA is unavailable.
    """
    torch = _torch()
    if not torch.cuda.is_available():
        return 1
    import time

    wl = spec_builder(probe_iters)
    for _ in range(warmup):
        wl.run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wl.run()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    wl.free()
    if dt <= 0:
        return max(1, probe_iters)
    per_iter = dt / probe_iters
    need = int(round(target_s / per_iter))
    return max(1, min(need, max_iters))


def measure_until_floor(
    spec_builder,
    measure_fn,
    *,
    target_s: float = 4.0,
    max_attempts: int = 3,
    safety: float = 1.15,
    max_iters: int = 1000000,
):
    """Build, measure, and if the window is too short, rescale inner_iters from
    the REAL measured wall-time and retry.

    This replaces fragile predict-by-timing: instead of trusting a cold/linear
    extrapolation, it trusts the actual measured duration under the same warmup
    and thermal conditions the run uses.

    Args:
      spec_builder(inner_iters) -> Workload.
      measure_fn(workload) -> PointResult. Typically a closure over
        measure_point(work.run, work.spec.label(), ...). MUST run the real
        controlled path (so wall-time reflects true conditions).
      target_s: desired per-execution window.
      max_attempts: cap on rebuild+remeasure cycles.
      safety: overshoot factor applied when scaling up (aim slightly above floor).

    Returns (PointResult, inner_iters_used). The result carries short_window so
    the caller can record whether the floor was met even if attempts ran out.
    """
    torch = _torch()
    # Initial guess (cheap timing probe), or 1 if no CUDA.
    inner = calibrate_inner_iters(spec_builder, target_s=target_s)

    last_res = None
    for _attempt in range(max_attempts):
        work = spec_builder(inner)
        res = measure_fn(work)
        work.free()
        last_res = res

        # Use measured MEDIAN wall-time as the source of truth (same statistic
        # the short_window flag uses, so they cannot disagree).
        wsum = res.summary.get("wall_time_s", {})
        measured = wsum.get("median")
        if not res.short_window or measured is None or measured <= 0 or measured != measured:
            return res, inner

        # Scale inner_iters from the REAL per-iter cost to clear target_s.
        per_iter = measured / max(inner, 1)
        new_inner = int(round((target_s * safety) / per_iter))
        new_inner = max(inner + 1, min(new_inner, max_iters))
        if new_inner == inner:
            return res, inner   # cannot grow further; return with short_window set
        inner = new_inner

    return last_res, inner


def build_decoder_workload(
    model: str | TransformerConfig,
    *,
    phase: str,                       # "prefill" | "decode"
    seq_len: int,
    precision: str = "fp16",
    weights: str = "random",          # "random" | "pretrained"
    attn_kind: str = "flash",
    inner_iters: int = 1,
    batch_size: int = 1,
    device_index: int = 0,
    decode_tokens: int = 64,          # K, the decode window (tokens per execution)
    pretrained_id: Optional[str] = None,
) -> CallableWorkload:
    """Construct a runnable decoder-only workload.

    prefill: one forward over `seq_len` tokens (no generation), lm_head on the
             last token. Looped `inner_iters` times per execution.
    decode:  prefill a KV cache to `seq_len`, then generate `decode_tokens`
             tokens; the whole generation is one execution, looped `inner_iters`
             times. (Uses a real KV cache via transformers when weights are
             pretrained; for random weights, see note below.)

    NOTE: this first build implements the PREFILL path fully for both random and
    pretrained weights. The decode path is stubbed to raise NotImplementedError
    pending the KV-cache workload design (next step), so we do not silently
    measure a wrong decode.
    """
    if precision not in _DTYPE:
        raise ValueError(f"precision must be one of {list(_DTYPE)}, got {precision}")
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be 'prefill' or 'decode', got {phase}")

    cfg = model if isinstance(model, TransformerConfig) else get_config(model)
    if cfg.arch != "decoder_only":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not decoder_only")

    torch = _torch()
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    spec = WorkloadSpec(
        model_name=cfg.name, arch=cfg.arch, phase=phase, seq_len=seq_len,
        precision=precision, weights=weights, attn_kind=attn_kind,
        inner_iters=inner_iters, batch_size=batch_size,
        extra={"decode_tokens": decode_tokens} if phase == "decode" else {},
    )

    if phase == "decode":
        raise NotImplementedError(
            "decode workload (KV-cache path) is the next build step; "
            "prefill is implemented. This guard prevents measuring a wrong decode."
        )

    # ---- prefill ----
    if weights == "pretrained":
        model_obj, input_ids = _build_pretrained_prefill(
            cfg, seq_len, precision, device, batch_size, pretrained_id)

        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                model_obj(input_ids)

        def free():
            nonlocal model_obj
            del model_obj
            _empty_cache()

        return CallableWorkload(spec=spec, _run=run, _free=free)

    # random-init, shape-faithful
    dm = _DecoderModel(cfg, precision, device)
    ids = torch.randint(0, max(cfg.vocab_size, 1), (batch_size, seq_len), device=device)

    @torch.no_grad()
    def run():
        for _ in range(inner_iters):
            x = dm.embed(ids)
            dm._attn(x, attn_kind=attn_kind, last_token_only=True)

    def free():
        nonlocal dm
        del dm
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _build_pretrained_prefill(cfg, seq_len, precision, device, batch_size, pretrained_id):
    """Load a real pretrained decoder for prefill spot checks (downloads weights)."""
    torch = _torch()
    from transformers import AutoModelForCausalLM
    hf_id = pretrained_id or _default_hf_id(cfg.name)
    dtype = getattr(torch, _DTYPE[precision])
    model_obj = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    model_obj = model_obj.to(device).eval()
    vocab = model_obj.config.vocab_size
    ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    return model_obj, ids


def _default_hf_id(name: str) -> str:
    return {
        "GPT-2": "gpt2",
        "DistilGPT2": "distilgpt2",
        "GPT-2-medium": "gpt2-medium",
        "GPT-2-large": "gpt2-large",
        "GPT-2-XL": "gpt2-xl",
    }.get(name, name)


def _empty_cache() -> None:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
