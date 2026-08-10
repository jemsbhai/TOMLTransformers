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

PARITY FIX 2026-07-24 (pre-representativeness-run; approved; these pretrained
paths were never exercised by the frozen sweep, which is random-init only):
the pretrained builders now hold the COUNTED structure fixed so the
representativeness comparison isolates weight VALUES:
  - prefill runs the bare transformer stack (model.base_model) and applies the
    lm_head (get_output_embeddings) to the LAST token only, matching the
    random-init path and the TO accounting (HF's LMHeadModel forward computes
    all-position logits, ~2e10 extra MACs at s=512 on GPT-2, which would
    confound the weight comparison);
  - the decode cache build likewise runs the bare stack with use_cache=True and
    NO head, matching prefill_into_cache; per-step calls keep the 1-token head,
    matching decode_step;
  - attn_implementation="sdpa" is requested at load (with a graceful fallback
    recorded on the model config) so both arms run the same kernel family.

VARIANT FLAGS 2026-08-10 (Follow-up B; defaults preserve every prior behavior
and key byte-identically): _DecoderModel accepts use_bias and use_wpe so that
pretrained GPT-2 weights can be PORTED into our stack exactly (biases and
learned position embeddings included), enabling the exact logit-equivalence
gate in workloads/port_gpt2.py. Bias adds are O(d) against O(d^2) GEMMs and
the wpe gather is one embedding lookup: negligible energy, and the Follow-up B
random-variant arm measures that delta explicitly rather than assuming it.
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

    def __init__(self, cfg: TransformerConfig, dtype_str: str, device: str,
                 *, use_bias: bool = False, use_wpe: bool = False,
                 max_positions: int = 1024):
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
        self.use_bias = use_bias
        self.use_wpe = use_wpe

        Norm = nn.LayerNorm  # RMSNorm vs LayerNorm differ negligibly in energy;
        # both are counted via op(norm_type). LayerNorm is the available primitive.

        def lin(i, o):
            return nn.Linear(i, o, bias=use_bias)

        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.wpe = nn.Embedding(max_positions, d) if use_wpe else None
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
        self.lm_head = lin(d, cfg.vocab_size) if not use_bias else \
            _torch().nn.Linear(d, cfg.vocab_size, bias=False)
        # (GPT-2's lm_head has no bias even though other linears do; keep it
        # bias-free in every variant so porting maps 1:1.)

        self.module = nn.Module()
        self.module.embed = self.embed
        if self.wpe is not None:
            self.module.wpe = self.wpe
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

    def _embed(self, ids, pos_offset: int = 0):
        """Token embedding, plus learned position embedding when use_wpe."""
        x = self.embed(ids)
        if self.wpe is not None:
            torch = _torch()
            S = ids.shape[1]
            pos = torch.arange(pos_offset, pos_offset + S, device=ids.device)
            x = x + self.wpe(pos)
        return x

    def _layer_attn_block(self, layer, x, *, attn_kind, kv_cache=None, layer_idx=None):
        """One transformer layer: pre-norm, attention, post-norm, FFN, residuals.

        If kv_cache is None -> full self-attention over all positions in x
        (prefill: causal). If kv_cache is provided -> incremental decode: x is a
        single token; its K/V are appended to kv_cache[layer_idx] and attention
        runs over the whole cached context (the cache READ grows with context,
        which is the memory-bound behavior decode_step counts).
        """
        torch = _torch()
        import torch.nn.functional as F
        B, S, _ = x.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q_dim = nh * hd
        kv_dim = nkv * hd

        h = layer["norm1"](x)
        qkv = layer["qkv"](h)
        q = qkv[..., :q_dim].view(B, S, nh, hd).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].view(B, S, nkv, hd).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].view(B, S, nkv, hd).transpose(1, 2)

        is_causal = True
        if kv_cache is not None:
            # Append this step's K/V to the persistent cache, then attend over all.
            pk, pv = kv_cache[layer_idx]
            if pk is not None:
                k = torch.cat([pk, k], dim=2)   # grow along the sequence (time) axis
                v = torch.cat([pv, v], dim=2)
            kv_cache[layer_idx] = (k, v)
            # Single query attends to the entire cached context: no causal masking
            # needed (everything in the cache is a past/current position).
            is_causal = False

        if nkv != nh:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if attn_kind == "eager":
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        with torch.nn.attention.sdpa_kernel(backends):
            a = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        a = a.transpose(1, 2).reshape(B, S, q_dim)
        x = x + layer["out"](a)
        h2 = layer["norm2"](x)
        if self.is_gated:
            ff = layer["down"](self._act(layer["gate"](h2)) * layer["up"](h2))
        else:
            ff = layer["down"](self._act(layer["up"](h2)))
        x = x + ff
        return x

    def _attn(self, x, attn_kind: str, last_token_only: bool):
        """Prefill forward: full causal self-attention over all positions."""
        for layer in self.layers:
            x = self._layer_attn_block(layer, x, attn_kind=attn_kind)
        x = self.final_norm(x)
        if last_token_only:
            x = x[:, -1:, :]
        return self.lm_head(x)

    def new_kv_cache(self):
        """Fresh per-layer KV cache: list of (K, V) slots, initially empty."""
        return [(None, None) for _ in range(self.n_layers)]

    def prefill_into_cache(self, ids, *, attn_kind: str):
        """Run a prefill over `ids`, populating and returning a KV cache.

        This is the realistic decode setup: the context is established by a real
        prefill pass (so the cache holds true K/V for `ids.shape[1]` positions),
        after which decode steps append one token at a time.
        """
        x = self._embed(ids)
        cache = self.new_kv_cache()
        for li, layer in enumerate(self.layers):
            x = self._layer_attn_block(layer, x, attn_kind=attn_kind,
                                       kv_cache=cache, layer_idx=li)
        return cache

    def decode_step(self, token_ids, kv_cache, *, attn_kind: str,
                    pos_offset: int = 0):
        """One decode step: a single token, attending over the full cached context.

        token_ids: (B, 1) int. Computes 1-token QKV, appends K/V to kv_cache,
        attends over the grown cache, and returns the lm_head logits. Mirrors
        architectures/decoder.py decode_step: 1-token compute, cache read scales
        with context. pos_offset is the token's absolute position (used only
        when use_wpe).
        """
        x = self._embed(token_ids, pos_offset=pos_offset)
        for li, layer in enumerate(self.layers):
            x = self._layer_attn_block(layer, x, attn_kind=attn_kind,
                                       kv_cache=kv_cache, layer_idx=li)
        x = self.final_norm(x)
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


def _load_pretrained_causal(hf_id: str, dtype, device: str):
    """Load a pretrained causal LM with the SDPA attention implementation
    requested (parity with the random-init SDPA path); falls back gracefully
    on transformers versions that reject the kwarg. The implementation that
    actually loaded is readable at model.config._attn_implementation."""
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=dtype, attn_implementation="sdpa")
    except (TypeError, ValueError):
        m = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    return m.to(device).eval()


def build_decoder_workload(
    model: str | TransformerConfig,
    *,
    phase: str,                       # "prefill" | "decode"
    seq_len: int,
    precision: str = "fp16",
    weights: str = "random",          # "random" | "pretrained" | "ported" | "random_v"
    attn_kind: str = "flash",
    inner_iters: int = 1,
    batch_size: int = 1,
    device_index: int = 0,
    decode_tokens: int = 64,          # K: tokens generated per execution (growing mode)
    decode_mode: str = "growing",     # "growing" | "fixed_step"
    pretrained_id: Optional[str] = None,
) -> CallableWorkload:
    """Construct a runnable decoder-only workload.

    prefill: one forward over `seq_len` tokens (no generation), lm_head on the
             last token. Looped `inner_iters` times per execution.

    decode:  establish a KV cache at context length `seq_len` (via a real prefill),
             then generate tokens. Two selectable modes:
               - "growing" (default, TRUE decode): generate `decode_tokens` (K)
                 tokens; context grows seq_len -> seq_len+K and the cache read
                 grows each step. One execution = the K-token generation;
                 per-token energy = window_energy / K. Matches decoder.py
                 decode_total.
               - "fixed_step": hold context at exactly `seq_len` and repeatedly
                 run a single one-token decode step at that fixed context (the
                 cache is rebuilt to seq_len before each step so context does not
                 grow). Isolates per-step cost at a known context length. Matches
                 decoder.py decode_step.
             Both realize a real incremental KV cache: random-init uses our own
             module's cache; pretrained uses transformers' past_key_values.

    Weights arms (Follow-up B, 2026-08-10):
      - "random": our stack, random init, no bias/wpe (the sweep's arm).
      - "pretrained": the HF implementation with the 2026-07-24 parity fix.
      - "ported": pretrained GPT-2 values PORTED into our stack (bias+wpe
        variant), logit-verified against HF before any measurement.
      - "random_v": our stack, random init, bias+wpe variant (structure
        control for the ported arm).

    The window-length floor is met by looping: `inner_iters` repeats of the whole
    decode execution (use measure_until_floor to size it).
    """
    if precision not in _DTYPE:
        raise ValueError(f"precision must be one of {list(_DTYPE)}, got {precision}")
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be 'prefill' or 'decode', got {phase}")
    if phase == "decode" and decode_mode not in ("growing", "fixed_step"):
        raise ValueError(f"decode_mode must be 'growing' or 'fixed_step', got {decode_mode}")
    if weights not in ("random", "pretrained", "ported", "random_v"):
        raise ValueError(f"unknown weights arm '{weights}'")

    cfg = model if isinstance(model, TransformerConfig) else get_config(model)
    if cfg.arch != "decoder_only":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not decoder_only")

    torch = _torch()
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    extra = {}
    if phase == "decode":
        extra = {"decode_tokens": decode_tokens, "decode_mode": decode_mode}
    spec = WorkloadSpec(
        model_name=cfg.name, arch=cfg.arch, phase=phase, seq_len=seq_len,
        precision=precision, weights=weights, attn_kind=attn_kind,
        inner_iters=inner_iters, batch_size=batch_size,
        extra=extra,
    )

    if phase == "decode":
        if weights == "pretrained":
            return _build_pretrained_decode(
                cfg, spec, seq_len, precision, device, batch_size,
                decode_tokens, decode_mode, inner_iters, pretrained_id)
        return _build_our_stack_decode(
            cfg, spec, seq_len, precision, device, batch_size,
            decode_tokens, decode_mode, attn_kind, inner_iters,
            weights, pretrained_id)

    # ---- prefill ----
    if weights == "pretrained":
        model_obj, stack, head, input_ids = _build_pretrained_prefill(
            cfg, seq_len, precision, device, batch_size, pretrained_id)

        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                # PARITY (2026-07-24): bare stack + last-token head, matching the
                # random-init path and the TO accounting (see module docstring).
                hs = stack(input_ids).last_hidden_state
                head(hs[:, -1:, :])

        def free():
            nonlocal model_obj, stack, head
            del model_obj, stack, head
            _empty_cache()

        return CallableWorkload(spec=spec, _run=run, _free=free)

    # our stack: random | random_v | ported
    dm = _make_our_stack(cfg, precision, device, weights, pretrained_id)
    ids = torch.randint(0, max(cfg.vocab_size, 1), (batch_size, seq_len), device=device)

    @torch.no_grad()
    def run():
        for _ in range(inner_iters):
            x = dm._embed(ids)
            dm._attn(x, attn_kind=attn_kind, last_token_only=True)

    def free():
        nonlocal dm
        del dm
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _make_our_stack(cfg, precision, device, weights, pretrained_id):
    """Instantiate OUR _DecoderModel for the three our-stack arms."""
    if weights == "random":
        return _DecoderModel(cfg, precision, device)
    if weights == "random_v":
        return _DecoderModel(cfg, precision, device, use_bias=True, use_wpe=True)
    if weights == "ported":
        from .port_gpt2 import load_and_port_gpt2
        hf_id = pretrained_id or _default_hf_id(cfg.name)
        dm, diff = load_and_port_gpt2(hf_id, precision=precision, device=device)
        print(f"[ported] {hf_id}: logit-verified vs HF, max|diff|={diff:.2e} (fp32)")
        return dm
    raise ValueError(weights)


def _build_our_stack_decode(cfg, spec, seq_len, precision, device, batch_size,
                            decode_tokens, decode_mode, attn_kind, inner_iters,
                            weights, pretrained_id):
    """Decode via our own incremental KV cache (random / random_v / ported)."""
    torch = _torch()
    dm = _make_our_stack(cfg, precision, device, weights, pretrained_id)
    vocab = max(cfg.vocab_size, 1)
    ctx_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    next_tok = torch.randint(0, vocab, (batch_size, 1), device=device)

    if decode_mode == "growing":
        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                # Establish the cache with a real prefill to seq_len, then
                # generate K tokens; context (and cache read) grows each step.
                cache = dm.prefill_into_cache(ctx_ids, attn_kind=attn_kind)
                for _t in range(decode_tokens):
                    dm.decode_step(next_tok, cache, attn_kind=attn_kind,
                                   pos_offset=seq_len + _t)
    else:  # fixed_step
        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                # Rebuild the cache to exactly seq_len, then run ONE step at that
                # fixed context. Repeated inner_iters times (context never grows).
                cache = dm.prefill_into_cache(ctx_ids, attn_kind=attn_kind)
                dm.decode_step(next_tok, cache, attn_kind=attn_kind,
                               pos_offset=seq_len)

    def free():
        nonlocal dm
        del dm
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _build_pretrained_decode(cfg, spec, seq_len, precision, device, batch_size,
                             decode_tokens, decode_mode, inner_iters, pretrained_id):
    """Pretrained decode via transformers past_key_values (downloads weights).

    PARITY (2026-07-24): the cache is established through the BARE stack
    (model.base_model, no head), matching prefill_into_cache on the random-init
    side; per-step calls keep the full model (1-token head), matching
    decode_step.
    """
    torch = _torch()
    hf_id = pretrained_id or _default_hf_id(cfg.name)
    dtype = getattr(torch, _DTYPE[precision])
    model_obj = _load_pretrained_causal(hf_id, dtype, device)
    stack = model_obj.base_model
    vocab = model_obj.config.vocab_size
    ctx_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    next_tok = torch.randint(0, vocab, (batch_size, 1), device=device)

    def _prefill_cache():
        out = stack(ctx_ids, use_cache=True)   # no head: parity with random path
        return out.past_key_values

    if decode_mode == "growing":
        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                pkv = _prefill_cache()
                tok = next_tok
                for _t in range(decode_tokens):
                    out = model_obj(tok, past_key_values=pkv, use_cache=True)
                    pkv = out.past_key_values   # cache grows each step
    else:  # fixed_step
        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                pkv = _prefill_cache()
                model_obj(next_tok, past_key_values=pkv, use_cache=True)

    def free():
        nonlocal model_obj, stack
        del model_obj, stack
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _build_pretrained_prefill(cfg, seq_len, precision, device, batch_size, pretrained_id):
    """Load a real pretrained decoder for prefill spot checks (downloads weights).

    Returns (model_obj, stack, head, ids): the bare stack and the lm_head are
    applied separately by the caller for structural parity (see module docstring).
    """
    torch = _torch()
    hf_id = pretrained_id or _default_hf_id(cfg.name)
    dtype = getattr(torch, _DTYPE[precision])
    model_obj = _load_pretrained_causal(hf_id, dtype, device)
    stack = model_obj.base_model
    head = model_obj.get_output_embeddings()
    vocab = model_obj.config.vocab_size
    ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    return model_obj, stack, head, ids


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
