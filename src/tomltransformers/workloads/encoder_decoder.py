"""Encoder-decoder runnable workloads (shape-faithful to architectures/encoder_decoder.py).

The most involved class: it combines all three attention patterns, matching the
front-end's phase split (encode / decoder_prefill / decode_step).

  - Encoder: bidirectional self-attention over the source, run ONCE, no cache.
  - Decoder self-attention: causal, over the target, KV cache that GROWS with the
    target length as generation proceeds.
  - Cross-attention: queries from the decoder; keys/values projected ONCE from the
    final encoder output and cached for the whole generation. During decode,
    cross-attention does Q-projection only (no K/V projection, no cache write),
    just a constant-size cached read whose size scales with the SOURCE length.

Decoder layers therefore have THREE norms (self-attn, cross-attn, FFN). T5
(RMSNorm, relative-position bias, ReLU) vs BART (LayerNorm, learned positions,
GELU) fall out of the config; we use nn.LayerNorm for both norm types (the
energy difference is negligible and is accounted in the TO op cost).

Phases are exposed as SEPARATE workloads so cost attribution across the three
attention types is explicit. Decode and decoder_prefill physically require an
encoder output to cross-attend to, so each phase workload establishes its
prerequisite state in (un-measured) setup and measures only the phase of
interest:
  - encode: measures the encoder stack over src (standalone).
  - decoder_prefill: setup runs the encoder once (enc_out); measured run() does
    the decoder prefill, projecting+caching the cross-attention K/V from enc_out
    and self-attending causally over the target prompt. Each looped iteration
    rebuilds caches (so we measure prefill repeatedly, not accumulate).
  - decode: setup runs encoder + decoder_prefill to establish self- and cross-
    caches at a target context; measured run() does decode steps (growing self-
    cache, static cross-cache), matching decode_step / decode_total.

Weights are random-init by default. A pretrained path (T5/BART via HF) is provided
for spot checks.
"""

from __future__ import annotations

from typing import Optional

from .protocol import CallableWorkload, WorkloadSpec
from ..architectures.configs import TransformerConfig, get as get_config

_DTYPE = {"fp16": "float16", "fp32": "float32"}


def _torch():
    import torch
    return torch


class _EncDecModel:
    """Lazily-built torch.nn module mirroring the counted encoder-decoder structure.

    Holds an encoder stack (bidirectional) and a decoder stack whose layers carry
    self-attention (causal, growing cache) and cross-attention (to the encoder
    output, static cache). Built so the package imports without torch.
    """

    def __init__(self, cfg: TransformerConfig, dtype_str: str, device: str):
        torch = _torch()
        import torch.nn as nn

        self.cfg = cfg
        torch_dtype_name = _DTYPE.get(dtype_str, dtype_str)
        if not hasattr(torch, torch_dtype_name):
            raise ValueError(
                f"dtype '{dtype_str}' is neither a precision key {list(_DTYPE)} "
                f"nor a torch dtype name"
            )
        self.dtype = getattr(torch, torch_dtype_name)
        self.device = device
        d = cfg.d_model
        self.d_model = d
        self.n_enc_layers = cfg.n_encoder_layers
        self.n_dec_layers = cfg.n_decoder_layers
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.kv_heads
        self.head_dim = cfg.d_head
        self.is_gated = cfg.is_gated

        Norm = nn.LayerNorm

        def lin(i, o):
            return nn.Linear(i, o, bias=False)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_dim = q_dim
        self.kv_dim = kv_dim

        # Shared token embedding (T5/BART tie encoder+decoder embeddings).
        self.embed = nn.Embedding(cfg.vocab_size, d)

        # --- encoder stack (bidirectional, fused QKV, 2 norms/layer) ---
        self.enc_layers = nn.ModuleList()
        for _ in range(cfg.n_encoder_layers):
            layer = nn.ModuleDict()
            layer["norm1"] = Norm(d)
            layer["qkv"] = lin(d, q_dim + 2 * kv_dim)
            layer["out"] = lin(q_dim, d)
            layer["norm2"] = Norm(d)
            self._add_ffn(layer, lin, cfg, d)
            self.enc_layers.append(layer)
        self.enc_final_norm = Norm(d)

        # --- decoder stack (self-attn + cross-attn + FFN, 3 norms/layer) ---
        self.dec_layers = nn.ModuleList()
        for _ in range(cfg.n_decoder_layers):
            layer = nn.ModuleDict()
            # self-attention: fused QKV (causal, growing cache)
            layer["sa_norm"] = Norm(d)
            layer["sa_qkv"] = lin(d, q_dim + 2 * kv_dim)
            layer["sa_out"] = lin(q_dim, d)
            # cross-attention: Q from decoder; K,V from encoder output (separate projections)
            layer["ca_norm"] = Norm(d)
            layer["ca_q"] = lin(d, q_dim)
            layer["ca_k"] = lin(d, kv_dim)
            layer["ca_v"] = lin(d, kv_dim)
            layer["ca_out"] = lin(q_dim, d)
            # FFN
            layer["ffn_norm"] = Norm(d)
            self._add_ffn(layer, lin, cfg, d)
            self.dec_layers.append(layer)
        self.dec_final_norm = Norm(d)
        self.lm_head = lin(d, cfg.vocab_size)

        # Register everything for .to(device, dtype).
        self.module = nn.Module()
        self.module.embed = self.embed
        self.module.enc_layers = self.enc_layers
        self.module.enc_final_norm = self.enc_final_norm
        self.module.dec_layers = self.dec_layers
        self.module.dec_final_norm = self.dec_final_norm
        self.module.lm_head = self.lm_head
        self.module = self.module.to(device=device, dtype=self.dtype).eval()

        self._act = self._activation_fn(cfg.activation)

    @staticmethod
    def _add_ffn(layer, lin, cfg, d):
        if cfg.is_gated:
            layer["gate"] = lin(d, cfg.d_ff)
            layer["up"] = lin(d, cfg.d_ff)
            layer["down"] = lin(cfg.d_ff, d)
        else:
            layer["up"] = lin(d, cfg.d_ff)
            layer["down"] = lin(cfg.d_ff, d)

    @staticmethod
    def _activation_fn(kind: str):
        torch = _torch()
        import torch.nn.functional as F
        return {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}.get(kind, F.gelu)

    def _ffn(self, layer, x):
        if self.is_gated:
            return layer["down"](self._act(layer["gate"](x)) * layer["up"](x))
        return layer["down"](self._act(layer["up"](x)))

    def _sdpa(self, q, k, v, *, causal, attn_kind):
        torch = _torch()
        import torch.nn.functional as F
        if attn_kind == "eager":
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        with torch.nn.attention.sdpa_kernel(backends):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def _split_heads(self, t, n):
        B, S, _ = t.shape
        return t.view(B, S, n, self.head_dim).transpose(1, 2)

    def _maybe_expand_kv(self, k, v):
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        return k, v

    # --- encoder ---
    def encode(self, src_ids, *, attn_kind: str):
        """Bidirectional encoder pass over the source; returns enc_out (B, src, d)."""
        x = self.embed(src_ids)
        for layer in self.enc_layers:
            h = layer["norm1"](x)
            qkv = layer["qkv"](h)
            q = self._split_heads(qkv[..., :self.q_dim], self.n_heads)
            k = self._split_heads(qkv[..., self.q_dim:self.q_dim + self.kv_dim], self.n_kv_heads)
            v = self._split_heads(qkv[..., self.q_dim + self.kv_dim:], self.n_kv_heads)
            k, v = self._maybe_expand_kv(k, v)
            a = self._sdpa(q, k, v, causal=False, attn_kind=attn_kind)
            a = a.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.q_dim)
            x = x + layer["out"](a)
            x = x + self._ffn(layer, layer["norm2"](x))
        return self.enc_final_norm(x)

    # --- cross-attention K/V cache (projected once from enc_out) ---
    def build_cross_cache(self, enc_out):
        """Project K,V from the encoder output for every decoder layer, once.

        Returns a list (per decoder layer) of (k, v) with kv_heads, already
        head-split and GQA-expanded, ready for repeated cross-attention reads.
        This is the static cache: it does NOT change across decode steps.
        """
        cross = []
        for layer in self.dec_layers:
            k = self._split_heads(layer["ca_k"](enc_out), self.n_kv_heads)
            v = self._split_heads(layer["ca_v"](enc_out), self.n_kv_heads)
            k, v = self._maybe_expand_kv(k, v)
            cross.append((k, v))
        return cross

    def new_self_cache(self):
        """Fresh per-decoder-layer self-attention KV cache (grows with target)."""
        return [(None, None) for _ in range(self.n_dec_layers)]

    def _decoder_layer(self, li, layer, x, cross_cache, self_cache, *, causal, attn_kind):
        """One decoder layer: self-attn (cache-aware) -> cross-attn (static) -> FFN."""
        torch = _torch()
        B, S, _ = x.shape
        # --- self-attention (causal; grow self-cache) ---
        h = layer["sa_norm"](x)
        qkv = layer["sa_qkv"](h)
        q = self._split_heads(qkv[..., :self.q_dim], self.n_heads)
        k = self._split_heads(qkv[..., self.q_dim:self.q_dim + self.kv_dim], self.n_kv_heads)
        v = self._split_heads(qkv[..., self.q_dim + self.kv_dim:], self.n_kv_heads)
        if self_cache is not None:
            pk, pv = self_cache[li]
            if pk is not None:
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            self_cache[li] = (k, v)
            causal_eff = False if S == 1 else causal  # single query attends to all cached
        else:
            causal_eff = causal
        ke, ve = self._maybe_expand_kv(k, v)
        a = self._sdpa(q, ke, ve, causal=causal_eff, attn_kind=attn_kind)
        a = a.transpose(1, 2).reshape(B, S, self.q_dim)
        x = x + layer["sa_out"](a)

        # --- cross-attention (Q from decoder; static K,V from cross_cache) ---
        h = layer["ca_norm"](x)
        cq = self._split_heads(layer["ca_q"](h), self.n_heads)
        ck, cv = cross_cache[li]                       # already head-split + expanded
        a = self._sdpa(cq, ck, cv, causal=False, attn_kind=attn_kind)
        a = a.transpose(1, 2).reshape(B, S, self.q_dim)
        x = x + layer["ca_out"](a)

        # --- FFN ---
        x = x + self._ffn(layer, layer["ffn_norm"](x))
        return x

    def decoder_prefill(self, tgt_ids, cross_cache, *, attn_kind: str, self_cache=None):
        """Decoder over tgt_ids (causal self-attn, cross-attn to cross_cache).

        If self_cache is provided, self-attention K/V are written into it (so a
        subsequent decode continues from this context). Returns lm_head logits on
        the last token.
        """
        x = self.embed(tgt_ids)
        for li, layer in enumerate(self.dec_layers):
            x = self._decoder_layer(li, layer, x, cross_cache, self_cache,
                                    causal=True, attn_kind=attn_kind)
        x = self.dec_final_norm(x[:, -1:, :])
        return self.lm_head(x)

    def decode_step(self, tok_ids, cross_cache, self_cache, *, attn_kind: str):
        """One decode step: 1-token self-attn (grow self-cache) + cross-attn (static)."""
        x = self.embed(tok_ids)
        for li, layer in enumerate(self.dec_layers):
            x = self._decoder_layer(li, layer, x, cross_cache, self_cache,
                                    causal=True, attn_kind=attn_kind)
        x = self.dec_final_norm(x)
        return self.lm_head(x)


def build_enc_dec_workload(
    model: str | TransformerConfig,
    *,
    phase: str,                       # "encode" | "decoder_prefill" | "decode"
    src_len: int,
    tgt_len: int = 1,                 # target prompt length for decoder_prefill
    tgt_ctx: int = 128,               # established target context for decode
    decode_tokens: int = 64,          # K: tokens generated per execution (decode)
    decode_mode: str = "growing",     # "growing" | "fixed_step"
    precision: str = "fp16",
    weights: str = "random",          # "random" | "pretrained"
    attn_kind: str = "flash",
    inner_iters: int = 1,
    batch_size: int = 1,
    device_index: int = 0,
    pretrained_id: Optional[str] = None,
) -> CallableWorkload:
    """Construct a runnable encoder-decoder workload for one phase.

    phase="encode": encoder stack over src_len (bidirectional, once).
    phase="decoder_prefill": encoder runs once in setup (un-measured); measured
        run() does the decoder prefill over tgt_len target tokens, projecting and
        caching the cross-attention K/V. Each loop rebuilds caches.
    phase="decode": encoder + a tgt_ctx-length decoder prefill run in setup
        (un-measured) to establish self- and cross-caches; measured run() does
        decode steps. "growing" generates decode_tokens with a growing self-cache
        (cross-cache static); "fixed_step" rebuilds to tgt_ctx and runs one step.
    """
    if precision not in _DTYPE:
        raise ValueError(f"precision must be one of {list(_DTYPE)}, got {precision}")
    if phase not in ("encode", "decoder_prefill", "decode"):
        raise ValueError(f"phase must be encode|decoder_prefill|decode, got {phase}")
    if phase == "decode" and decode_mode not in ("growing", "fixed_step"):
        raise ValueError(f"decode_mode must be 'growing' or 'fixed_step', got {decode_mode}")

    cfg = model if isinstance(model, TransformerConfig) else get_config(model)
    if cfg.arch != "encoder_decoder":
        raise ValueError(f"{cfg.name} is {cfg.arch}, not encoder_decoder")

    if weights == "pretrained":
        return _build_pretrained_enc_dec(
            cfg, phase, src_len, tgt_len, tgt_ctx, decode_tokens, decode_mode,
            precision, attn_kind, inner_iters, batch_size, device_index, pretrained_id)

    return _build_random_enc_dec(
        cfg, phase, src_len, tgt_len, tgt_ctx, decode_tokens, decode_mode,
        precision, attn_kind, inner_iters, batch_size, device_index)


def _build_random_enc_dec(cfg, phase, src_len, tgt_len, tgt_ctx, decode_tokens,
                          decode_mode, precision, attn_kind, inner_iters,
                          batch_size, device_index):
    torch = _torch()
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
    em = _EncDecModel(cfg, precision, device)
    vocab = max(cfg.vocab_size, 1)

    extra = {"phase": phase}
    if phase == "decoder_prefill":
        extra["tgt_len"] = tgt_len
    if phase == "decode":
        extra.update({"tgt_ctx": tgt_ctx, "decode_tokens": decode_tokens,
                      "decode_mode": decode_mode})
    spec = WorkloadSpec(
        model_name=cfg.name, arch=cfg.arch, phase=phase, seq_len=src_len,
        precision=precision, weights="random", attn_kind=attn_kind,
        inner_iters=inner_iters, batch_size=batch_size, extra=extra,
    )

    src_ids = torch.randint(0, vocab, (batch_size, src_len), device=device)

    if phase == "encode":
        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                em.encode(src_ids, attn_kind=attn_kind)

    elif phase == "decoder_prefill":
        # Encoder runs ONCE in setup (un-measured); enc_out is fixed context.
        with torch.no_grad():
            enc_out = em.encode(src_ids, attn_kind=attn_kind)
        tgt_ids = torch.randint(0, vocab, (batch_size, tgt_len), device=device)

        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                # Rebuild the cross-cache each loop (measures prefill repeatedly,
                # including the cross-K/V projection that prefill is defined to do).
                cross = em.build_cross_cache(enc_out)
                em.decoder_prefill(tgt_ids, cross, attn_kind=attn_kind,
                                   self_cache=em.new_self_cache())

    else:  # decode
        # Setup (un-measured): encode + prefill a tgt_ctx-length target to fill
        # both caches to the established context.
        with torch.no_grad():
            enc_out = em.encode(src_ids, attn_kind=attn_kind)
        tok = torch.randint(0, vocab, (batch_size, 1), device=device)
        prompt = torch.randint(0, vocab, (batch_size, tgt_ctx), device=device)

        if decode_mode == "growing":
            @torch.no_grad()
            def run():
                for _ in range(inner_iters):
                    cross = em.build_cross_cache(enc_out)
                    sc = em.new_self_cache()
                    em.decoder_prefill(prompt, cross, attn_kind=attn_kind, self_cache=sc)
                    for _t in range(decode_tokens):
                        em.decode_step(tok, cross, sc, attn_kind=attn_kind)
        else:  # fixed_step
            @torch.no_grad()
            def run():
                for _ in range(inner_iters):
                    cross = em.build_cross_cache(enc_out)
                    sc = em.new_self_cache()
                    em.decoder_prefill(prompt, cross, attn_kind=attn_kind, self_cache=sc)
                    em.decode_step(tok, cross, sc, attn_kind=attn_kind)

    def free():
        nonlocal em
        del em
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _build_pretrained_enc_dec(cfg, phase, src_len, tgt_len, tgt_ctx, decode_tokens,
                              decode_mode, precision, attn_kind, inner_iters,
                              batch_size, device_index, pretrained_id):
    """Pretrained T5/BART via transformers (downloads weights).

    encode: run model.get_encoder() over source ids.
    decoder_prefill / decode: use encoder_outputs + the decoder with use_cache.
    """
    torch = _torch()
    from transformers import AutoModelForSeq2SeqLM
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
    hf_id = pretrained_id or _default_hf_id(cfg.name)
    dtype = getattr(torch, _DTYPE[precision])
    model_obj = AutoModelForSeq2SeqLM.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    vocab = getattr(model_obj.config, "vocab_size", max(cfg.vocab_size, 1))

    extra = {"phase": phase}
    spec = WorkloadSpec(
        model_name=cfg.name, arch=cfg.arch, phase=phase, seq_len=src_len,
        precision=precision, weights="pretrained", attn_kind=attn_kind,
        inner_iters=inner_iters, batch_size=batch_size, extra=extra,
    )

    src_ids = torch.randint(0, vocab, (batch_size, src_len), device=device)
    encoder = model_obj.get_encoder()

    if phase == "encode":
        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                encoder(src_ids)

    elif phase == "decoder_prefill":
        with torch.no_grad():
            enc_out = encoder(src_ids)
        dec_ids = torch.randint(0, vocab, (batch_size, tgt_len), device=device)

        @torch.no_grad()
        def run():
            for _ in range(inner_iters):
                model_obj(encoder_outputs=enc_out, decoder_input_ids=dec_ids, use_cache=True)

    else:  # decode
        with torch.no_grad():
            enc_out = encoder(src_ids)
        tok = torch.randint(0, vocab, (batch_size, 1), device=device)
        prompt = torch.randint(0, vocab, (batch_size, tgt_ctx), device=device)

        if decode_mode == "growing":
            @torch.no_grad()
            def run():
                for _ in range(inner_iters):
                    out = model_obj(encoder_outputs=enc_out, decoder_input_ids=prompt,
                                    use_cache=True)
                    pkv = out.past_key_values
                    for _t in range(decode_tokens):
                        out = model_obj(encoder_outputs=enc_out, decoder_input_ids=tok,
                                        past_key_values=pkv, use_cache=True)
                        pkv = out.past_key_values
        else:  # fixed_step
            @torch.no_grad()
            def run():
                for _ in range(inner_iters):
                    out = model_obj(encoder_outputs=enc_out, decoder_input_ids=prompt,
                                    use_cache=True)
                    model_obj(encoder_outputs=enc_out, decoder_input_ids=tok,
                              past_key_values=out.past_key_values, use_cache=True)

    def free():
        nonlocal model_obj
        del model_obj
        _empty_cache()

    return CallableWorkload(spec=spec, _run=run, _free=free)


def _default_hf_id(name: str) -> str:
    return {
        "T5-base": "t5-base",
        "T5-small": "t5-small",
        "BART-base": "facebook/bart-base",
        "BART-large": "facebook/bart-large",
    }.get(name, name)


def _empty_cache() -> None:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
