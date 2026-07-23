"""Split construction for the EXP-002 fit (fit_plan.md section 5).

Two split families, both deterministic:

1. stratified_split: the Step-2 main-table 80/20 split, seed 42, strata
   (arch, phase-class, precision). Deterministic given the seed: strata are
   processed in sorted order and each stratum is sorted by spec.key before
   the seeded shuffle.

2. extrapolation_split: the pre-registered Step-3 split in both declared
   readings. E1 (strict-literal): forward phases only. E2 (broad, PRIMARY):
   all phases. Train = pre-registered train models with every sequence
   dimension <= TRAIN_DIM_MAX; predict = held-out-model points with any
   sequence dimension == PREDICT_DIM, plus the vision held-out model at its
   native shape (recorded clarification 1: ViT has no sequence axis).

Sequence dimensions per phase mirror what physically ran (see
fit/bridge.py): prefill/encode use seq_len; decoder-only decode uses
seq_len (the context); decoder_prefill uses seq_len (src) and tgt_len;
enc-dec decode uses seq_len (src) and tgt_ctx. Vision models have no
sequence dimensions at all.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence

from ..architectures.configs import get as get_config

FORWARD_PHASES = frozenset({"prefill", "encode"})
DECODE_PHASES = frozenset({"decode", "decoder_prefill"})

TRAIN_DIM_MAX = 1024
PREDICT_DIM = 2048

TRAIN_MODELS: dict[str, frozenset[str]] = {
    "decoder_only": frozenset({"DistilGPT2", "GPT-2", "GPT-2-medium", "GPT-2-large"}),
    "encoder_only": frozenset({"DistilBERT", "BERT-base", "BERT-large", "ViT-B/16"}),
    "encoder_decoder": frozenset({"T5-small", "T5-base", "BART-base"}),
}

HELD_OUT_MODELS = frozenset({"GPT-2-XL", "ViT-L/16", "BART-large"})


def phase_class(phase: str) -> str:
    if phase in FORWARD_PHASES:
        return "forward"
    if phase in DECODE_PHASES:
        return "decode_like"
    raise ValueError(f"unknown phase {phase!r}")


def stratum(record: Mapping) -> tuple[str, str, str]:
    s = record["spec"]
    return (s["arch"], phase_class(s["phase"]), s["precision"])


def stratified_split(records: Sequence[Mapping], *, test_frac: float = 0.2,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic stratified train/test split (main table, decision D2)."""
    by: dict[tuple, list] = defaultdict(list)
    for r in records:
        by[stratum(r)].append(r)
    rng = random.Random(seed)
    train: list[dict] = []
    test: list[dict] = []
    for key in sorted(by):
        group = sorted(by[key], key=lambda r: r["spec"]["key"])
        rng.shuffle(group)
        n_test = max(1, round(test_frac * len(group)))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def _is_vision(model: str) -> bool:
    return bool(getattr(get_config(model), "is_vision", False))


def sequence_dims(spec: Mapping) -> list[int]:
    """The sequence dimensions that physically shaped this point (empty for
    vision models, whose sequence is fixed by the architecture)."""
    if _is_vision(spec["model"]):
        return []
    arch, phase = spec["arch"], spec["phase"]
    if phase in FORWARD_PHASES:
        return [spec["seq_len"]]
    if arch == "decoder_only" and phase == "decode":
        return [spec["seq_len"]]
    if phase == "decoder_prefill":
        return [spec["seq_len"], spec["tgt_len"]]
    if arch == "encoder_decoder" and phase == "decode":
        return [spec["seq_len"], spec["tgt_ctx"]]
    raise ValueError(f"cannot derive dims for arch={arch!r} phase={phase!r}")


def extrapolation_split(records: Sequence[Mapping], reading: str,
                        ) -> tuple[list[dict], list[dict]]:
    """The pre-registered extrapolation split, reading 'E1' or 'E2'.

    Returns (train, predict). Held-out models NEVER appear in train; train
    models never appear in predict. Points on held-out models that are
    neither native-vision nor at the predict dimension are excluded from
    both sides (they exist in the dataset but play no role in this test).
    """
    if reading not in ("E1", "E2"):
        raise ValueError(f"reading must be 'E1' or 'E2', got {reading!r}")
    forward_only = reading == "E1"

    train: list[dict] = []
    predict: list[dict] = []
    for r in records:
        s = r["spec"]
        model, arch, phase = s["model"], s["arch"], s["phase"]
        if model in HELD_OUT_MODELS:
            if _is_vision(model):
                predict.append(r)            # native shape (clarification 1)
                continue
            if forward_only and phase not in FORWARD_PHASES:
                continue
            if PREDICT_DIM in sequence_dims(s):
                predict.append(r)
            continue
        if model not in TRAIN_MODELS.get(arch, frozenset()):
            raise ValueError(f"model {model!r} is neither train nor held-out "
                             f"(key={s.get('key')})")
        if forward_only and phase not in FORWARD_PHASES:
            continue
        if all(d <= TRAIN_DIM_MAX for d in sequence_dims(s)):
            train.append(r)
    return train, predict
