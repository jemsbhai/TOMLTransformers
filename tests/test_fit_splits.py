"""Split-construction tests (fit_plan.md section 5, decisions D2 and D-E).

All invariants are structural, run against the REAL frozen dataset, and
never hardcode counts that were not pre-registered: exact set sizes are
printed by the fit script and reviewed there.
"""

from pathlib import Path

import pytest

from tomltransformers.fit.bridge import load_latest_records
from tomltransformers.fit.splits import (FORWARD_PHASES, HELD_OUT_MODELS,
                                         PREDICT_DIM, TRAIN_DIM_MAX,
                                         extrapolation_split, sequence_dims,
                                         stratified_split, stratum)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "experiments" / "exp_002_size_sweep" / "energy.jsonl"


@pytest.fixture(scope="module")
def records():
    recs = load_latest_records(DATA)
    assert len(recs) == 296
    return recs


def _keys(recs):
    return [r["spec"]["key"] for r in recs]


# ------------------------------------------------------------------------------
# Main-table stratified split (D2)
# ------------------------------------------------------------------------------
def test_stratified_split_deterministic(records):
    a_train, a_test = stratified_split(records, seed=42)
    b_train, b_test = stratified_split(records, seed=42)
    assert _keys(a_train) == _keys(b_train)
    assert _keys(a_test) == _keys(b_test)


def test_stratified_split_disjoint_and_complete(records):
    train, test = stratified_split(records, seed=42)
    train_k, test_k = set(_keys(train)), set(_keys(test))
    assert not (train_k & test_k)
    assert len(train_k | test_k) == len(records)


def test_stratified_split_fraction_per_stratum(records):
    train, test = stratified_split(records, test_frac=0.2, seed=42)
    from collections import Counter
    n_test = Counter(stratum(r) for r in test)
    n_all = Counter(stratum(r) for r in records)
    for st, n in n_all.items():
        expected = max(1, round(0.2 * n))
        assert n_test[st] == expected, (st, n, n_test[st])


def test_stratified_split_seed_matters(records):
    a_test = set(_keys(stratified_split(records, seed=42)[1]))
    b_test = set(_keys(stratified_split(records, seed=43)[1]))
    assert a_test != b_test


# ------------------------------------------------------------------------------
# Pre-registered extrapolation split (D-E), both readings
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("reading", ["E1", "E2"])
def test_extrapolation_membership(records, reading):
    train, predict = extrapolation_split(records, reading)
    assert train and predict
    for r in train:
        s = r["spec"]
        assert s["model"] not in HELD_OUT_MODELS
        assert all(d <= TRAIN_DIM_MAX for d in sequence_dims(s))
        if reading == "E1":
            assert s["phase"] in FORWARD_PHASES
    for r in predict:
        s = r["spec"]
        assert s["model"] in HELD_OUT_MODELS
        if s["model"] != "ViT-L/16":
            assert PREDICT_DIM in sequence_dims(s)
            if reading == "E1":
                assert s["phase"] in FORWARD_PHASES


def test_extrapolation_e1_subset_of_e2(records):
    t1, p1 = extrapolation_split(records, "E1")
    t2, p2 = extrapolation_split(records, "E2")
    assert set(_keys(t1)) <= set(_keys(t2))
    assert set(_keys(p1)) <= set(_keys(p2))


@pytest.mark.parametrize("reading", ["E1", "E2"])
def test_extrapolation_vit_l_native_points_predicted(records, reading):
    _, predict = extrapolation_split(records, reading)
    vit_l = [r for r in predict if r["spec"]["model"] == "ViT-L/16"]
    precisions = {r["spec"]["precision"] for r in vit_l}
    assert precisions == {"fp16", "fp32"}


def test_extrapolation_e2_covers_all_three_classes(records):
    _, predict = extrapolation_split(records, "E2")
    arches = {r["spec"]["arch"] for r in predict}
    assert arches == {"decoder_only", "encoder_only", "encoder_decoder"}


def test_extrapolation_split_sides_disjoint(records):
    for reading in ("E1", "E2"):
        train, predict = extrapolation_split(records, reading)
        assert not (set(_keys(train)) & set(_keys(predict)))
