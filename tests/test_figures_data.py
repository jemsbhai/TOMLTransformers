"""Tests for the figure data layer (Step 8a).

The central test is the lineage gate: every per-point quantity the figures
recompute must reproduce the committed artifact it is derived from. If this
suite goes red, either the frozen data moved or the figures would have been
drawn from numbers that disagree with the recorded verdicts.
"""

from __future__ import annotations

import pytest

from tomltransformers.figures import data as fd


@pytest.fixture(scope="module")
def gate():
    return fd.run_lineage_gate(strict=False)


def test_lineage_gate_all_checks_pass(gate):
    bad = [r for r in gate if not r.ok]
    assert not bad, "\n".join(r.line() for r in bad)


def test_lineage_gate_is_not_vacuous(gate):
    # Guard against a gate that silently checks nothing.
    assert len(gate) >= 25
    names = {r.name for r in gate}
    for expected in ("4090 pooled A-B median", "A100 pooled A-B median",
                     "A100 T1 held-out MAPE", "A100 T3 pooled MAPE",
                     "4090 fp32/fp16 forward med", "A100 fp32/fp16 forward med"):
        assert expected in names


def test_lineage_gate_strict_raises_when_a_check_fails(monkeypatch):
    real = fd._close
    calls = {"n": 0}

    def flaky(a, b, rtol=fd.GATE_RTOL):
        calls["n"] += 1
        return False if calls["n"] == 1 else real(a, b, rtol)

    monkeypatch.setattr(fd, "_close", flaky)
    with pytest.raises(fd.GateFailure):
        fd.run_lineage_gate(strict=True)


# ----------------------------------------------------------------------
# Frozen dataset shape
# ----------------------------------------------------------------------

def test_record_counts():
    assert len(fd.load_records(fd.R4090_ENERGY)) == 296
    assert len(fd.load_records(fd.A100_ENERGY)) == 98


def test_prediction_row_counts_and_columns():
    p4090 = fd.load_4090_predictions()
    assert len(p4090) == 296
    for col in ("y_j", "yhat_full_j", "yhat_r1_j", "in_main_test",
                "mcer_fit", "mcer_fit_r1"):
        assert col in p4090[0]

    pa100 = fd.load_a100_predictions()
    assert len(pa100) == 98
    for col in ("y_j", "yhat_t1_train_fit_j", "yhat_t2_scaled_j",
                "yhat_t3_all84_fit_j", "t1_role", "t2_role", "stratum"):
        assert col in pa100[0]


def test_a100_strata_partition():
    rows = fd.load_a100_predictions()
    counts = {}
    for r in rows:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    assert counts == {"shared": 84, "extension": 10, "spot": 4}


# ----------------------------------------------------------------------
# Precision pairs (figure F4)
# ----------------------------------------------------------------------

def test_precision_pairs_match_validation_reports():
    p4090 = fd.precision_pairs(fd.load_records(fd.R4090_ENERGY))
    assert len(p4090) == 143
    assert sum(1 for p in p4090 if p["inversion"]) == 0

    pa100 = fd.precision_pairs(fd.load_records(fd.A100_ENERGY))
    assert len(pa100) == 34
    assert sum(1 for p in pa100 if p["inversion"]) == 0


def test_forward_ratio_ranges_are_disjoint_across_platforms():
    """The recorded claim: A100 min 6.20 sits above the 4090 max 4.11."""
    f4090 = fd.forward_ratios(fd.precision_pairs(fd.load_records(fd.R4090_ENERGY)))
    fa100 = fd.forward_ratios(fd.precision_pairs(fd.load_records(fd.A100_ENERGY)))
    assert len(f4090) == 62
    assert len(fa100) == 20
    assert min(fa100) > max(f4090)


def test_a100_forward_ratios_all_exceed_the_model_implied_ceiling():
    fa100 = fd.forward_ratios(fd.precision_pairs(fd.load_records(fd.A100_ENERGY)))
    assert min(fa100) > fd.MODEL_IMPLIED_RATIO_CEILING


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def test_mape_of_matches_the_energy_model_definition():
    from tomltransformers.energy_model import mape
    import numpy as np

    y = [1.0, 2.0, 4.0]
    yhat = [1.1, 1.5, 5.0]
    assert fd.mape_of(zip(y, yhat)) == pytest.approx(
        mape(np.array(y), np.array(yhat)))


def test_validator_is_imported_by_path_not_reimplemented():
    v = fd.validator()
    assert v.PAIR_EXCLUDE == ("precision", "key", "seed")
    assert callable(v.shape_pair_key)
    # The pairing key must ignore precision, so twins collide by construction.
    a = {"model": "GPT-2", "phase": "prefill", "precision": "fp16",
         "key": "k16", "seed": 1}
    b = dict(a, precision="fp32", key="k32", seed=2)
    assert v.shape_pair_key(a) == v.shape_pair_key(b)


def test_ab_percentages_are_fractions_not_percents():
    rows = fd.ab_percentages(fd.load_records(fd.A100_ENERGY))
    assert len(rows) == 98
    assert max(r["ab"] for r in rows) < 0.5
    assert {r["phase_class"] for r in rows} == {"forward", "decode_like"}
