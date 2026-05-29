"""Tests for the energy-model family (tomltransformers.energy_model).

Synthetic data with known coefficients. The headline additions over the basic
suite: the memory-split model (M5) recovers distinct SRAM and HBM coefficients
and beats the aggregate model when the true ratio differs from the prior (the
node-correction case), and information criteria still reject the extra split
parameters when they are redundant. No GPU or measurements needed.
"""

import numpy as np
import pytest

from tomltransformers import energy_model as em


def _records(**arrays):
    keys = list(arrays)
    n = len(arrays[keys[0]])
    return [{k: float(arrays[k][i]) for k in keys} for i in range(n)]


def _split(records, y, n_train):
    return records[:n_train], y[:n_train], records[n_train:], y[n_train:]


# --- Family structure --------------------------------------------------------
def test_family_param_counts():
    fam = em.model_family()
    expected = {
        "M0_flops": 1, "M1_comp_mem": 2, "M2_overhead": 3, "M3_dispatch": 4,
        "M4_fused": 5, "M5_mem_split": 4, "M6_compute_split": 4,
        "M7_full_split": 5, "M8_split_dispatch": 6, "M9_full": 7,
    }
    assert {k: v.n_params for k, v in fam.items()} == expected


def test_m0_sums_all_base_features():
    recs = _records(to_mac=[1.0], to_nonlinear=[2.0], to_sram=[3.0], to_hbm=[4.0])
    A = em.model_family()["M0_flops"].design_matrix(recs)
    assert A.shape == (1, 1)
    assert A[0, 0] == pytest.approx(1.0 + 2.0 + 3.0 + 4.0)


def test_column_names():
    fam = em.model_family()
    assert fam["M5_mem_split"].column_names == ["to_mac+to_nonlinear", "to_sram", "to_hbm", "intercept"]
    assert fam["M7_full_split"].column_names[-1] == "intercept"


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        em.model_family()["M1_comp_mem"].predict([{"to_mac": 1.0}])


# --- Coefficient recovery ----------------------------------------------------
def test_full_split_recovers_four_distinct_coefficients():
    rng = np.random.default_rng(42)
    n = 600
    to_mac = rng.uniform(1.0, 10.0, n)
    to_nl = rng.uniform(1.0, 10.0, n)
    to_sram = rng.uniform(1.0, 10.0, n)
    to_hbm = rng.uniform(1.0, 10.0, n)
    a_mac, a_nl, a_sram, a_hbm, base = 1.0, 4.0, 0.5, 3.0, 1.0
    energy = (a_mac * to_mac + a_nl * to_nl + a_sram * to_sram + a_hbm * to_hbm + base)
    energy *= 1 + 0.01 * rng.standard_normal(n)

    recs = _records(to_mac=to_mac, to_nonlinear=to_nl, to_sram=to_sram, to_hbm=to_hbm)
    m7 = em.model_family()["M7_full_split"].fit(recs, energy)
    got = dict(zip(m7.column_names, m7.coef_))
    assert got["to_mac"] == pytest.approx(a_mac, rel=0.08)
    assert got["to_nonlinear"] == pytest.approx(a_nl, rel=0.08)
    assert got["to_sram"] == pytest.approx(a_sram, rel=0.10)
    assert got["to_hbm"] == pytest.approx(a_hbm, rel=0.08)
    assert got["intercept"] == pytest.approx(base, rel=0.30)


# --- The node-correction case ------------------------------------------------
def test_mem_split_recovers_ratio_and_beats_aggregate():
    # True SRAM and HBM effective costs differ by 6x (the prior ratio is wrong).
    # The aggregate memory coefficient (M2) cannot capture this when the
    # SRAM/HBM mix varies; the split model (M5) can.
    rng = np.random.default_rng(7)
    n = 600
    to_mac = rng.uniform(1.0, 10.0, n)
    to_nl = rng.uniform(1.0, 10.0, n)
    to_sram = rng.uniform(1.0, 10.0, n)
    to_hbm = rng.uniform(1.0, 10.0, n)  # independent of sram -> mix varies
    a_c, a_sram, a_hbm, base = 1.0, 0.5, 3.0, 2.0
    energy = a_c * (to_mac + to_nl) + a_sram * to_sram + a_hbm * to_hbm + base
    energy *= 1 + 0.01 * rng.standard_normal(n)

    recs = _records(to_mac=to_mac, to_nonlinear=to_nl, to_sram=to_sram, to_hbm=to_hbm)
    rtr, ytr, rte, yte = _split(recs, energy, 400)
    results = em.fit_and_select(rtr, ytr, rte, yte)
    by_name = {r.name: r for r in results}

    # M5 recovers the two distinct memory coefficients.
    m5 = em.model_family()["M5_mem_split"].fit(rtr, ytr)
    got = dict(zip(m5.column_names, m5.coef_))
    assert got["to_sram"] == pytest.approx(a_sram, rel=0.12)
    assert got["to_hbm"] == pytest.approx(a_hbm, rel=0.08)

    # Splitting memory generalizes better than lumping it.
    assert by_name["M5_mem_split"].r2_test > by_name["M2_overhead"].r2_test
    assert by_name["M5_mem_split"].aic < by_name["M2_overhead"].aic


def test_compute_split_beats_aggregate_when_mac_and_nonlinear_differ():
    rng = np.random.default_rng(11)
    n = 600
    to_mac = rng.uniform(1.0, 10.0, n)
    to_nl = rng.uniform(1.0, 10.0, n)
    to_sram = rng.uniform(1.0, 10.0, n)
    to_hbm = rng.uniform(1.0, 10.0, n)
    a_mac, a_nl, a_m, base = 1.0, 5.0, 1.0, 1.0  # nonlinear 5x more costly than MAC
    energy = a_mac * to_mac + a_nl * to_nl + a_m * (to_sram + to_hbm) + base
    energy *= 1 + 0.01 * rng.standard_normal(n)

    recs = _records(to_mac=to_mac, to_nonlinear=to_nl, to_sram=to_sram, to_hbm=to_hbm)
    rtr, ytr, rte, yte = _split(recs, energy, 400)
    by_name = {r.name: r for r in em.fit_and_select(rtr, ytr, rte, yte)}
    assert by_name["M6_compute_split"].r2_test > by_name["M2_overhead"].r2_test


# --- Non-negativity ----------------------------------------------------------
def test_nnls_coefficients_nonnegative():
    rng = np.random.default_rng(1)
    n = 300
    recs = _records(
        to_mac=rng.uniform(1, 10, n), to_nonlinear=rng.uniform(1, 10, n),
        to_sram=rng.uniform(1, 10, n), to_hbm=rng.uniform(1, 10, n),
        n_launches=rng.uniform(0, 5, n), n_fused_steps=rng.uniform(0, 5, n),
    )
    energy = np.array([2 * r["to_mac"] + 3 * r["to_hbm"] + 1.0 for r in recs])
    for m in em.model_family().values():
        m.fit(recs, energy)
        assert np.all(m.coef_ >= 0.0), m.name


# --- Information criteria penalize unneeded complexity -----------------------
def test_ic_rejects_redundant_splits():
    # Data generated with a single compute coeff and a single memory coeff
    # (SRAM and HBM share the same effective cost), no dispatch/fused terms.
    # The split and dispatch models add genuinely redundant parameters.
    rng = np.random.default_rng(123)
    n = 600
    to_mac = rng.uniform(1, 10, n)
    to_nl = rng.uniform(1, 10, n)
    to_sram = rng.uniform(1, 10, n)
    to_hbm = rng.uniform(1, 10, n)
    a_c, a_m, base = 2.0, 1.5, 1.0
    energy = a_c * (to_mac + to_nl) + a_m * (to_sram + to_hbm) + base
    energy *= 1 + 0.01 * rng.standard_normal(n)
    recs = _records(
        to_mac=to_mac, to_nonlinear=to_nl, to_sram=to_sram, to_hbm=to_hbm,
        n_launches=rng.uniform(0, 5, n), n_fused_steps=rng.uniform(0, 5, n),
    )
    rtr, ytr, rte, yte = _split(recs, energy, 400)
    winner = em.best_by(em.fit_and_select(rtr, ytr, rte, yte), "bic")
    assert winner.n_params <= 3                       # M0/M1/M2, not a split/dispatch model
    assert winner.name not in ("M7_full_split", "M8_split_dispatch", "M9_full")


# --- Metrics and table -------------------------------------------------------
def test_metrics_basic_behavior():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert em.r2_score(y, y) == pytest.approx(1.0)
    assert em.mape(y, y) == pytest.approx(0.0)
    assert em.r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)
    assert np.isfinite(em.mape(np.array([0.0, 100.0]), np.array([5.0, 110.0])))


def test_summary_table_runs():
    rng = np.random.default_rng(0)
    n = 200
    recs = _records(
        to_mac=rng.uniform(1, 10, n), to_nonlinear=rng.uniform(1, 10, n),
        to_sram=rng.uniform(1, 10, n), to_hbm=rng.uniform(1, 10, n),
    )
    energy = np.array([r["to_mac"] + 2 * r["to_hbm"] + 0.5 for r in recs])
    rtr, ytr, rte, yte = _split(recs, energy, 140)
    table = em.summary_table(em.fit_and_select(rtr, ytr, rte, yte))
    assert "model" in table and "M9_full" in table and "M0_flops" in table
