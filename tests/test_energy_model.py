"""Tests for the energy-model family (tomltransformers.energy_model).

These use synthetic data with known ground-truth coefficients, so we can check
coefficient recovery, generalization, non-negativity, and that the model
selection penalizes unnecessary complexity. No GPU or real measurements needed.
"""

import numpy as np
import pytest

from tomltransformers import energy_model as em


def _make_records(**arrays):
    keys = list(arrays)
    n = len(arrays[keys[0]])
    return [{k: float(arrays[k][i]) for k in keys} for i in range(n)]


def _split(records, y, n_train):
    return records[:n_train], y[:n_train], records[n_train:], y[n_train:]


def test_family_param_counts():
    fam = em.model_family()
    assert fam["M0_flops"].n_params == 1
    assert fam["M1_comp_mem"].n_params == 2
    assert fam["M2_overhead"].n_params == 3
    assert fam["M3_dispatch"].n_params == 4
    assert fam["M4_fused"].n_params == 5


def test_design_matrix_shapes_and_columns():
    recs = _make_records(
        to_compute=[1.0, 2.0], to_memory=[3.0, 4.0],
        n_launches=[5.0, 6.0], n_fused_steps=[7.0, 8.0],
    )
    m4 = em.model_family()["M4_fused"]
    A = m4.design_matrix(recs)
    assert A.shape == (2, 5)
    assert A[:, -1].tolist() == [1.0, 1.0]  # intercept column
    assert m4.column_names[-1] == "intercept"

    m0 = em.model_family()["M0_flops"]
    A0 = m0.design_matrix(recs)
    assert A0.shape == (2, 1)
    assert A0[0, 0] == pytest.approx(1.0 + 3.0)  # to_compute + to_memory


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        em.model_family()["M1_comp_mem"].predict([{"to_compute": 1.0, "to_memory": 1.0}])


def test_m4_coefficient_recovery():
    rng = np.random.default_rng(42)
    n = 600
    # Conditioned so every term contributes comparably to total energy.
    to_compute = rng.uniform(0.5e12, 1.5e12, n)
    to_memory = rng.uniform(0.5e12, 1.5e12, n)
    n_launches = rng.uniform(0, 5000, n)
    n_fused = rng.uniform(0, 2048, n)
    a_c, a_m, a_o, a_f, base = 5e-12, 1.5e-11, 1e-3, 2e-3, 2.0
    energy = a_c * to_compute + a_m * to_memory + a_o * n_launches + a_f * n_fused + base
    energy *= 1 + 0.01 * rng.standard_normal(n)  # 1% measurement noise

    recs = _make_records(
        to_compute=to_compute, to_memory=to_memory,
        n_launches=n_launches, n_fused_steps=n_fused,
    )
    m4 = em.model_family()["M4_fused"].fit(recs, energy)
    # coef order matches column order: [compute, memory, launches, fused, intercept]
    recovered = dict(zip(m4.column_names, m4.coef_))
    assert recovered["to_compute"] == pytest.approx(a_c, rel=0.05)
    assert recovered["to_memory"] == pytest.approx(a_m, rel=0.05)
    assert recovered["n_launches"] == pytest.approx(a_o, rel=0.10)
    assert recovered["n_fused_steps"] == pytest.approx(a_f, rel=0.10)
    assert recovered["intercept"] == pytest.approx(base, rel=0.25)


def test_nnls_coefficients_nonnegative():
    rng = np.random.default_rng(1)
    n = 300
    to_compute = rng.uniform(1e12, 1e13, n)
    to_memory = rng.uniform(1e12, 1e13, n)
    energy = 5e-12 * to_compute + 2e-11 * to_memory
    recs = _make_records(to_compute=to_compute, to_memory=to_memory,
                         n_launches=np.zeros(n), n_fused_steps=np.zeros(n))
    for m in em.model_family().values():
        m.fit(recs, energy)
        assert np.all(m.coef_ >= 0.0), m.name


def test_separating_compute_and_memory_beats_single_total():
    # When compute and memory have very different per-TO costs, a single
    # total-TO coefficient (M0, calibrated FLOPs) underfits. This is the
    # signals-paper result: separating terms is essential.
    rng = np.random.default_rng(7)
    n = 600
    to_compute = rng.uniform(1e12, 1e13, n)
    to_memory = rng.uniform(1e12, 1e13, n)  # independent of compute
    energy = 5e-12 * to_compute + 5e-11 * to_memory  # 10x different coefficients
    energy *= 1 + 0.01 * rng.standard_normal(n)
    recs = _make_records(to_compute=to_compute, to_memory=to_memory,
                         n_launches=np.zeros(n), n_fused_steps=np.zeros(n))
    rtr, ytr, rte, yte = _split(recs, energy, 400)
    results = em.fit_and_select(rtr, ytr, rte, yte)

    by_name = {r.name: r for r in results}
    assert by_name["M1_comp_mem"].r2_test > by_name["M0_flops"].r2_test
    assert by_name["M0_flops"].r2_test < 0.97          # M0 genuinely underfits
    assert by_name["M1_comp_mem"].r2_test > 0.99       # separation fits well
    assert em.best_by(results, "r2_test").name != "M0_flops"


def test_information_criteria_penalize_unneeded_terms():
    # Data generated from M1 structure (compute + memory only). The dispatch and
    # fused terms are genuinely absent, so BIC should not select M4.
    rng = np.random.default_rng(123)
    n = 600
    to_compute = rng.uniform(1e12, 1e13, n)
    to_memory = rng.uniform(1e12, 1e13, n)
    energy = 6e-12 * to_compute + 2e-11 * to_memory
    energy *= 1 + 0.01 * rng.standard_normal(n)
    # Provide non-trivial (but causally irrelevant) launch/fused features.
    recs = _make_records(
        to_compute=to_compute, to_memory=to_memory,
        n_launches=rng.uniform(0, 5000, n), n_fused_steps=rng.uniform(0, 2048, n),
    )
    rtr, ytr, rte, yte = _split(recs, energy, 400)
    results = em.fit_and_select(rtr, ytr, rte, yte)

    winner = em.best_by(results, "bic")
    assert winner.n_params <= 3            # M0/M1/M2, not the bloated M3/M4
    assert winner.name != "M4_fused"


def test_metrics_basic_behavior():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert em.r2_score(y, y) == pytest.approx(1.0)
    assert em.mape(y, y) == pytest.approx(0.0)
    # A constant-mean prediction yields R^2 = 0 by construction.
    assert em.r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)
    # MAPE excludes zero-energy targets rather than dividing by zero.
    assert np.isfinite(em.mape(np.array([0.0, 100.0]), np.array([5.0, 110.0])))


def test_summary_table_runs():
    rng = np.random.default_rng(0)
    n = 200
    to_compute = rng.uniform(1e12, 1e13, n)
    to_memory = rng.uniform(1e12, 1e13, n)
    energy = 5e-12 * to_compute + 2e-11 * to_memory + 0.5
    recs = _make_records(to_compute=to_compute, to_memory=to_memory,
                         n_launches=np.zeros(n), n_fused_steps=np.zeros(n))
    rtr, ytr, rte, yte = _split(recs, energy, 140)
    table = em.summary_table(em.fit_and_select(rtr, ytr, rte, yte))
    assert "model" in table and "AIC" in table and "M0_flops" in table
