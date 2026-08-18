#!/usr/bin/env python
"""EXP-002 A100 phase fit: the pre-registered tests T0-T3 and secondaries of
experiments/exp_002_size_sweep/a100_amendment.md (sections 3, 9, 10 and 16).

Run from the repo root with the code committed (artifact provenance records
the commit):

    python scripts/fit_exp002_a100.py

Reads:
    experiments/exp_002_size_sweep/a100/energy.jsonl     (98 frozen records)
    experiments/exp_002_size_sweep/fit/fit_results.json  (frozen 4090 vectors,
                                                          db1f984 lineage)
    configs/exp_002_a100.yaml                            (strata by pass name)
Writes:
    experiments/exp_002_size_sweep/a100/fit/fit_report.txt
    experiments/exp_002_size_sweep/a100/fit/fit_results.json
    experiments/exp_002_size_sweep/a100/fit/per_point_predictions.jsonl

Estimators: the R1 relative-error NNLS (PRIMARY, amendment section 3), the
absolute NNLS (secondary), the target y = per_execution_median_j["B"] /
inner_iters and the per-record units gate are imported BY PATH from
scripts/fit_exp002.py, so R1 is exactly the frozen 4090 implementation and
that script is not modified. Model FORM is frozen (M8_split_dispatch); no
model selection is run here.

Sets are resolved mechanically: shared 84 / extension 10 / spot 4 by pass
name (fit/a100_strata.py); the T2 calibration cells by explicit seed-less
keys (fit/a100_calibration.py, amendment section 16.2). Spot cells enter no
fit and no test. Verdicts print as they fall; nothing here is re-selected.

Deterministic (seed 42 stratified split; NNLS is deterministic; the T2 scalar
is closed-form).
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tomltransformers import to_costs as tc                              # noqa: E402
from tomltransformers.energy_model import EnergyModel, mape, r2_score    # noqa: E402
from tomltransformers.fit.a100_calibration import (                      # noqa: E402
    T2_CALIBRATION_KEYS, T2_ENC_DEC_DECODE_KEY, T2_SENSITIVITY_ALTERNATES,
    seedless_key)
from tomltransformers.fit.a100_strata import partition_records          # noqa: E402
from tomltransformers.fit.baselines import (                             # noqa: E402
    LayerwiseBaseline, RooflineBaseline, TDP_W_BY_DEVICE, roofline_constants)
from tomltransformers.fit.bridge import features_for_spec, load_latest_records  # noqa: E402
from tomltransformers.fit.splits import phase_class, stratified_split, stratum   # noqa: E402
from tomltransformers.fit.stats import ALPHA, ape_pct, holm_adjust, wilcoxon_less  # noqa: E402
from tomltransformers.sweep.grid import load_config                     # noqa: E402


def _load_frozen_fit_script():
    """The 4090 fit script, loaded by path: its module level defines only
    constants and functions (main is guarded), so this executes no fit."""
    p = REPO / "scripts" / "fit_exp002.py"
    spec = importlib.util.spec_from_file_location("fit_exp002_frozen", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load_frozen_fit_script()

RUN_DIR = REPO / "experiments" / "exp_002_size_sweep"
A100_DIR = RUN_DIR / "a100"
DATA = A100_DIR / "energy.jsonl"
FIT_DIR = A100_DIR / "fit"
CFG = REPO / "configs" / "exp_002_a100.yaml"
FROZEN_4090 = RUN_DIR / "fit" / "fit_results.json"

DEVICE = "a100"
FORM = "M8_split_dispatch"
SEED = 42

T0_TARGET = 0.05        # pooled A-B median (fixed, mirrors the 4090)
T1_BAND = 25.0          # D1
T2_BAND = 30.0          # D2
T3_BAND = 30.0          # D3

FROZEN_4090_COMMIT_PREFIX = "db1f984"
# The frozen 4090 vectors (winner form, all-296 fits) as committed at db1f984,
# pinned here so T2 cannot silently run on a regenerated artifact.
FROZEN_R1_4090 = {
    "to_mac": 3.441445472855863e-15, "to_nonlinear": 0.0,
    "to_sram": 2.6928653279927877e-14, "to_hbm": 7.408168956099821e-15,
    "n_launches": 0.000757073086298195, "intercept": 0.0,
}
FROZEN_ABS_4090 = {
    "to_mac": 2.7039919934497613e-15, "to_nonlinear": 0.0,
    "to_sram": 1.968317523311135e-12, "to_hbm": 5.369672178902042e-15,
    "n_launches": 0.0009264997869067472, "intercept": 0.0,
}

HBM_WORD_A100 = tc.mem_word(tc.offchip_tier("a100"))       # 192,000 fJ (hbm2e)
HBM_WORD_4090 = tc.mem_word(tc.offchip_tier("rtx4090"))    # 240,000 fJ (gddr6)

LINES: list[str] = []


def out(s: str = "") -> None:
    LINES.append(s)
    print(s)


def _jf(v):
    v = float(v)
    return v if math.isfinite(v) else None


def model_from_coef(name: str, coef: dict) -> EnergyModel:
    m = F.fresh(name)
    m.coef_ = np.array([float(coef[c]) for c in m.column_names])
    return m


def coef_dict(m: EnergyModel) -> dict:
    return dict(zip(m.column_names, map(float, m.coef_)))


def signed_rel_pct(y, yhat) -> np.ndarray:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return (yhat - y) / y * 100.0


def breakdown(recs, ys, yhat, key_fn) -> dict:
    """{label: {n, mape, mean_signed_pct}} for a labelling of the points."""
    g = defaultdict(lambda: ([], []))
    for r, y, yh in zip(recs, ys, yhat):
        a, b = g[key_fn(r)]
        a.append(float(y))
        b.append(float(yh))
    res = {}
    for k in sorted(g):
        yt, yp = map(np.array, g[k])
        res[str(k)] = {"n": int(len(yt)),
                       "mape": float(mape(yt, yp)),
                       "mean_signed_pct": float(np.mean(signed_rel_pct(yt, yp)))}
    return res


def print_breakdown(title: str, bd: dict) -> None:
    out(f"    {title}:")
    for k, v in bd.items():
        out(f"      {k:26s} n={v['n']:3d}  MAPE {v['mape']:7.2f}%  "
            f"mean signed {v['mean_signed_pct']:+8.2f}%")


def label_prec(r):
    return r["spec"]["precision"]


def label_pc(r):
    return phase_class(r["spec"]["phase"])


def label_prec_pc(r):
    return f"{r['spec']['precision']}/{phase_class(r['spec']['phase'])}"


def label_arch(r):
    return r["spec"]["arch"]


def calib_scalar_relative(yhat, y) -> float:
    """argmin_s sum(((s*yhat - y)/y)^2), closed form."""
    r = np.asarray(yhat, float) / np.asarray(y, float)
    return float(np.sum(r) / np.sum(r * r))


def calib_scalar_absolute(yhat, y) -> float:
    """argmin_s sum((s*yhat - y)^2), closed form."""
    yh = np.asarray(yhat, float)
    yv = np.asarray(y, float)
    return float(np.dot(yh, yv) / np.dot(yh, yh))


def regime_of_extension(spec) -> str:
    if spec["phase"] == "prefill":
        return "prefill_s8192" if int(spec["seq_len"]) == 8192 else "prefill_s1024_4096"
    return "decode"


def verdict(value: float, band: float) -> str:
    return "PASS" if value <= band else "FAIL"


def main() -> int:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    commit = F.git_commit()

    out("EXP-002 A100 phase fit report (a100_amendment.md sections 3, 9, 10, 16)")
    out(f"generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    out(f"git commit: {commit}")
    out(f"data: {DATA}")
    out(f"frozen 4090 vectors: {FROZEN_4090}")
    out(f"estimator: R1 relative-error NNLS (primary), absolute NNLS (secondary), "
        f"imported by path from scripts/fit_exp002.py; form {FORM} frozen")
    out(f"device registry: {DEVICE} -> {tc.offchip_tier(DEVICE)} "
        f"({HBM_WORD_A100:,.0f} fJ/word); 4090 -> {tc.offchip_tier('rtx4090')} "
        f"({HBM_WORD_4090:,.0f} fJ/word)")

    # ------------------------------------------------------------------
    records = load_latest_records(DATA)
    assert len(records) == 98, len(records)
    assert all(r.get("ok") and not r.get("short_window") for r in records)
    cfg = load_config(str(CFG))
    parts = partition_records(records, cfg)
    shared, ext, spot = parts["shared"], parts["extension"], parts["spot"]
    assert (len(shared), len(ext), len(spot)) == (84, 10, 4), \
        (len(shared), len(ext), len(spot))

    feat_of = {r["spec"]["key"]: features_for_spec(r["spec"], device=DEVICE)
               for r in records}
    y_of = {r["spec"]["key"]: F.target_y(r) for r in records}
    F.units_gate(records, [y_of[r["spec"]["key"]] for r in records])

    def FE(rs):
        return [feat_of[r["spec"]["key"]] for r in rs]

    def Y(rs):
        return [y_of[r["spec"]["key"]] for r in rs]

    def P(rs):
        return [r["spec"]["precision"] for r in rs]

    out(f"records: 98 (all ok, none short-window); strata by pass name: "
        f"shared {len(shared)}, extension {len(ext)}, spot {len(spot)}")
    out("target: per_execution_median_j['B'] / inner_iters = J per composite "
        "execution; per_unit consistency gate passed on all 98")
    J: dict = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "git_commit": commit, "device": DEVICE, "form": FORM, "seed": SEED,
               "n_records": 98, "strata": {"shared": 84, "extension": 10, "spot": 4},
               "verdicts": {}}

    # ------------------------------------------------------------------
    out()
    out("== T0. Instrument agreement (pre-registered: pooled A-B median <= 5%) ==")
    ab_all = [float(r["agreement"]["A-B"]) for r in records]
    ab_fwd = [float(r["agreement"]["A-B"]) for r in records
              if phase_class(r["spec"]["phase"]) == "forward"]
    ab_dec = [float(r["agreement"]["A-B"]) for r in records
              if phase_class(r["spec"]["phase"]) == "decode_like"]
    ab_shared = [float(r["agreement"]["A-B"]) for r in shared]
    bc_all = [float(r["agreement"]["B-C"]) for r in records if "B-C" in r["agreement"]]
    t0_pooled = statistics.median(ab_all)
    t0_verdict = verdict(t0_pooled, T0_TARGET)
    out(f"  pooled A-B median over all 98 points: {100 * t0_pooled:.2f}%  "
        f"(forward {100 * statistics.median(ab_fwd):.2f}%, decode-like "
        f"{100 * statistics.median(ab_dec):.2f}%; shared-84 median "
        f"{100 * statistics.median(ab_shared):.2f}%)")
    out(f"  B-C (Zeus) median over {len(bc_all)} points: "
        f"{100 * statistics.median(bc_all):.3f}%  (descriptive)")
    out(f"  T0 verdict: {t0_verdict} ({100 * t0_pooled:.2f}% vs 5%)")
    J["t0"] = {"pooled_ab_median": t0_pooled, "forward_median": statistics.median(ab_fwd),
               "decode_like_median": statistics.median(ab_dec),
               "shared_median": statistics.median(ab_shared),
               "bc_median": statistics.median(bc_all), "band": T0_TARGET,
               "verdict": t0_verdict}
    J["verdicts"]["T0"] = t0_verdict

    # ------------------------------------------------------------------
    fr = json.loads(FROZEN_4090.read_text(encoding="utf-8"))
    assert fr["winner"] == FORM, fr["winner"]
    r1_4090 = fr["exploratory"]["r1_final_coef"]
    abs_4090 = fr["final_coef_all296"]
    for label, loaded, pinned in (("R1", r1_4090, FROZEN_R1_4090),
                                  ("absolute", abs_4090, FROZEN_ABS_4090)):
        for c, v in pinned.items():
            assert math.isclose(float(loaded[c]), v, rel_tol=1e-9, abs_tol=1e-30), \
                (label, c, loaded[c], v)
    lineage_ok = str(fr.get("git_commit", "")).startswith(FROZEN_4090_COMMIT_PREFIX)
    m4090_r1 = model_from_coef(FORM, r1_4090)
    m4090_abs = model_from_coef(FORM, abs_4090)
    out()
    out("== Frozen 4090 vectors (winner form, all-296 fits) ==")
    out(f"  artifact commit {str(fr.get('git_commit'))[:9]} "
        f"({'db1f984 lineage' if lineage_ok else 'NOTE: commit differs from db1f984; values verified against the pinned vector'})")
    out(f"  R1:       {F.coef_line(m4090_r1)}")
    out(f"  absolute: {F.coef_line(m4090_abs)}")
    J["frozen_4090"] = {"artifact_commit": fr.get("git_commit"), "lineage_db1f984": lineage_ok,
                        "r1_coef": {k: float(v) for k, v in r1_4090.items()},
                        "abs_coef": {k: float(v) for k, v in abs_4090.items()}}

    # ------------------------------------------------------------------
    out()
    out("== T1. On-platform refit (pre-registered: held-out MAPE <= 25%, R1) ==")
    train_r, test_r = stratified_split(shared, seed=SEED)
    strata_n = defaultdict(lambda: [0, 0])
    for r in shared:
        strata_n[stratum(r)][0] += 1
    for r in test_r:
        strata_n[stratum(r)][1] += 1
    out(f"  stratified 80/20, seed {SEED}, strata (arch, phase-class, precision), "
        f"same machinery as the 4090 fit: train {len(train_r)} / test {len(test_r)}")
    for k in sorted(strata_n):
        out(f"    {k[0]:16s} {k[1]:12s} {k[2]:5s} n={strata_n[k][0]:3d}  test={strata_n[k][1]}")
    f_tr, y_tr, f_te, y_te = FE(train_r), Y(train_r), FE(test_r), Y(test_r)
    m_t1 = F.fit_relative(FORM, f_tr, y_tr)
    yhat_te = m_t1.predict(f_te)
    t1_mape = float(mape(np.array(y_te), yhat_te))
    t1_verdict = verdict(t1_mape, T1_BAND)
    m_t1_abs = F.fit_absolute(FORM, f_tr, y_tr)
    t1_mape_abs = float(mape(np.array(y_te), m_t1_abs.predict(f_te)))
    out(f"  R1 held-out MAPE {t1_mape:.2f}%  ->  T1 verdict: {t1_verdict} (band 25%)")
    out(f"     coefficients (R1, train): {F.coef_line(m_t1)}")
    out(f"  absolute companion held-out MAPE {t1_mape_abs:.2f}% (secondary)")
    out(f"     coefficients (absolute, train): {F.coef_line(m_t1_abs)}")
    out(f"  R1 in-sample MAPE on train: {float(mape(np.array(y_tr), m_t1.predict(f_tr))):.2f}%")
    bd_t1_prec = breakdown(test_r, y_te, yhat_te, label_prec)
    bd_t1_pc = breakdown(test_r, y_te, yhat_te, label_pc)
    bd_t1_ppc = breakdown(test_r, y_te, yhat_te, label_prec_pc)
    out("  section 16.4 breakdown of the held-out R1 residuals (descriptive):")
    print_breakdown("by precision", bd_t1_prec)
    print_breakdown("by phase class", bd_t1_pc)
    print_breakdown("by precision x phase class", bd_t1_ppc)
    J["t1"] = {"n_train": len(train_r), "n_test": len(test_r),
               "strata": {"/".join(k): {"n": v[0], "n_test": v[1]} for k, v in strata_n.items()},
               "r1_mape_test": t1_mape, "band": T1_BAND, "verdict": t1_verdict,
               "r1_coef_train": coef_dict(m_t1),
               "abs_mape_test": t1_mape_abs, "abs_coef_train": coef_dict(m_t1_abs),
               "r1_mape_train": float(mape(np.array(y_tr), m_t1.predict(f_tr))),
               "breakdown_test": {"precision": bd_t1_prec, "phase_class": bd_t1_pc,
                                  "precision_x_phase_class": bd_t1_ppc},
               "test_keys": [r["spec"]["key"] for r in test_r]}
    J["verdicts"]["T1"] = t1_verdict

    # ------------------------------------------------------------------
    out()
    out("== T2. Cross-platform transfer, scale-calibrated (pre-registered: MAPE <= 30% on the remaining 76, R1) ==")
    cal_set = set(T2_CALIBRATION_KEYS)
    cal_r = [r for r in shared if seedless_key(r["spec"]["key"]) in cal_set]
    eval_r = [r for r in shared if seedless_key(r["spec"]["key"]) not in cal_set]
    assert (len(cal_r), len(eval_r)) == (8, 76), (len(cal_r), len(eval_r))
    yhat0_cal = m4090_r1.predict(FE(cal_r))
    yhat0_eval = m4090_r1.predict(FE(eval_r))
    s_star = calib_scalar_relative(yhat0_cal, Y(cal_r))
    t2_mape = float(mape(np.array(Y(eval_r)), s_star * yhat0_eval))
    t2_verdict = verdict(t2_mape, T2_BAND)
    zs_mape_76 = float(mape(np.array(Y(eval_r)), yhat0_eval))
    zs_mape_84 = float(mape(np.array(Y(shared)), m4090_r1.predict(FE(shared))))
    out(f"  frozen 4090 R1 vector x A100 features (hbm2e column); calibration = the 8 "
        f"fp16 cells of amendment section 16.2; evaluation = the remaining 76")
    out("  calibration cells (y measured J, yhat zero-shot J, y/yhat):")
    for r, yh in zip(cal_r, yhat0_cal):
        y = y_of[r["spec"]["key"]]
        out(f"    {seedless_key(r['spec']['key']):78s} y={y:9.4g}  yhat={float(yh):9.4g}  "
            f"y/yhat={y / float(yh):6.3f}")
    out(f"  s* = {s_star:.4f} (relative loss, closed form)")
    out(f"  scaled MAPE on the remaining 76: {t2_mape:.2f}%  ->  T2 verdict: {t2_verdict} (band 30%)")
    out(f"  zero-shot (s = 1) MAPE: {zs_mape_76:.2f}% on the 76, {zs_mape_84:.2f}% on all 84 (descriptive)")
    yhatA_cal = m4090_abs.predict(FE(cal_r))
    yhatA_eval = m4090_abs.predict(FE(eval_r))
    s_abs = calib_scalar_absolute(yhatA_cal, Y(cal_r))
    t2_mape_abs = float(mape(np.array(Y(eval_r)), s_abs * yhatA_eval))
    out(f"  absolute companion (frozen 4090 absolute vector, absolute-loss scalar "
        f"s = {s_abs:.4f}): MAPE {t2_mape_abs:.2f}% on the 76 (secondary)")
    y_eval = Y(eval_r)
    yh_eval = s_star * yhat0_eval
    bd_t2_prec = breakdown(eval_r, y_eval, yh_eval, label_prec)
    bd_t2_pc = breakdown(eval_r, y_eval, yh_eval, label_pc)
    bd_t2_ppc = breakdown(eval_r, y_eval, yh_eval, label_prec_pc)
    bd_t2_arch = breakdown(eval_r, y_eval, yh_eval, label_arch)
    out("  section 16.4 breakdown of the scaled T2 residuals on the 76 (descriptive):")
    print_breakdown("by precision", bd_t2_prec)
    print_breakdown("by phase class", bd_t2_pc)
    print_breakdown("by precision x phase class", bd_t2_ppc)
    print_breakdown("by arch", bd_t2_arch)
    out("  section 16.3 sensitivity companion (descriptive): cell 7 replaced in turn")
    sens = {}
    fixed7 = cal_set - {T2_ENC_DEC_DECODE_KEY}
    for alt in T2_SENSITIVITY_ALTERNATES:
        cs = fixed7 | {alt}
        c_r = [r for r in shared if seedless_key(r["spec"]["key"]) in cs]
        e_r = [r for r in shared if seedless_key(r["spec"]["key"]) not in cs]
        assert (len(c_r), len(e_r)) == (8, 76)
        s_alt = calib_scalar_relative(m4090_r1.predict(FE(c_r)), Y(c_r))
        m_alt = float(mape(np.array(Y(e_r)), s_alt * m4090_r1.predict(FE(e_r))))
        sens[alt] = {"s_star": s_alt, "mape_76": m_alt, "would_be": verdict(m_alt, T2_BAND)}
        out(f"    {alt:78s} s*={s_alt:.4f}  MAPE {m_alt:6.2f}%  [{verdict(m_alt, T2_BAND)}]")
    J["t2"] = {"calibration_keys": list(T2_CALIBRATION_KEYS), "n_calibration": 8, "n_eval": 76,
               "s_star": s_star, "mape_76": t2_mape, "band": T2_BAND, "verdict": t2_verdict,
               "zero_shot_mape_76": zs_mape_76, "zero_shot_mape_84": zs_mape_84,
               "abs_companion": {"s": s_abs, "mape_76": t2_mape_abs},
               "calibration_table": [{"key": r["spec"]["key"], "y_j": y_of[r["spec"]["key"]],
                                      "yhat_zero_shot_j": float(yh)}
                                     for r, yh in zip(cal_r, yhat0_cal)],
               "breakdown_76": {"precision": bd_t2_prec, "phase_class": bd_t2_pc,
                                "precision_x_phase_class": bd_t2_ppc, "arch": bd_t2_arch},
               "sensitivity": sens}
    J["verdicts"]["T2"] = t2_verdict

    # ------------------------------------------------------------------
    out()
    out("== T3. Extrapolation to 7B-class (pre-registered: pooled MAPE <= 30% on the 10 extension points, R1) ==")
    f_sh, y_sh = FE(shared), Y(shared)
    m_t3 = F.fit_relative(FORM, f_sh, y_sh)
    yhat_ext = m_t3.predict(FE(ext))
    t3_mape = float(mape(np.array(Y(ext)), yhat_ext))
    t3_verdict = verdict(t3_mape, T3_BAND)
    m_t3_abs = F.fit_absolute(FORM, f_sh, y_sh)
    t3_mape_abs = float(mape(np.array(Y(ext)), m_t3_abs.predict(FE(ext))))
    out(f"  R1 fit on all 84 shared points; predict the 10 extension points")
    out(f"     coefficients (R1, all 84): {F.coef_line(m_t3)}")
    out(f"  pooled MAPE {t3_mape:.2f}%  ->  T3 verdict: {t3_verdict} (band 30%)")
    out(f"  absolute companion pooled MAPE {t3_mape_abs:.2f}% (secondary)")
    out(f"     coefficients (absolute, all 84): {F.coef_line(m_t3_abs)}")
    bd_t3_reg = breakdown(ext, Y(ext), yhat_ext, lambda r: regime_of_extension(r["spec"]))
    bd_t3_model = breakdown(ext, Y(ext), yhat_ext, lambda r: r["spec"]["model"])
    print_breakdown("per regime (descriptive)", bd_t3_reg)
    print_breakdown("per model (descriptive)", bd_t3_model)
    out("  per point (y J, yhat J, signed %):")
    for r, yh in zip(ext, yhat_ext):
        y = y_of[r["spec"]["key"]]
        out(f"    {seedless_key(r['spec']['key']):60s} y={y:9.4g}  yhat={float(yh):9.4g}  "
            f"{100 * (float(yh) - y) / y:+7.1f}%")
    yhat_sh = m_t3.predict(f_sh)
    bd_84_prec = breakdown(shared, y_sh, yhat_sh, label_prec)
    bd_84_pc = breakdown(shared, y_sh, yhat_sh, label_pc)
    bd_84_ppc = breakdown(shared, y_sh, yhat_sh, label_prec_pc)
    out(f"  in-sample residuals of the all-84 R1 fit (descriptive; MAPE "
        f"{float(mape(np.array(y_sh), yhat_sh)):.2f}%, R2 {r2_score(np.array(y_sh), yhat_sh):.4f}):")
    print_breakdown("by precision", bd_84_prec)
    print_breakdown("by phase class", bd_84_pc)
    print_breakdown("by precision x phase class", bd_84_ppc)
    J["t3"] = {"n_train": 84, "n_predict": 10, "r1_mape_pooled": t3_mape, "band": T3_BAND,
               "verdict": t3_verdict, "r1_coef_all84": coef_dict(m_t3),
               "abs_mape_pooled": t3_mape_abs, "abs_coef_all84": coef_dict(m_t3_abs),
               "per_regime": bd_t3_reg, "per_model": bd_t3_model,
               "per_point": [{"key": r["spec"]["key"], "y_j": y_of[r["spec"]["key"]],
                              "yhat_j": float(yh)} for r, yh in zip(ext, yhat_ext)],
               "in_sample_84": {"mape": float(mape(np.array(y_sh), yhat_sh)),
                                "r2": float(r2_score(np.array(y_sh), yhat_sh)),
                                "precision": bd_84_prec, "phase_class": bd_84_pc,
                                "precision_x_phase_class": bd_84_ppc}}
    J["verdicts"]["T3"] = t3_verdict

    # ------------------------------------------------------------------
    out()
    out("== Precision ratio check (descriptive; LOGBOOK 2026-08-17 predictions) ==")
    pairs = defaultdict(dict)
    for r in shared:
        s = r["spec"]
        pk = tuple(sorted((k, v) for k, v in s.items() if k not in ("precision", "key", "seed")))
        pairs[pk][s["precision"]] = r
    ratio_rows = []
    for pk, d in pairs.items():
        if "fp16" in d and "fp32" in d:
            k16, k32 = d["fp16"]["spec"]["key"], d["fp32"]["spec"]["key"]
            meas = y_of[k32] / y_of[k16]
            mod = float(m_t3.predict([feat_of[k32]])[0]) / float(m_t3.predict([feat_of[k16]])[0])
            ratio_rows.append({"key_fp16": k16, "phase_class": phase_class(d["fp16"]["spec"]["phase"]),
                               "measured": meas, "model_r1_all84": mod})
    for pc in ("forward", "decode_like"):
        ms = sorted(x["measured"] for x in ratio_rows if x["phase_class"] == pc)
        mo = sorted(x["model_r1_all84"] for x in ratio_rows if x["phase_class"] == pc)
        if ms:
            out(f"  {pc:12s} pairs {len(ms):2d}: measured fp32/fp16 median {statistics.median(ms):.2f} "
                f"[{ms[0]:.2f}, {ms[-1]:.2f}]; model-implied median {statistics.median(mo):.2f} "
                f"[{mo[0]:.2f}, {mo[-1]:.2f}]")
    out("  (the frozen form's implied ratio is bounded by the prior multipliers, "
        "MAC 0.33 and words 0.5 for fp16: at most 3.03 for any beta >= 0)")
    J["precision_ratio_check"] = ratio_rows

    # ------------------------------------------------------------------
    out()
    out("== Coefficient-transfer table (amendment section 10; D6 report-only) ==")
    out("  alpha_A100 (R1, all 84) / alpha_4090 (R1, all 296); expectation recorded 2026-08-10: "
        "hbm ratio ~1.0 if the registry priors are exact")
    ct = {}
    a100_r1 = coef_dict(m_t3)
    for c in m_t3.column_names:
        a, b = a100_r1[c], float(r1_4090[c])
        ratio = (a / b) if b > 0 else None
        ct[c] = {"a100": a, "4090": b, "ratio": ratio}
        out(f"    {c:12s} A100 {a:.4g}   4090 {b:.4g}   ratio "
            f"{'n/a (4090 coefficient is zero)' if ratio is None else f'{ratio:.3f}'}")
    eff = None
    if float(r1_4090["to_hbm"]) > 0:
        eff = (a100_r1["to_hbm"] * HBM_WORD_A100) / (float(r1_4090["to_hbm"]) * HBM_WORD_4090)
        out(f"  effective per-word off-chip energy ratio (alpha_hbm x prior word cost), "
            f"A100/4090: {eff:.3f}  [{a100_r1['to_hbm'] * HBM_WORD_A100 * 1e12:.4g} pJ/word vs "
            f"{float(r1_4090['to_hbm']) * HBM_WORD_4090 * 1e12:.4g} pJ/word; alpha in J/TO x prior TO/word]")
    out("  note: the 40 GB SXM4 part carries HBM2 per the vendor datasheet; the registry tier is "
        "labeled hbm2e at 6.00 pJ/bit (HBM2 6.25 pJ/bit would rescale the to_hbm column by 200/192, "
        "absorbed by alpha_hbm; expectation 0.96 under that prior)")
    a100_abs = coef_dict(m_t3_abs)
    ct_abs = {c: {"a100": a100_abs[c], "4090": float(abs_4090[c]),
                  "ratio": (a100_abs[c] / float(abs_4090[c])) if float(abs_4090[c]) > 0 else None}
              for c in m_t3_abs.column_names}
    J["coefficient_transfer"] = {"r1": ct, "effective_offchip_ratio_r1": eff,
                                 "prior_word_fj": {"a100": HBM_WORD_A100, "rtx4090": HBM_WORD_4090},
                                 "absolute": ct_abs}

    # ------------------------------------------------------------------
    out()
    out("== Calibrated MCER on the A100 (R1, all 84) vs the 4090 (R1, all 296) ==")
    assert F.separable(m_t3)
    mcer_pts = {}
    by_ap = defaultdict(list)
    for r in shared:
        mem, comp = F.fitted_split_energy(m_t3, feat_of[r["spec"]["key"]])
        v = (mem / comp) if comp > 0 else float("inf")
        mcer_pts[r["spec"]["key"]] = v
        by_ap[(r["spec"]["arch"], r["spec"]["phase"])].append(v)
    mcer_4090 = fr["exploratory"].get("mcer_summary_r1", {})
    mcer_summary = {}
    for (arch, phase), vals in sorted(by_ap.items()):
        med = statistics.median(vals)
        ref = mcer_4090.get(f"{arch}/{phase}", {}).get("median")
        out(f"  {arch:16s} {phase:16s} A100 median MCER {med:9.3f}  [{min(vals):.3f}, {max(vals):.3f}]  "
            f"n={len(vals):2d}   4090 median {ref if ref is None else f'{ref:.3f}'}")
        mcer_summary[f"{arch}/{phase}"] = {"median": _jf(med), "min": _jf(min(vals)),
                                           "max": _jf(max(vals)), "n": len(vals),
                                           "median_4090_r1": ref}
    J["mcer"] = {"summary": mcer_summary}

    # ------------------------------------------------------------------
    out()
    out("== D7. Baseline companion on the T1 split, both estimators (secondary Wilcoxon + Holm) ==")
    p_tr, p_te = P(train_r), P(test_r)
    roof_kw = roofline_constants(DEVICE)
    d7 = {}
    for est in ("R1", "absolute"):
        rel = est == "R1"
        m_win = m_t1 if rel else m_t1_abs
        ape_win = ape_pct(y_te, m_win.predict(f_te))
        m0 = F.fit_relative("M0_flops", f_tr, y_tr) if rel else F.fit_absolute("M0_flops", f_tr, y_tr)
        roof = RooflineBaseline(**roof_kw, device=DEVICE).fit(f_tr, p_tr, y_tr, relative=rel)
        layer = LayerwiseBaseline(device=DEVICE).fit(f_tr, p_tr, y_tr, relative=rel)
        comps = [("M0_flops", ape_pct(y_te, m0.predict(f_te))),
                 ("roofline", ape_pct(y_te, roof.predict(f_te, p_te))),
                 ("layerwise", ape_pct(y_te, layer.predict(f_te, p_te)))]
        pv = [wilcoxon_less(ape_win, a) for _, a in comps]
        pa = holm_adjust(pv)
        out(f"  estimator {est}: {FORM} MAPE {float(np.mean(ape_win)):7.2f}%  (TOML)")
        entry = {"winner_mape": float(np.mean(ape_win)), "comparisons": {},
                 "roofline_p_avg_w": float(roof.p_avg_w_), "tdp_w": TDP_W_BY_DEVICE[DEVICE],
                 "layerwise_coef": layer.coef_dict()}
        for (name, a), p, padj in zip(comps, pv, pa):
            beat = padj <= ALPHA
            out(f"    {name:12s} MAPE {float(np.mean(a)):7.2f}%  p={p:.4g}  Holm p={padj:.4g}  "
                f"[{'BEATEN' if beat else 'NOT beaten'}]")
            entry["comparisons"][name] = {"mape": float(np.mean(a)), "p_one_sided": float(p),
                                          "p_holm": float(padj), "beaten": bool(beat)}
        out(f"    roofline fitted P_avg {roof.p_avg_w_:.1f} W vs the {TDP_W_BY_DEVICE[DEVICE]:.0f} W envelope "
            f"(4090 diagnostic: 378 W vs a 150 W part)")
        d7[est] = entry
    out("  roofline constants: A100 FP32 19.49 / FP16 311.9 TFLOP/s, 1,555 GB/s (fit/baselines.py, "
        "vendor-cited); the roofline encodes the 16x fp32/fp16 datapath gap")
    J["d7_baselines"] = d7

    # ------------------------------------------------------------------
    out()
    out("== Spot cells: Follow-up B decomposition on Ampere (descriptive; excluded from every fit) ==")
    e_arm = {r["spec"]["weights"]: y_of[r["spec"]["key"]] for r in spot}
    for arm in ("random", "random_v", "ported", "pretrained"):
        out(f"    {arm:11s} E = {e_arm[arm]:.4f} J per forward (GPT-2 prefill s512 fp16)")
    gap_hf = 1.0 - e_arm["random"] / e_arm["pretrained"]
    impl = e_arm["pretrained"] / e_arm["ported"]
    val_v = e_arm["ported"] / e_arm["random_v"] - 1.0
    val_r = e_arm["ported"] / e_arm["random"] - 1.0
    out(f"  1 - E_random/E_HF = {gap_hf:+.4f}   (4090 Step-4 primary gap: 0.24-0.33; findings.md 2026-08-10)")
    out(f"  E_HF/E_ported     = {impl:.4f}   (4090: 1.405, the HF-fp16 implementation overhead)")
    out(f"  E_ported/E_random_v - 1 = {val_v:+.4f};  E_ported/E_random - 1 = {val_r:+.4f}   "
        f"(4090 implementation-free value effect ~0.05-0.065)")
    J["spot_cells"] = {"energy_j": e_arm, "gap_hf_1_minus_random_over_hf": gap_hf,
                       "impl_hf_over_ported": impl, "value_ported_over_random_v_minus_1": val_v,
                       "value_ported_over_random_minus_1": val_r,
                       "reference_4090": {"gap_hf": "0.24-0.33", "impl_hf_over_ported": 1.405,
                                          "value_effect": "~0.05-0.065",
                                          "source": "findings.md 2026-08-10 (Follow-up B)"}}

    # ------------------------------------------------------------------
    out()
    out("== Verdict summary ==")
    for t, v in J["verdicts"].items():
        out(f"  {t}: {v}")

    # Artifacts
    test_keys = {r["spec"]["key"] for r in test_r}
    train_keys = {r["spec"]["key"] for r in train_r}
    cal_keys = {r["spec"]["key"] for r in cal_r}
    stratum_of = {}
    for name, rs in (("shared", shared), ("extension", ext), ("spot", spot)):
        for r in rs:
            stratum_of[r["spec"]["key"]] = name
    with (FIT_DIR / "per_point_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            k = r["spec"]["key"]
            f = feat_of[k]
            y = y_of[k]
            st = stratum_of[k]
            row = {"key": k, "stratum": st, "arch": r["spec"]["arch"], "phase": r["spec"]["phase"],
                   "precision": r["spec"]["precision"], "seq_len": r["spec"].get("seq_len"),
                   "tgt_ctx": r["spec"].get("tgt_ctx"), "y_j": float(y),
                   "t1_role": ("test" if k in test_keys else "train" if k in train_keys else "none"),
                   "yhat_t1_train_fit_j": _jf(m_t1.predict([f])[0]) if st == "shared" else None,
                   "t2_role": ("calibration" if k in cal_keys else "eval" if st == "shared" else "none"),
                   "yhat_t2_zero_shot_j": _jf(m4090_r1.predict([f])[0]),
                   "yhat_t2_scaled_j": _jf(s_star * m4090_r1.predict([f])[0]),
                   "yhat_t3_all84_fit_j": _jf(m_t3.predict([f])[0]),
                   "mcer_r1_all84": _jf(mcer_pts.get(k, float("nan")))}
            for src in ("t1_train_fit", "t2_scaled", "t3_all84_fit"):
                v = row[f"yhat_{src}_j"]
                row[f"ape_{src}_pct"] = None if v is None else float(abs(y - v) / y * 100.0)
            fh.write(json.dumps(row) + "\n")
    (FIT_DIR / "fit_results.json").write_text(json.dumps(J, indent=2), encoding="utf-8")
    (FIT_DIR / "fit_report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print()
    print(f"wrote {FIT_DIR / 'fit_report.txt'}")
    print(f"wrote {FIT_DIR / 'fit_results.json'}")
    print(f"wrote {FIT_DIR / 'per_point_predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
