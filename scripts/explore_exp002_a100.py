#!/usr/bin/env python
"""EXPLORATORY analysis for the EXP-002 A100 phase.

Pre-registered as post-verdict exploratory work in a100_amendment.md section
16.5(a) (2026-08-17); run only after the T0-T3 verdicts were recorded
(findings.md 2026-08-17). NOTHING here has verdict authority; every output
carries the EXPLORATORY label. Same protocol as the 4090 R1 exploratory
(findings.md 2026-07-24): the confirmatory artifacts in a100/fit/ are not
touched; this writes a100/fit/exploratory/.

    python scripts/explore_exp002_a100.py

Two analyses:

(a) M8p, a precision-split MAC model: to_mac split into fp16 and fp32
    columns, everything else identical to M8_split_dispatch (to_nonlinear,
    to_sram, to_hbm, n_launches, intercept), fitted under R1 (rows scaled by
    1/y, RHS ones, NNLS; the same math as fit_relative). Repeated on the
    IDENTICAL T1 split, on all 84 for the T3 targets, and for the T2 scalar
    transfer with an M8p refit on all 296 4090 points and the same 8
    calibration cells. The fitted fp16 MAC multiplier (alpha_fp16 x 0.33 /
    alpha_fp32) is reported per platform against the asserted 0.33.
    Quantifies how much of each FAIL the precision prior explains and how
    much remains.

(b) Operating-point table (descriptive): residuals of the frozen M8 all-84
    R1 fit and of M8p, binned by the recorded per-point median SM clock
    (1095 MHz DVFS state / intermediate / 1410 MHz boost) x precision x
    phase class, with net active power, total board power (net + measured
    idle) and max repeat temperature per bin. Documents mechanism 2 of the
    findings entry; no model change.
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
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tomltransformers.energy_model import mape, r2_score                 # noqa: E402
from tomltransformers.fit.a100_calibration import (                      # noqa: E402
    T2_CALIBRATION_KEYS, seedless_key)
from tomltransformers.fit.a100_strata import partition_records          # noqa: E402
from tomltransformers.fit.bridge import features_for_spec, load_latest_records  # noqa: E402
from tomltransformers.fit.splits import phase_class, stratified_split   # noqa: E402
from tomltransformers.sweep.grid import load_config                     # noqa: E402
from tomltransformers import to_costs as tc                              # noqa: E402


def _load_frozen_fit_script():
    p = REPO / "scripts" / "fit_exp002.py"
    spec = importlib.util.spec_from_file_location("fit_exp002_frozen", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load_frozen_fit_script()

RUN_DIR = REPO / "experiments" / "exp_002_size_sweep"
A100_DIR = RUN_DIR / "a100"
DATA_A100 = A100_DIR / "energy.jsonl"
DATA_4090 = RUN_DIR / "energy.jsonl"
CFG_A100 = REPO / "configs" / "exp_002_a100.yaml"
CONFIRMATORY = A100_DIR / "fit" / "fit_results.json"
OUT_DIR = A100_DIR / "fit" / "exploratory"

SEED = 42
FORM = "M8_split_dispatch"
MAC_MULT_FP16 = float(tc.PRECISION_MAC_MULT["fp16"].value)   # the asserted prior, 0.33
MAC_MULT_FP32 = float(tc.PRECISION_MAC_MULT["fp32"].value)   # 1.00

M8P_COLUMNS = ("to_mac_fp16", "to_mac_fp32", "to_nonlinear", "to_sram",
               "to_hbm", "n_launches", "intercept")

LINES: list[str] = []


def out(s: str = "") -> None:
    LINES.append(s)
    print(s)


def _jf(v):
    v = float(v)
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- M8p ---------

def design_m8p(feats, precs) -> np.ndarray:
    rows = []
    for f, p in zip(feats, precs):
        m16 = float(f["to_mac"]) if p == "fp16" else 0.0
        m32 = float(f["to_mac"]) if p == "fp32" else 0.0
        rows.append([m16, m32, float(f["to_nonlinear"]), float(f["to_sram"]),
                     float(f["to_hbm"]), float(f["n_launches"]), 1.0])
    return np.array(rows, float)


def fit_m8p(feats, precs, ys, *, relative: bool = True) -> np.ndarray:
    A = design_m8p(feats, precs)
    y = np.asarray(ys, float)
    if relative:                              # same math as fit_exp002.fit_relative
        w = 1.0 / y
        coef, _ = nnls(A * w[:, None], np.ones_like(y))
    else:
        coef, _ = nnls(A, y)
    return coef


def predict_m8p(coef, feats, precs) -> np.ndarray:
    return design_m8p(feats, precs) @ coef


def coefs_m8p(coef) -> dict:
    return dict(zip(M8P_COLUMNS, map(float, coef)))


def coef_line_m8p(coef) -> str:
    return "  ".join(f"{c}={v:.4g}" for c, v in coefs_m8p(coef).items())


def fitted_fp16_multiplier(coef) -> float | None:
    """J per fp16 MAC over J per fp32 MAC implied by the fitted split
    coefficients, on the same scale as the asserted PRECISION_MAC_MULT (0.33):
    (alpha_fp16 x mult_fp16) / (alpha_fp32 x mult_fp32)."""
    a16, a32 = float(coef[0]), float(coef[1])
    if a32 <= 0:
        return None
    return (a16 * MAC_MULT_FP16) / (a32 * MAC_MULT_FP32)


def calib_scalar_relative(yhat, y) -> float:
    r = np.asarray(yhat, float) / np.asarray(y, float)
    return float(np.sum(r) / np.sum(r * r))


def signed_pct(y, yhat) -> np.ndarray:
    y = np.asarray(y, float)
    return (np.asarray(yhat, float) - y) / y * 100.0


def breakdown(recs, ys, yhat, key_fn) -> dict:
    g = defaultdict(lambda: ([], []))
    for r, y, yh in zip(recs, ys, yhat):
        a, b = g[key_fn(r)]
        a.append(float(y))
        b.append(float(yh))
    res = {}
    for k in sorted(g):
        yt, yp = map(np.array, g[k])
        res[str(k)] = {"n": int(len(yt)), "mape": float(mape(yt, yp)),
                       "mean_signed_pct": float(np.mean(signed_pct(yt, yp)))}
    return res


def print_breakdown(title: str, bd: dict) -> None:
    out(f"    {title}:")
    for k, v in bd.items():
        out(f"      {k:26s} n={v['n']:3d}  MAPE {v['mape']:7.2f}%  mean signed {v['mean_signed_pct']:+8.2f}%")


def label_prec_pc(r):
    return f"{r['spec']['precision']}/{phase_class(r['spec']['phase'])}"


def regime_of_extension(spec) -> str:
    if spec["phase"] == "prefill":
        return "prefill_s8192" if int(spec["seq_len"]) == 8192 else "prefill_s1024_4096"
    return "decode"


def med_clock(rec) -> float:
    cl = [c.get("sm") for c in (rec.get("clocks_mhz") or [])[1:] if isinstance(c, dict)]
    cl = [c for c in cl if c is not None]
    if not cl:
        cl = [c.get("sm") for c in (rec.get("clocks_mhz") or []) if isinstance(c, dict) and c.get("sm") is not None]
    return float(statistics.median(cl)) if cl else float("nan")


def clock_bin(mhz: float) -> str:
    if mhz <= 1100:
        return "1095 (low DVFS state)"
    if mhz >= 1400:
        return "1410 (rated boost)"
    return "1100-1399 (intermediate)"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = F.git_commit()
    out("EXPLORATORY: EXP-002 A100 post-verdict analyses (amendment section 16.5(a)); NO VERDICT AUTHORITY")
    out(f"generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    out(f"git commit: {commit}")
    out(f"confirmatory artifacts (untouched): {CONFIRMATORY}")
    conf = json.loads(CONFIRMATORY.read_text(encoding="utf-8"))
    out(f"confirmatory verdicts on record: {conf['verdicts']}  (T1 {conf['t1']['r1_mape_test']:.2f}%, "
        f"T2 {conf['t2']['mape_76']:.2f}%, T3 {conf['t3']['r1_mape_pooled']:.2f}%)")

    # ------------------------------------------------------------ data ------
    recs_a = load_latest_records(DATA_A100)
    assert len(recs_a) == 98
    parts = partition_records(recs_a, load_config(str(CFG_A100)))
    shared, ext = parts["shared"], parts["extension"]
    assert (len(shared), len(ext)) == (84, 10)
    feat_a = {r["spec"]["key"]: features_for_spec(r["spec"], device="a100") for r in recs_a}
    y_a = {r["spec"]["key"]: F.target_y(r) for r in recs_a}
    F.units_gate(recs_a, [y_a[r["spec"]["key"]] for r in recs_a])
    rec_by_key = {r["spec"]["key"]: r for r in recs_a}

    recs_4 = load_latest_records(DATA_4090)
    feat_4 = {r["spec"]["key"]: features_for_spec(r["spec"], device="rtx4090") for r in recs_4}
    y_4 = {r["spec"]["key"]: F.target_y(r) for r in recs_4}
    F.units_gate(recs_4, [y_4[r["spec"]["key"]] for r in recs_4])
    out(f"data: A100 98 records (shared 84, extension 10); 4090 {len(recs_4)} records; units gates passed")

    def FE(rs, fm):
        return [fm[r["spec"]["key"]] for r in rs]

    def Y(rs, ym):
        return [ym[r["spec"]["key"]] for r in rs]

    def P(rs):
        return [r["spec"]["precision"] for r in rs]

    J: dict = {"label": "EXPLORATORY", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "git_commit": commit, "confirmatory_verdicts": conf["verdicts"],
               "m8p_columns": list(M8P_COLUMNS), "asserted_fp16_mac_multiplier": MAC_MULT_FP16}

    # ------------------------------------------------ (a) M8p on the T1 split
    out()
    out("== (a1) EXPLORATORY: M8p (precision-split MAC) on the identical T1 split, R1 ==")
    train_r, test_r = stratified_split(shared, seed=SEED)
    assert [r["spec"]["key"] for r in test_r] == conf["t1"]["test_keys"], "split differs from the confirmatory run"
    c_tr = fit_m8p(FE(train_r, feat_a), P(train_r), Y(train_r, y_a), relative=True)
    yhat_te = predict_m8p(c_tr, FE(test_r, feat_a), P(test_r))
    m8p_t1 = float(mape(np.array(Y(test_r, y_a)), yhat_te))
    mult_tr = fitted_fp16_multiplier(c_tr)
    out(f"  M8 (confirmatory, on record) held-out MAPE {conf['t1']['r1_mape_test']:.2f}%  ->  "
        f"M8p held-out MAPE {m8p_t1:.2f}%  (band would be 25%; NO verdict)")
    out(f"    coefficients (R1, train): {coef_line_m8p(c_tr)}")
    out(f"    fitted fp16 MAC multiplier {('n/a (alpha_fp32 = 0)' if mult_tr is None else f'{mult_tr:.3f}')} "
        f"vs asserted {MAC_MULT_FP16:.2f}")
    bd = breakdown(test_r, Y(test_r, y_a), yhat_te, label_prec_pc)
    print_breakdown("held-out by precision x phase class", bd)
    m8_conf_bd = conf["t1"]["breakdown_test"]["precision_x_phase_class"]
    out("    (M8 confirmatory, same cells: " + "; ".join(
        f"{k} {v['mean_signed_pct']:+.1f}%" for k, v in m8_conf_bd.items()) + ")")
    J["a1_t1_repeat"] = {"m8_mape_on_record": conf["t1"]["r1_mape_test"], "m8p_mape": m8p_t1,
                         "m8p_coef_train": coefs_m8p(c_tr), "fitted_fp16_multiplier": mult_tr,
                         "breakdown_test": bd}

    # ------------------------------------------------ (a2) M8p all-84 -> 7B
    out()
    out("== (a2) EXPLORATORY: M8p fit on all 84, predict the 10 extension points ==")
    c_84 = fit_m8p(FE(shared, feat_a), P(shared), Y(shared, y_a), relative=True)
    yhat_ext = predict_m8p(c_84, FE(ext, feat_a), P(ext))
    m8p_t3 = float(mape(np.array(Y(ext, y_a)), yhat_ext))
    mult_84 = fitted_fp16_multiplier(c_84)
    yhat_84 = predict_m8p(c_84, FE(shared, feat_a), P(shared))
    out(f"  M8 (confirmatory) pooled MAPE {conf['t3']['r1_mape_pooled']:.2f}%  ->  M8p pooled MAPE {m8p_t3:.2f}%  "
        f"(band would be 30%; NO verdict)")
    out(f"    coefficients (R1, all 84): {coef_line_m8p(c_84)}")
    out(f"    fitted fp16 MAC multiplier {('n/a' if mult_84 is None else f'{mult_84:.3f}')} vs asserted {MAC_MULT_FP16:.2f}; "
        f"in-sample MAPE {float(mape(np.array(Y(shared, y_a)), yhat_84)):.2f}%, R2 {r2_score(np.array(Y(shared, y_a)), yhat_84):.4f}")
    bd_reg = breakdown(ext, Y(ext, y_a), yhat_ext, lambda r: regime_of_extension(r["spec"]))
    print_breakdown("per regime", bd_reg)
    bd84 = breakdown(shared, Y(shared, y_a), yhat_84, label_prec_pc)
    print_breakdown("in-sample by precision x phase class", bd84)
    J["a2_t3_repeat"] = {"m8_mape_on_record": conf["t3"]["r1_mape_pooled"], "m8p_mape": m8p_t3,
                         "m8p_coef_all84": coefs_m8p(c_84), "fitted_fp16_multiplier": mult_84,
                         "per_regime": bd_reg, "in_sample_84": bd84,
                         "per_point": [{"key": r["spec"]["key"], "y_j": y_a[r["spec"]["key"]], "yhat_j": float(yh)}
                                       for r, yh in zip(ext, yhat_ext)]}

    # ------------------------------------------------ (a3) M8p T2 repeat
    out()
    out("== (a3) EXPLORATORY: T2 repeat with M8p refit on all 296 4090 points, one scalar on the same 8 cells ==")
    c_4090 = fit_m8p(FE(recs_4, feat_4), P(recs_4), Y(recs_4, y_4), relative=True)
    yhat_4090_in = predict_m8p(c_4090, FE(recs_4, feat_4), P(recs_4))
    mult_4090 = fitted_fp16_multiplier(c_4090)
    out(f"  4090 M8p (R1, all {len(recs_4)}): {coef_line_m8p(c_4090)}")
    fr4090 = json.loads((RUN_DIR / "fit" / "fit_results.json").read_text(encoding="utf-8"))
    m8_4090_full = float(fr4090["exploratory"]["r1_full_mape"])
    out(f"    4090 in-sample MAPE {float(mape(np.array(Y(recs_4, y_4)), yhat_4090_in)):.2f}% "
        f"(frozen M8 R1 all-296 on record: {m8_4090_full:.2f}%); fitted fp16 MAC multiplier "
        f"{('n/a' if mult_4090 is None else f'{mult_4090:.3f}')} vs asserted {MAC_MULT_FP16:.2f}")
    cal_set = set(T2_CALIBRATION_KEYS)
    cal_r = [r for r in shared if seedless_key(r["spec"]["key"]) in cal_set]
    eval_r = [r for r in shared if seedless_key(r["spec"]["key"]) not in cal_set]
    assert (len(cal_r), len(eval_r)) == (8, 76)
    yh0_cal = predict_m8p(c_4090, FE(cal_r, feat_a), P(cal_r))
    yh0_eval = predict_m8p(c_4090, FE(eval_r, feat_a), P(eval_r))
    s_p = calib_scalar_relative(yh0_cal, Y(cal_r, y_a))
    m8p_t2 = float(mape(np.array(Y(eval_r, y_a)), s_p * yh0_eval))
    zs_p = float(mape(np.array(Y(eval_r, y_a)), yh0_eval))
    out(f"  M8 (confirmatory) scaled MAPE {conf['t2']['mape_76']:.2f}% (s* {conf['t2']['s_star']:.4f})  ->  "
        f"M8p scaled MAPE {m8p_t2:.2f}% (s* {s_p:.4f}); zero-shot {zs_p:.2f}%  (band would be 30%; NO verdict)")
    bd_t2 = breakdown(eval_r, Y(eval_r, y_a), s_p * yh0_eval, label_prec_pc)
    print_breakdown("scaled residuals on the 76 by precision x phase class", bd_t2)
    out(f"  cross-platform fitted fp16 MAC multiplier: 4090 {('n/a' if mult_4090 is None else f'{mult_4090:.3f}')}, "
        f"A100 {('n/a' if mult_84 is None else f'{mult_84:.3f}')} (asserted 0.33 on both): the precision prior is a "
        f"device property, not a circuit constant")
    J["a3_t2_repeat"] = {"m8_mape_on_record": conf["t2"]["mape_76"], "m8p_mape": m8p_t2, "s_star": s_p,
                         "zero_shot_mape": zs_p, "m8p_coef_4090_all296": coefs_m8p(c_4090),
                         "fitted_fp16_multiplier_4090": mult_4090, "fitted_fp16_multiplier_a100": mult_84,
                         "breakdown_76": bd_t2}

    # ------------------------------------------------ (b) operating-point table
    out()
    out("== (b) DESCRIPTIVE: residuals by recorded operating point (median SM clock bin x precision x phase class) ==")
    out("  residual = signed % of the R1 all-84 fit; M8 = confirmatory frozen form (from per_point_predictions), M8p = above;")
    out("  P_net = window energy / window wall (W above the per-point measured idle); P_total = P_net + idle_power_w")
    ppp = {}
    with (A100_DIR / "fit" / "per_point_predictions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ppp[row["key"]] = row
    m8p_pred = {r["spec"]["key"]: float(v) for r, v in zip(shared, yhat_84)}
    m8p_pred.update({r["spec"]["key"]: float(v) for r, v in zip(ext, yhat_ext)})
    table = defaultdict(list)
    for r in shared + ext:
        k = r["spec"]["key"]
        mc = med_clock(r)
        e_m8 = (ppp[k]["yhat_t3_all84_fit_j"] - y_a[k]) / y_a[k] * 100.0
        e_m8p = (m8p_pred[k] - y_a[k]) / y_a[k] * 100.0
        p_net = float(r["per_execution_median_j"]["B"]) / float(r["wall_time_s_median"])
        p_tot = p_net + float(r.get("idle_power_w") or 0.0)
        strat = "extension" if r in ext else "shared"
        table[(strat, clock_bin(mc), r["spec"]["precision"], phase_class(r["spec"]["phase"]),
               r["spec"]["attn_kind"])].append((mc, p_net, p_tot, max(r.get("temps_c") or [float("nan")]), e_m8, e_m8p))
    rows_out = []
    for key in sorted(table):
        v = table[key]
        mcs, pn, pt, tmax, em8, em8p = zip(*v)
        rec = {"stratum": key[0], "clock_bin": key[1], "precision": key[2], "phase_class": key[3],
               "attn_kind": key[4], "n": len(v), "median_clock_mhz": statistics.median(mcs),
               "median_p_net_w": statistics.median(pn), "median_p_total_w": statistics.median(pt),
               "median_max_temp_c": statistics.median(tmax),
               "m8_mean_signed_pct": statistics.mean(em8), "m8_min": min(em8), "m8_max": max(em8),
               "m8p_mean_signed_pct": statistics.mean(em8p), "m8p_min": min(em8p), "m8p_max": max(em8p)}
        rows_out.append(rec)
        out(f"  {key[0]:9s} {key[1]:26s} {key[2]:5s} {key[3]:11s} {key[4]:5s} n={len(v):2d}  clk {rec['median_clock_mhz']:5.0f}  "
            f"P_net {rec['median_p_net_w']:5.0f} W  P_total {rec['median_p_total_w']:5.0f} W  Tmax {rec['median_max_temp_c']:3.0f} C  "
            f"M8 {rec['m8_mean_signed_pct']:+6.1f}% [{rec['m8_min']:+.0f},{rec['m8_max']:+.0f}]  "
            f"M8p {rec['m8p_mean_signed_pct']:+6.1f}% [{rec['m8p_min']:+.0f},{rec['m8p_max']:+.0f}]")
    out("  reading: after the precision split, what remains is the operating-point structure (low DVFS state and "
        "power-capped cells over-predicted, 1410 MHz mid-power cells fit); the frozen form has no term for it")
    J["b_operating_point_table"] = rows_out

    # ------------------------------------------------ artifacts
    (OUT_DIR / "explore_results.json").write_text(json.dumps(J, indent=2), encoding="utf-8")
    (OUT_DIR / "explore_report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print()
    print(f"wrote {OUT_DIR / 'explore_report.txt'}")
    print(f"wrote {OUT_DIR / 'explore_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
