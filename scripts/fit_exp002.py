#!/usr/bin/env python
"""EXP-002 fit: plan sections 5-7 (main table, robustness, extrapolation,
derived quantities). Baselines and significance tests (section 8) are the
next step and are NOT run here.

Run from the repo root, after the bridge test gates are green and the code
is committed (artifact provenance records the commit):

    python scripts/fit_exp002.py

Reads the frozen dataset, fits the M0-M9 family per the approved fit plan
(experiments/exp_002_size_sweep/fit_plan.md), and writes:

    experiments/exp_002_size_sweep/fit/fit_report.txt
    experiments/exp_002_size_sweep/fit/fit_results.json
    experiments/exp_002_size_sweep/fit/per_point_predictions.jsonl

Everything printed is also in fit_report.txt. This script is deterministic
(seed 42 stratified split; NNLS is deterministic).
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tomltransformers.energy_model import (EnergyModel, best_by,
                                           fit_and_select, mape, model_family,
                                           r2_score, summary_table)
from tomltransformers.fit.bridge import features_for_spec, load_latest_records
from tomltransformers.fit.splits import (extrapolation_split, phase_class,
                                         stratified_split)

RUN_DIR = REPO / "experiments" / "exp_002_size_sweep"
DATA = RUN_DIR / "energy.jsonl"
FIT_DIR = RUN_DIR / "fit"

SEED = 42
CV_B_EXCLUDE = 0.075          # R2: the five CV(B)-flagged points
INNER_ITERS_EXCLUDE = 2       # R3: the three inner_iters <= 2 points
AB_TARGET = 0.05              # pre-registered pooled A-B median target
EXTRAP_TARGET_MAPE = 25.0     # pre-registered extrapolation band (percent)

MEMORY_FEATS = frozenset({"to_sram", "to_hbm"})
COMPUTE_FEATS = frozenset({"to_mac", "to_nonlinear"})

LINES: list[str] = []


def out(s: str = "") -> None:
    LINES.append(s)
    print(s)


def _jf(v: float):
    """JSON-safe float: non-finite values become None."""
    v = float(v)
    return v if math.isfinite(v) else None


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "<unknown>"
    except Exception:
        return "<unknown>"


def target_y(rec: dict) -> float:
    return float(rec["per_execution_median_j"]["B"])


def y_std(rec: dict) -> float:
    return float(rec.get("per_execution_std_j", {}).get("B", 0.0))


def fresh(name: str) -> EnergyModel:
    return model_family()[name]


def fit_absolute(name: str, feats, ys) -> EnergyModel:
    return fresh(name).fit(feats, ys)


def fit_relative(name: str, feats, ys) -> EnergyModel:
    """R1: rows of A and y scaled by 1/y_i (relative-error NNLS)."""
    m = fresh(name)
    A = m.design_matrix(feats)
    yv = np.asarray(ys, float)
    w = 1.0 / yv
    coef, _ = nnls(A * w[:, None], np.ones_like(yv))
    m.coef_ = coef
    return m


def coef_line(m: EnergyModel) -> str:
    return "  ".join(f"{n}={c:.4g}" for n, c in zip(m.column_names, m.coef_))


def col_kind(colspec: tuple[str, ...]) -> str:
    s = set(colspec)
    if s <= MEMORY_FEATS:
        return "memory"
    if s <= COMPUTE_FEATS:
        return "compute"
    return "other"


def separable(m: EnergyModel) -> bool:
    kinds = {col_kind(c) for c in m.columns}
    return "memory" in kinds and "compute" in kinds


def fitted_split_energy(m: EnergyModel, feat: dict) -> tuple[float, float]:
    """(fitted memory energy, fitted compute energy) for one point."""
    mem = comp = 0.0
    for spec, c in zip(m.columns, m.coef_[: len(m.columns)]):
        val = c * sum(feat.get(f, 0.0) for f in spec)
        kind = col_kind(spec)
        if kind == "memory":
            mem += val
        elif kind == "compute":
            comp += val
    return mem, comp


def group_mape(recs, feats, ys, model, key_fn):
    groups = defaultdict(lambda: ([], []))
    for r, f, y in zip(recs, feats, ys):
        yt, yp = groups[key_fn(r)]
        yt.append(y)
        yp.append(float(model.predict([f])[0]))
    return {k: mape(np.array(yt), np.array(yp))
            for k, (yt, yp) in sorted(groups.items())}


def main() -> int:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    out("EXP-002 fit report (plan sections 5-7; baselines are the next step)")
    out(f"generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    out(f"git commit: {commit}")
    out(f"data: {DATA}")

    records = load_latest_records(DATA)
    assert len(records) == 296, len(records)
    feats = [features_for_spec(r["spec"]) for r in records]
    ys = [target_y(r) for r in records]
    out(f"records: {len(records)} (all ok; target = per_execution_median_j['B'])")

    # ------------------------------------------------------------------
    out()
    out("== Pooled A-B agreement (pre-registered target: median <= 5%) ==")
    ab_all = [float(r["agreement"]["A-B"]) for r in records]
    ab_fwd = [float(r["agreement"]["A-B"]) for r in records
              if phase_class(r["spec"]["phase"]) == "forward"]
    ab_dec = [float(r["agreement"]["A-B"]) for r in records
              if phase_class(r["spec"]["phase"]) == "decode_like"]
    pooled = statistics.median(ab_all)
    out(f"  pooled median A-B over 296 points: {100 * pooled:.2f}%  "
        f"(forward {100 * statistics.median(ab_fwd):.2f}%, "
        f"decode-like {100 * statistics.median(ab_dec):.2f}%)")
    ab_pass = pooled <= AB_TARGET
    out(f"  pre-registered <= 5% pooled median: {'MET' if ab_pass else 'NOT MET'}")

    # ------------------------------------------------------------------
    out()
    out("== Main table: stratified 80/20 (seed 42), M0-M9, AIC selection ==")
    train_r, test_r = stratified_split(records, seed=SEED)
    idx = {r["spec"]["key"]: i for i, r in enumerate(records)}
    tr_i = [idx[r["spec"]["key"]] for r in train_r]
    te_i = [idx[r["spec"]["key"]] for r in test_r]
    f_tr = [feats[i] for i in tr_i]
    y_tr = [ys[i] for i in tr_i]
    f_te = [feats[i] for i in te_i]
    y_te = [ys[i] for i in te_i]
    out(f"  train {len(train_r)} / test {len(test_r)}")

    results = fit_and_select(f_tr, y_tr, f_te, y_te)
    out(summary_table(results))
    winner = best_by(results, "aic")
    out(f"  winner by AIC: {winner.name}  "
        f"(R2_test {winner.r2_test:.4f}, MAPE_test {winner.mape_test:.2f}%)")

    # Final coefficients: winner form refit on all 296 (conventional
    # select-on-split, refit-on-full; used for derived quantities below).
    final = fit_absolute(winner.name, feats, ys)
    out(f"  final coefficients (winner refit on all 296): {coef_line(final)}")
    yhat_full = final.predict(feats)
    out(f"  full-data fit quality: R2 {r2_score(np.array(ys), yhat_full):.4f}, "
        f"MAPE {mape(np.array(ys), yhat_full):.2f}%")

    # ------------------------------------------------------------------
    out()
    out("== Robustness (pre-specified; never used for selection) ==")
    robustness = {}

    m_r1 = fit_relative(winner.name, f_tr, y_tr)
    r1_mape = mape(np.array(y_te), m_r1.predict(f_te))
    out(f"  R1 relative-error NNLS:   test MAPE {r1_mape:.2f}%  "
        f"(primary {winner.mape_test:.2f}%)")
    out(f"     coefficients: {coef_line(m_r1)}")
    robustness["R1_relative"] = {"mape_test": float(r1_mape),
                                 "coef": dict(zip(m_r1.column_names,
                                                  map(float, m_r1.coef_)))}

    def refit_excluding(label, keep):
        ktr = [i for i in tr_i if keep(records[i])]
        kte = [i for i in te_i if keep(records[i])]
        m = fit_absolute(winner.name, [feats[i] for i in ktr],
                         [ys[i] for i in ktr])
        mp = mape(np.array([ys[i] for i in kte]),
                  m.predict([feats[i] for i in kte]))
        n_ex = len(records) - sum(1 for r in records if keep(r))
        out(f"  {label}: excluded {n_ex} points; test MAPE {mp:.2f}%")
        out(f"     coefficients: {coef_line(m)}")
        robustness[label] = {"excluded": n_ex, "mape_test": float(mp),
                             "coef": dict(zip(m.column_names,
                                              map(float, m.coef_)))}

    refit_excluding("R2_cv_flagged",
                    lambda r: float(r["per_execution_cv"]["B"]) <= CV_B_EXCLUDE)
    refit_excluding("R3_low_inner_iters",
                    lambda r: int(r["inner_iters"]) > INNER_ITERS_EXCLUDE)

    # ------------------------------------------------------------------
    out()
    out("== Pre-registered extrapolation (target: held-out MAPE <= 25%) ==")
    extrap = {}
    for reading, label in (("E2", "E2 broad (PRIMARY)"),
                           ("E1", "E1 strict-literal")):
        etr, epr = extrapolation_split(records, reading)
        fe_tr = [feats[idx[r["spec"]["key"]]] for r in etr]
        ye_tr = [ys[idx[r["spec"]["key"]]] for r in etr]
        fe_pr = [feats[idx[r["spec"]["key"]]] for r in epr]
        ye_pr = [ys[idx[r["spec"]["key"]]] for r in epr]
        entry = {"n_train": len(etr), "n_predict": len(epr), "models": {}}
        out(f"  {label}: train {len(etr)}, predict {len(epr)}")
        for name in (winner.name, "M0_flops"):
            m = fit_absolute(name, fe_tr, ye_tr)
            yp = m.predict(fe_pr)
            pooled_m = mape(np.array(ye_pr), yp)
            per_class = group_mape(epr, fe_pr, ye_pr, m,
                                   lambda r: r["spec"]["arch"])
            cls = "  ".join(f"{k}={v:.1f}%" for k, v in per_class.items())
            verdict = "PASS" if pooled_m <= EXTRAP_TARGET_MAPE else "FAIL"
            out(f"    {name:18s} pooled MAPE {pooled_m:6.2f}%  "
                f"[{verdict} vs 25%]  {cls}")
            entry["models"][name] = {
                "pooled_mape": float(pooled_m),
                "per_class_mape": {k: float(v) for k, v in per_class.items()},
                "pass": bool(pooled_m <= EXTRAP_TARGET_MAPE)}
        entry["predict_keys"] = [r["spec"]["key"] for r in epr]
        extrap[reading] = entry

    # ------------------------------------------------------------------
    out()
    out("== Calibrated MCER by phase (fitted memory / fitted compute) ==")
    if separable(final):
        mcer_model = final
        mcer_note = f"winner {winner.name} (refit on all 296)"
    else:
        sep = next((r for r in results if separable(fresh(r.name))), None)
        assert sep is not None, "no separable model in the family"
        mcer_model = fit_absolute(sep.name, feats, ys)
        mcer_note = (f"winner {winner.name} does not separate compute/memory; "
                     f"using best separable model {sep.name} (refit on all 296)")
    out(f"  source: {mcer_note}")
    mcer_by_point = {}
    for r, f in zip(records, feats):
        mem, comp = fitted_split_energy(mcer_model, f)
        mcer_by_point[r["spec"]["key"]] = (mem / comp) if comp > 0 else float("inf")
    mcer_summary = {}
    by_phase = defaultdict(list)
    for r in records:
        by_phase[(r["spec"]["arch"], r["spec"]["phase"])].append(
            mcer_by_point[r["spec"]["key"]])
    for (arch, phase), vals in sorted(by_phase.items()):
        med = statistics.median(vals)
        out(f"  {arch:16s} {phase:16s} median MCER {med:10.3f}  "
            f"[{min(vals):.3f}, {max(vals):.3f}]  n={len(vals)}")
        mcer_summary[f"{arch}/{phase}"] = {
            "median": _jf(med), "min": _jf(min(vals)),
            "max": _jf(max(vals)), "n": len(vals)}

    # ------------------------------------------------------------------
    out()
    out("== Clean decode per-token (prefill-subtracted; plan section 7) ==")
    prefill_lut = {}
    dp_lut = {}
    for r in records:
        s = r["spec"]
        if s["arch"] == "decoder_only" and s["phase"] == "prefill" \
                and s["attn_kind"] == "flash":
            prefill_lut[(s["model"], s["precision"], s["seq_len"])] = r
        if s["arch"] == "encoder_decoder" and s["phase"] == "decoder_prefill":
            dp_lut[(s["model"], s["precision"], s["seq_len"], s["tgt_len"])] = r

    per_token_rows = []
    n_measured = n_model = 0
    for r in records:
        s = r["spec"]
        if s["phase"] != "decode":
            continue
        k = int(s["decode_tokens"])
        e_d, sd_d = target_y(r), y_std(r)
        if s["arch"] == "decoder_only":
            comp_spec = dict(s, phase="prefill")
            match = prefill_lut.get((s["model"], s["precision"], s["seq_len"]))
        else:
            comp_spec = dict(s, phase="decoder_prefill", tgt_len=s["tgt_ctx"])
            match = dp_lut.get((s["model"], s["precision"], s["seq_len"],
                                s["tgt_ctx"]))
        if match is not None:
            e_p, sd_p = target_y(match), y_std(match)
            source = "measured"
            std = (sd_d ** 2 + sd_p ** 2) ** 0.5 / k
            n_measured += 1
        else:
            e_p = float(final.predict([features_for_spec(comp_spec)])[0])
            source = "model"       # std excludes component model uncertainty
            std = sd_d / k
            n_model += 1
        per_token_rows.append({
            "key": s["key"], "model": s["model"], "arch": s["arch"],
            "precision": s["precision"], "src_len": s["seq_len"],
            "context": s["seq_len"] if s["arch"] == "decoder_only"
            else s["tgt_ctx"],
            "per_token_j": (e_d - e_p) / k, "std_j": std,
            "subtraction": source,
        })
    out(f"  rows: {len(per_token_rows)} decode points "
        f"({n_measured} measured-subtracted, {n_model} model-subtracted; "
        f"model-subtracted std excludes component model uncertainty)")
    out("  decoder-only fp16 per-token (mJ):")
    dec_rows = [row for row in per_token_rows
                if row["arch"] == "decoder_only" and row["precision"] == "fp16"]
    for model in sorted({row["model"] for row in dec_rows}):
        cells = "  ".join(
            f"ctx{row['context']}={1000 * row['per_token_j']:.1f}"
            f"{'*' if row['subtraction'] == 'model' else ''}"
            for row in sorted(dec_rows, key=lambda x: x["context"])
            if row["model"] == model)
        out(f"    {model:14s} {cells}")
    out("    (* = model-subtracted; full rows incl. fp32 and enc-dec in JSON)")

    # ------------------------------------------------------------------
    # Artifacts
    test_keys = {r["spec"]["key"] for r in test_r}
    with (FIT_DIR / "per_point_predictions.jsonl").open(
            "w", encoding="utf-8") as fh:
        for r, f, y, yh in zip(records, feats, ys, yhat_full):
            fh.write(json.dumps({
                "key": r["spec"]["key"], "arch": r["spec"]["arch"],
                "phase": r["spec"]["phase"],
                "precision": r["spec"]["precision"],
                "y_j": float(y), "yhat_full_j": float(yh),
                "ape_pct": float(abs(y - yh) / y * 100.0),
                "in_main_test": r["spec"]["key"] in test_keys,
                "mcer_fit": _jf(mcer_by_point[r["spec"]["key"]]),
            }) + "\n")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "git_commit": commit,
        "seed": SEED,
        "n_records": len(records),
        "pooled_ab_median": float(pooled),
        "pooled_ab_target_met": bool(ab_pass),
        "main_table": [{
            "name": r.name, "n_params": r.n_params,
            "r2_train": float(r.r2_train), "r2_test": float(r.r2_test),
            "mape_test": float(r.mape_test), "aic": float(r.aic),
            "bic": float(r.bic),
            "coef": dict(zip(r.column_names, map(float, r.coef))),
        } for r in results],
        "winner": winner.name,
        "final_coef_all296": dict(zip(final.column_names,
                                      map(float, final.coef_))),
        "robustness": robustness,
        "extrapolation": extrap,
        "mcer_source": mcer_note,
        "mcer_summary": mcer_summary,
        "decode_per_token": per_token_rows,
    }
    (FIT_DIR / "fit_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (FIT_DIR / "fit_report.txt").write_text("\n".join(LINES) + "\n",
                                            encoding="utf-8")
    print()
    print(f"wrote {FIT_DIR / 'fit_report.txt'}")
    print(f"wrote {FIT_DIR / 'fit_results.json'}")
    print(f"wrote {FIT_DIR / 'per_point_predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
