"""Data layer for the EXP-002 paper figures (Step 8a).

Every figure in scripts/make_figures.py reads from this module. Nothing here
fits anything: the confirmatory and exploratory fits were computed once, by
scripts/fit_exp002.py (4090, artifact lineage db1f984),
scripts/fit_exp002_a100.py (A100 confirmatory, 176183d) and
scripts/explore_exp002_a100.py (A100 exploratory, 4e774c7). This module loads
those committed artifacts and recomputes only two per-point quantities that
the artifacts summarize but do not store point by point:

  1. instrument A vs B agreement per record (the artifacts store medians only)
  2. fp32/fp16 matched-shape ratios per pair (the validator stores a
     five-number summary only)

Both recomputations use the frozen validator's own pairing function, imported
by path so it cannot drift from the code that produced the committed reports.

LINEAGE GATE. run_lineage_gate() recomputes those quantities and compares them
against the corresponding fields in the committed artifacts, and additionally
cross-checks the per-point prediction files against the aggregate verdicts they
were derived from. No gate value is hardcoded here: each expectation is read
from the artifact it must match, so the gate cannot silently drift. Any
mismatch raises GateFailure and the figure script emits nothing. This makes a
figure run a reproduction check on the recorded results rather than an
independent path that might quietly disagree with them.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from ..fit.bridge import load_latest_records

REPO = Path(__file__).resolve().parents[3]
RUN_DIR = REPO / "experiments" / "exp_002_size_sweep"

R4090_ENERGY = RUN_DIR / "energy.jsonl"
R4090_FIT = RUN_DIR / "fit" / "fit_results.json"
R4090_PREDICTIONS = RUN_DIR / "fit" / "per_point_predictions.jsonl"
R4090_VALIDATION = RUN_DIR / "validation_report.json"

A100_DIR = RUN_DIR / "a100"
A100_ENERGY = A100_DIR / "energy.jsonl"
A100_FIT = A100_DIR / "fit" / "fit_results.json"
A100_PREDICTIONS = A100_DIR / "fit" / "per_point_predictions.jsonl"
A100_EXPLORE = A100_DIR / "fit" / "exploratory" / "explore_results.json"
A100_VALIDATION = A100_DIR / "validation_report.json"

VALIDATOR_PATH = REPO / "scripts" / "validate_exp002.py"

# Tolerance for the lineage gate. These are reproductions of the same
# arithmetic over the same frozen bytes, so agreement should be at float
# round-off; the tolerance exists to absorb ordering, not disagreement.
GATE_RTOL = 1e-9

# The analytic ceiling on the fp32/fp16 energy ratio implied by the frozen
# M8_split_dispatch form under its asserted priors (fp16 MAC multiplier 0.33,
# fp16 word 0.5) for any non-negative coefficient vector. Recorded in
# findings.md 2026-08-17 and reproduced in the T0-T3 scorecard.
MODEL_IMPLIED_RATIO_CEILING = 3.03

_VALIDATOR = None


class GateFailure(AssertionError):
    """A recomputation disagreed with the committed artifact it must match."""


@dataclass(frozen=True)
class GateResult:
    name: str
    reproduced: float
    committed: float
    rtol: float
    ok: bool

    def line(self) -> str:
        status = "ok  " if self.ok else "FAIL"
        return (f"  [{status}] {self.name}: reproduced {self.reproduced!r} "
                f"vs committed {self.committed!r}")


def validator():
    """The frozen validator module, imported by path (never re-implemented)."""
    global _VALIDATOR
    if _VALIDATOR is None:
        spec = importlib.util.spec_from_file_location(
            "validate_exp002_for_figures", VALIDATOR_PATH)
        if spec is None or spec.loader is None:
            raise GateFailure(f"cannot load validator at {VALIDATOR_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VALIDATOR = mod
    return _VALIDATOR


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def _json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GateFailure(f"{path}:{lineno}: JSON parse error: {exc}") from exc
    return rows


def load_records(path: Path) -> list[dict]:
    """Frozen sweep records, latest per spec.key (the sweep resume convention)."""
    return load_latest_records(path)


def load_4090_fit() -> dict:
    return _json(R4090_FIT)


def load_4090_predictions() -> list[dict]:
    """296 rows: y_j, yhat_full_j (absolute all-296), yhat_r1_j (R1 all-296)."""
    return _jsonl(R4090_PREDICTIONS)


def load_a100_fit() -> dict:
    return _json(A100_FIT)


def load_a100_predictions() -> list[dict]:
    """98 rows: y_j plus the T1, T2 and T3 predictions and roles."""
    return _jsonl(A100_PREDICTIONS)


def load_a100_explore() -> dict:
    """EXPLORATORY artifact (M8p). Carries no verdict authority."""
    return _json(A100_EXPLORE)


def load_4090_validation() -> dict:
    return _json(R4090_VALIDATION)


def load_a100_validation() -> dict:
    return _json(A100_VALIDATION)


# ----------------------------------------------------------------------
# Recomputed per-point quantities
# ----------------------------------------------------------------------

def ab_percentages(records) -> list[dict]:
    """Per-record instrument agreement, as stored on each frozen record.

    Returns rows with the A-B (and B-C where present) agreement as a
    FRACTION, matching the record field and the fit scripts' convention.
    """
    v = validator()
    rows = []
    for r in records:
        s = r["spec"]
        agree = r.get("agreement") or {}
        if "A-B" not in agree:
            continue
        bc = agree.get("B-C")
        rows.append({
            "key": s["key"],
            "model": s.get("model"),
            "arch": s.get("arch"),
            "phase": s.get("phase"),
            "precision": s.get("precision"),
            "phase_class": ("forward" if s["phase"] in v.FORWARD_PHASES
                            else "decode_like"),
            "ab": float(agree["A-B"]),
            "bc": None if bc is None else float(bc),
        })
    return rows


def precision_pairs(records) -> list[dict]:
    """fp32/fp16 matched-shape pairs, per pair.

    Pairing, energy field (per_unit_j['B']) and forward-phase selection are
    exactly the frozen validator's section 10 logic; only the return shape
    differs (per pair here, five-number summary there).
    """
    v = validator()
    buckets: dict[tuple, dict] = {}
    for r in records:
        s = r["spec"]
        pu = (r.get("per_unit_j") or {}).get("B")
        if pu is None:
            continue
        buckets.setdefault(v.shape_pair_key(s), {})[s.get("precision")] = (
            float(pu), s.get("key"), s.get("phase"))

    rows = []
    for _pair_key, d in buckets.items():
        if "fp16" not in d or "fp32" not in d:
            continue
        e16, k16, phase = d["fp16"]
        e32, k32, _ = d["fp32"]
        rows.append({
            "key_fp16": k16,
            "key_fp32": k32,
            "phase": phase,
            "phase_class": ("forward" if phase in v.FORWARD_PHASES
                            else "decode_like"),
            "e_fp16_j": e16,
            "e_fp32_j": e32,
            "ratio": (e32 / e16) if e16 > 0 else None,
            "inversion": e32 <= e16,
        })
    return rows


def forward_ratios(pairs) -> list[float]:
    """The ratio list the validator summarizes (forward phases, e16 > 0)."""
    return [p["ratio"] for p in pairs
            if p["phase_class"] == "forward" and p["ratio"] is not None]


def mape_of(pairs_y_yhat) -> float:
    """Mean absolute percentage error, matching energy_model.mape."""
    vals = [abs((y - yhat) / y) for y, yhat in pairs_y_yhat if y]
    if not vals:
        return float("nan")
    return 100.0 * sum(vals) / len(vals)


# ----------------------------------------------------------------------
# Lineage gate
# ----------------------------------------------------------------------

def _close(a: float, b: float, rtol: float = GATE_RTOL) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= rtol


def _check(results, name, reproduced, committed, rtol=GATE_RTOL):
    ok = _close(float(reproduced), float(committed), rtol)
    results.append(GateResult(name, float(reproduced), float(committed),
                              rtol, ok))


def run_lineage_gate(*, strict: bool = True) -> list[GateResult]:
    """Reproduce, then compare against the committed artifacts.

    Raises GateFailure on any mismatch when strict. Every expectation is read
    from an artifact; nothing is hardcoded.
    """
    v = validator()
    results: list[GateResult] = []

    fit4090 = load_4090_fit()
    val4090 = load_4090_validation()
    fitA100 = load_a100_fit()
    valA100 = load_a100_validation()

    # --- RTX 4090: instrument agreement and precision pairs ---------------
    rec4090 = load_records(R4090_ENERGY)
    _check(results, "4090 records", len(rec4090), val4090["lines"], 0.0)

    ab4090 = ab_percentages(rec4090)
    _check(results, "4090 pooled A-B median",
           statistics.median([r["ab"] for r in ab4090]),
           fit4090["pooled_ab_median"])

    pairs4090 = precision_pairs(rec4090)
    _check(results, "4090 fp pairs n", len(pairs4090),
           val4090["fp_pairs"]["n"], 0.0)
    _check(results, "4090 fp pair inversions",
           sum(1 for p in pairs4090 if p["inversion"]),
           val4090["fp_pairs"]["inversions"], 0.0)

    fwd4090 = sorted(forward_ratios(pairs4090))
    d4090 = val4090["dists"]["fp32_over_fp16_forward"]
    _check(results, "4090 fp32/fp16 forward n", len(fwd4090), d4090["n"], 0.0)
    for p, field in ((0.0, "min"), (0.25, "p25"), (0.5, "med"),
                     (0.75, "p75"), (1.0, "max")):
        _check(results, f"4090 fp32/fp16 forward {field}",
               v.pctile(fwd4090, p), d4090[field])

    # --- RTX 4090: per-point predictions vs recorded aggregates -----------
    pred4090 = load_4090_predictions()
    _check(results, "4090 prediction rows", len(pred4090),
           fit4090["n_records"], 0.0)
    _check(results, "4090 R1 full-data MAPE",
           mape_of([(r["y_j"], r["yhat_r1_j"]) for r in pred4090]),
           fit4090["exploratory"]["r1_full_mape"], 1e-6)

    # --- A100: instrument agreement and precision pairs -------------------
    recA100 = load_records(A100_ENERGY)
    _check(results, "A100 records", len(recA100), valA100["lines"], 0.0)

    abA100 = ab_percentages(recA100)
    _check(results, "A100 pooled A-B median",
           statistics.median([r["ab"] for r in abA100]),
           fitA100["t0"]["pooled_ab_median"])
    for cls, field in (("forward", "forward_median"),
                       ("decode_like", "decode_like_median")):
        _check(results, f"A100 A-B median ({cls})",
               statistics.median([r["ab"] for r in abA100
                                  if r["phase_class"] == cls]),
               fitA100["t0"][field])

    pairsA100 = precision_pairs(recA100)
    _check(results, "A100 fp pairs n", len(pairsA100),
           valA100["fp_pairs"]["n"], 0.0)
    _check(results, "A100 fp pair inversions",
           sum(1 for p in pairsA100 if p["inversion"]),
           valA100["fp_pairs"]["inversions"], 0.0)

    fwdA100 = sorted(forward_ratios(pairsA100))
    dA100 = valA100["dists"]["fp32_over_fp16_forward"]
    _check(results, "A100 fp32/fp16 forward n", len(fwdA100), dA100["n"], 0.0)
    for p, field in ((0.0, "min"), (0.25, "p25"), (0.5, "med"),
                     (0.75, "p75"), (1.0, "max")):
        _check(results, f"A100 fp32/fp16 forward {field}",
               v.pctile(fwdA100, p), dA100[field])

    # --- A100: per-point predictions vs recorded verdicts -----------------
    predA100 = load_a100_predictions()
    _check(results, "A100 prediction rows", len(predA100),
           fitA100["n_records"], 0.0)
    _check(results, "A100 T1 held-out MAPE",
           mape_of([(r["y_j"], r["yhat_t1_train_fit_j"]) for r in predA100
                    if r["t1_role"] == "test"]),
           fitA100["t1"]["r1_mape_test"], 1e-6)
    _check(results, "A100 T2 eval MAPE",
           mape_of([(r["y_j"], r["yhat_t2_scaled_j"]) for r in predA100
                    if r["t2_role"] == "eval"]),
           fitA100["t2"]["mape_76"], 1e-6)
    _check(results, "A100 T3 pooled MAPE",
           mape_of([(r["y_j"], r["yhat_j"]) for r in fitA100["t3"]["per_point"]]),
           fitA100["t3"]["r1_mape_pooled"], 1e-6)

    # --- A100 exploratory artifact: internal consistency ------------------
    exp = load_a100_explore()
    _check(results, "A100 M8p T3 MAPE (exploratory)",
           mape_of([(r["y_j"], r["yhat_j"])
                    for r in exp["a2_t3_repeat"]["per_point"]]),
           exp["a2_t3_repeat"]["m8p_mape"], 1e-6)
    _check(results, "A100 exploratory cites recorded T3",
           exp["a2_t3_repeat"]["m8_mape_on_record"],
           fitA100["t3"]["r1_mape_pooled"])

    if strict:
        bad = [r for r in results if not r.ok]
        if bad:
            detail = "\n".join(r.line() for r in bad)
            raise GateFailure(
                f"{len(bad)} of {len(results)} lineage checks failed; no "
                f"figures written.\n{detail}")
    return results
