#!/usr/bin/env python
"""EXP-002 Step 4: pre-registered representativeness check (random-init vs
pretrained weights), per configs/exp_002.yaml `representativeness` and the
2026-07-24 operationalization note in LOGBOOK.md.

PRIMARY cells (12): fp16, flash, batch 1
  GPT-2      prefill s=512   x {pretrained, random seed 42, 1234, 2025}
  GPT-2      decode  ctx=512 K=64 growing x {same four arms}
  BERT-base  encode  s=512   x {same four arms}
VERDICT (pre-registered, FROZEN): PASS iff max over the 9 primary fp16
ratios <= 0.33. Recorded 2026-08-10: FAIL at 0.3303. The verdict is computed
from the primary cells ONLY and re-prints identically on every run.

FOLLOW-UP A cells (4; fired 2026-07-24 trigger; NOT part of the verdict):
  GPT-2 prefill s=512 fp32 x {pretrained, random seed 42, 1234, 2025}
Result 2026-08-10: ratios 0.121-0.125, init CV 0.22%; fp16-saturation
mechanism supported; flag scoped to compute-bound fp16.

FOLLOW-UP B cells (3; implementation-free isolation; predictions pre-stated
in findings.md 2026-08-10 BEFORE any B code existed):
  GPT-2 prefill s=512 fp16 weights=ported     (HF values in OUR stack,
                                               logit-verified before measuring)
  GPT-2 prefill s=512 fp32 weights=ported
  GPT-2 prefill s=512 fp16 weights=random_v   (bias+wpe random: structure
                                               control for the ported arm)
B comparisons (all |x - y| / y, y named per line):
  pure value effect   = ported fp16 vs random_v fp16      [predicted 0.10-0.20]
  structure delta     = random_v fp16 vs random fp16 s42  [predicted ~0]
  implementation floor= HF-pretrained fp16 vs ported fp16 [predicted 0.12-0.16]
  fp32 value effect   = ported fp32 vs random fp32 (x3)   [predicted 0.00-0.03]
An ok ported record implies the fp32 logit-equivalence gate passed (the
loader refuses unverified ports). MUST run on the RTX 4090 Laptop GPU.

Metric: y = per_execution_median_j[B] / inner_iters per cell.
Resumable (last-write-wins on spec.key; identical command). Repos this run
fetches are deleted from the HF cache on exit; pre-existing ones kept.

KEY NOTE (2026-08-10): for decoder-only decode the physical context is
spec.seq_len; the key's ctx{tgt_ctx} segment shows the unused tgt_ctx default
(128). All analysis derives keys from the cell objects (never hand-built).

GATE NOTE (2026-08-10): the provenance gate ignores untracked copies of this
harness's OWN output files (and only those). Any other dirt still refuses.

Run from the repo root:
    python scripts/run_exp002_representativeness.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tomltransformers.measure.hf_cache import ephemeral_hf_repos
from tomltransformers.sweep.point import PointSpec, measure_single_point

RUN_DIR = REPO / "experiments" / "exp_002_size_sweep"
OUT = RUN_DIR / "representativeness.jsonl"
REPORT_TXT = RUN_DIR / "representativeness_report.txt"
REPORT_JSON = RUN_DIR / "representativeness_report.json"

OWN_OUTPUTS = {
    "experiments/exp_002_size_sweep/representativeness.jsonl",
    "experiments/exp_002_size_sweep/representativeness_report.txt",
    "experiments/exp_002_size_sweep/representativeness_report.json",
}

BAND = 0.33
INIT_SEEDS = (42, 1234, 2025)
PRETRAINED_INPUT_SEED = 42
HF_IDS = {"GPT-2": "gpt2", "BERT-base": "bert-base-uncased"}

LINES: list[str] = []


def out(s: str = "") -> None:
    LINES.append(s)
    print(s)


def git_clean_or_die() -> str:
    st = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                        capture_output=True, text=True)
    if st.returncode != 0:
        print("[gate] git status failed; refusing to run"); sys.exit(2)
    offending = []
    for line in st.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:].strip().strip('"')
        if status == "??" and path.replace("\\", "/") in OWN_OUTPUTS:
            continue
        offending.append(line)
    if offending:
        print("[gate] working tree is DIRTY beyond this harness's own outputs; "
              "commit first (provenance gate):")
        for line in offending:
            print(line)
        sys.exit(2)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    print(f"[gate] tree clean at {sha[:9]} (own outputs permitted)")
    return sha


def _arms(cells: list[PointSpec], base_kwargs: dict) -> None:
    cells.append(PointSpec(**base_kwargs, weights="pretrained",
                           pretrained_id=HF_IDS[base_kwargs["model"]],
                           seed=PRETRAINED_INPUT_SEED))
    for sd in INIT_SEEDS:
        cells.append(PointSpec(**base_kwargs, weights="random", seed=sd))


def build_primary_cells() -> list[PointSpec]:
    cells: list[PointSpec] = []
    _arms(cells, dict(model="GPT-2", arch="decoder_only", phase="prefill",
                      seq_len=512, precision="fp16", attn_kind="flash"))
    _arms(cells, dict(model="GPT-2", arch="decoder_only", phase="decode",
                      seq_len=512, precision="fp16", attn_kind="flash",
                      decode_tokens=64, decode_mode="growing"))
    _arms(cells, dict(model="BERT-base", arch="encoder_only", phase="encode",
                      seq_len=512, precision="fp16", attn_kind="flash"))
    return cells


def build_followup_fp32_cells() -> list[PointSpec]:
    cells: list[PointSpec] = []
    _arms(cells, dict(model="GPT-2", arch="decoder_only", phase="prefill",
                      seq_len=512, precision="fp32", attn_kind="flash"))
    return cells


def build_followup_b_cells() -> list[PointSpec]:
    base = dict(model="GPT-2", arch="decoder_only", phase="prefill",
                seq_len=512, attn_kind="flash")
    return [
        PointSpec(**base, precision="fp16", weights="ported",
                  pretrained_id="gpt2", seed=PRETRAINED_INPUT_SEED),
        PointSpec(**base, precision="fp32", weights="ported",
                  pretrained_id="gpt2", seed=PRETRAINED_INPUT_SEED),
        PointSpec(**base, precision="fp16", weights="random_v",
                  seed=PRETRAINED_INPUT_SEED),
    ]


def load_latest(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("spec", {}).get("key")
            if key:
                latest[key] = rec
    return latest


def append_record(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def y_of(rec: dict) -> float:
    return float(rec["per_execution_median_j"]["B"]) / int(rec["inner_iters"])


def y_mean_unit(rec: dict) -> float:
    return float(rec["per_unit_j"]["B"])


def _done(latest: dict[str, dict], key: str) -> bool:
    rec = latest.get(key, {})
    return bool(rec.get("ok")) and not rec.get("short_window")


def measure_all(cells: list[PointSpec]) -> dict[str, dict]:
    latest = load_latest(OUT)
    todo = [c for c in cells if not _done(latest, c.key())]
    print(f"[plan] {len(cells)} cells; {len(cells) - len(todo)} resume-complete; "
          f"{len(todo)} to measure")
    t0 = time.perf_counter()
    for i, ps in enumerate(todo, 1):
        repeats = 10 if ps.phase == "decode" else 5
        print(f"[rep] ({i}/{len(todo)}) measuring {ps.key()} (repeats={repeats}) ...",
              flush=True)
        rec = measure_single_point(ps, target_s=4.0, repeats=repeats,
                                   min_window_s=4.0)
        append_record(OUT, rec)
        latest[ps.key()] = rec
        if rec.get("ok"):
            print(f"[rep]   -> OK y={y_of(rec):.4g} J/exec "
                  f"(elapsed {time.perf_counter() - t0:.0f}s)")
        else:
            print(f"[rep]   -> FAILED: {rec.get('error') or rec.get('skip_reason')}")
    return latest


def _group(cells: list[PointSpec]):
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for c in cells:
        gk = (c.model, c.phase, c.precision)
        if gk not in groups:
            groups[gk] = {"pre": None, "rand": {}}
            order.append(gk)
        if c.weights == "pretrained":
            groups[gk]["pre"] = c
        else:
            groups[gk]["rand"][c.seed] = c
    return order, groups


def _ratio_table(cells, latest, rows_out):
    order, groups = _group(cells)
    section_max = 0.0
    for gk in order:
        model, phase, precision = gk
        label = f"{model}/{phase}/s512/{precision}"
        g = groups[gk]
        pre = latest[g["pre"].key()]
        e_pre, u_pre = y_of(pre), y_mean_unit(pre)
        rnd_ys = []
        for sd in INIT_SEEDS:
            rnd = latest[g["rand"][sd].key()]
            e_rnd = y_of(rnd)
            rnd_ys.append(e_rnd)
            ratio = abs(e_rnd - e_pre) / e_pre
            ratio_u = abs(y_mean_unit(rnd) - u_pre) / u_pre
            section_max = max(section_max, ratio)
            rows_out.append({"point": label, "seed": sd,
                             "e_pretrained_j": e_pre, "e_random_j": e_rnd,
                             "ratio": ratio, "ratio_per_unit": ratio_u,
                             "within_band": bool(ratio <= BAND)})
            out(f"{label:31s} {sd:5d}  {e_pre:9.4g}  {e_rnd:9.4g}  "
                f"{ratio:7.4f}  {ratio_u:7.4f}")
        m = statistics.mean(rnd_ys)
        cv = (statistics.pstdev(rnd_ys) / m) if m else 0.0
        out(f"{label:31s} random-arm init-seed CV: {100 * cv:.2f}%")
    return section_max


def _r(x: float, y: float) -> float:
    return abs(x - y) / y


def analyze(primary, followup_a, followup_b, latest) -> int:
    out("EXP-002 representativeness report (pre-registered; band 0.33)")
    out(f"generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    out(f"data: {OUT}")
    out("note: decoder-only decode context is spec.seq_len (=512); the key's")
    out("ctx128 segment is the unused tgt_ctx default (see module docstring).")
    everything = primary + followup_a + followup_b
    missing = [c.key() for c in everything if not _done(latest, c.key())]
    if missing:
        out(f"INCOMPLETE: {len(missing)} cells not resume-complete; re-run to finish:")
        for k in missing:
            out(f"  - {k}")
        REPORT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        return 1

    out()
    out("== PRIMARY (pre-registered verdict; fp16; FROZEN cells) ==")
    out("point                           seed   E_pre(J)    E_rand(J)   ratio     ratio(per_unit)")
    primary_rows: list[dict] = []
    worst = _ratio_table(primary, latest, primary_rows)
    verdict = "PASS" if worst <= BAND else "FAIL"
    out()
    out(f"VERDICT: max primary ratio {worst:.4f} vs band {BAND} -> {verdict}")
    if verdict == "FAIL":
        out("  Representativeness NOT established for random-init energies; the")
        out("  sweep is flagged per the pre-registration. Logged as a finding.")
    else:
        out("  Random-init energies are representative within the pre-registered")
        out("  band; the sweep's Fork-1 premise stands. Logged as a finding.")

    out()
    out("== FOLLOW-UP A (fp32; post-verdict, trigger 2026-07-24; NOT part of")
    out("   the pre-registered verdict; mechanism probe) ==")
    out("point                           seed   E_pre(J)    E_rand(J)   ratio     ratio(per_unit)")
    followup_rows: list[dict] = []
    fmax = _ratio_table(followup_a, latest, followup_rows)
    out(f"  max fp32 ratio {fmax:.4f} (band {BAND} as reference only).")

    out()
    out("== FOLLOW-UP B (values ported into OUR stack; implementation-free;")
    out("   post-verdict; predictions pre-stated findings.md 2026-08-10) ==")
    b = {(c.precision, c.weights): latest[c.key()] for c in followup_b}
    e_ported16 = y_of(b[("fp16", "ported")])
    e_ported32 = y_of(b[("fp32", "ported")])
    e_rv16 = y_of(b[("fp16", "random_v")])

    def cell(cells_list, precision, weights, seed):
        for c in cells_list:
            if (c.precision, c.weights, c.seed) == (precision, weights, seed):
                return latest[c.key()]
        raise KeyError((precision, weights, seed))

    prim_prefill = [c for c in primary
                    if c.model == "GPT-2" and c.phase == "prefill"]
    e_hf16 = y_of(cell(prim_prefill, "fp16", "pretrained", PRETRAINED_INPUT_SEED))
    e_r16 = {sd: y_of(cell(prim_prefill, "fp16", "random", sd))
             for sd in INIT_SEEDS}
    e_hf32 = y_of(cell(followup_a, "fp32", "pretrained", PRETRAINED_INPUT_SEED))
    e_r32 = {sd: y_of(cell(followup_a, "fp32", "random", sd))
             for sd in INIT_SEEDS}

    out(f"  E(ported fp16)={e_ported16:.4g} J  E(random_v fp16 s42)={e_rv16:.4g} J  "
        f"E(HF fp16)={e_hf16:.4g} J")
    pure16 = _r(e_ported16, e_rv16)
    out(f"  pure value effect    |ported-random_v|/random_v (fp16): "
        f"{pure16:.4f}   [predicted 0.10-0.20]")
    sd_delta = _r(e_rv16, e_r16[42])
    out(f"  structure delta      |random_v-random_s42|/random_s42 (fp16): "
        f"{sd_delta:.4f}   [predicted ~0]")
    floor16 = _r(e_hf16, e_ported16)
    out(f"  implementation floor |HF-ported|/ported (fp16, identical values): "
        f"{floor16:.4f}   [predicted 0.12-0.16]")
    pv16 = {sd: _r(e_ported16, e_r16[sd]) for sd in INIT_SEEDS}
    out("  ported vs plain random (fp16, per seed): "
        + "  ".join(f"s{sd}={v:.4f}" for sd, v in pv16.items()))
    pv32 = {sd: _r(e_ported32, e_r32[sd]) for sd in INIT_SEEDS}
    out(f"  E(ported fp32)={e_ported32:.4g} J; fp32 value effect vs random: "
        + "  ".join(f"s{sd}={v:.4f}" for sd, v in pv32.items())
        + "   [predicted 0.00-0.03]")
    floor32 = _r(e_hf32, e_ported32)
    out(f"  implementation floor (fp32): |HF-ported|/ported = {floor32:.4f}")
    out("  (ok ported records imply the fp32 logit-equivalence gate passed.)")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "band": BAND,
        "init_seeds": list(INIT_SEEDS),
        "pretrained_input_seed": PRETRAINED_INPUT_SEED,
        "primary_rows": primary_rows,
        "primary_max_ratio": worst,
        "verdict": verdict,
        "followup_fp32": {"rows": followup_rows, "max_ratio": fmax,
                          "label": "post-verdict mechanism probe; not part of "
                                   "the pre-registered verdict"},
        "followup_b": {
            "label": "implementation-free isolation; predictions pre-stated "
                     "findings.md 2026-08-10",
            "e_ported_fp16_j": e_ported16, "e_ported_fp32_j": e_ported32,
            "e_random_v_fp16_j": e_rv16, "e_hf_fp16_j": e_hf16,
            "e_hf_fp32_j": e_hf32,
            "pure_value_effect_fp16": pure16,
            "structure_delta_fp16": sd_delta,
            "implementation_floor_fp16": floor16,
            "implementation_floor_fp32": floor32,
            "ported_vs_random_fp16": {str(k): v for k, v in pv16.items()},
            "ported_vs_random_fp32": {str(k): v for k, v in pv32.items()},
        },
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT_TXT}")
    print(f"wrote {REPORT_JSON}")
    return 0


def main() -> int:
    git_clean_or_die()
    primary = build_primary_cells()
    followup_a = build_followup_fp32_cells()
    followup_b = build_followup_b_cells()
    with ephemeral_hf_repos(sorted(set(HF_IDS.values()))):
        latest = measure_all(primary + followup_a + followup_b)
    return analyze(primary, followup_a, followup_b, latest)


if __name__ == "__main__":
    raise SystemExit(main())
