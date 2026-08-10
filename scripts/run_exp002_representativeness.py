#!/usr/bin/env python
"""EXP-002 Step 4: pre-registered representativeness check (random-init vs
pretrained weights), per configs/exp_002.yaml `representativeness` and the
2026-07-24 operationalization note in LOGBOOK.md.

Cells (12): fp16, flash, batch 1
  GPT-2      prefill s=512   x {pretrained, random seed 42, 1234, 2025}
  GPT-2      decode  ctx=512 K=64 growing x {same four arms}
  BERT-base  encode  s=512   x {same four arms}

The pretrained arm is input-seeded at 42 (weights fixed); random arms are
init+input seeded per init_seed. Since gpt2's vocab matches the config, the
same seed yields IDENTICAL input ids across arms, so the comparison isolates
weight values under the 2026-07-24 structural-parity fix.

Metric (primary): y = per_execution_median_j[B] / inner_iters per cell;
ratio = |y_random(seed) - y_pretrained| / y_pretrained per (point, seed).
VERDICT: PASS iff max over all 9 ratios <= 0.33 (pre-registered band).
Secondary: the same ratios on mean-based per_unit_j, reported for
transparency. Logged as a finding either way.

Resumable (last-write-wins on spec.key; identical command). Downloads gpt2
and bert-base-uncased (~0.5 GB each); repos this run fetches are deleted from
the HF cache on exit, pre-existing ones are kept (measure/hf_cache.py).

KEY NOTE (2026-08-10): for decoder-only decode the physical context is
spec.seq_len; the key's ctx{tgt_ctx} segment shows the unused tgt_ctx default
(128). Keys are identity strings; all analysis derives keys from the cell
objects themselves (fix for the 2026-08-10 KeyError: never duplicate key
construction).

GATE NOTE (2026-08-10): the provenance gate ignores untracked copies of this
harness's OWN output files (and only those), so a crash mid-run can be
resumed without committing partial data. Any other dirt still refuses.

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

# The harness's own outputs: permitted as UNTRACKED dirt so resume is never
# self-blocked. Everything else still refuses (provenance gate).
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
    """Provenance gate: refuse any dirt except untracked copies of this
    harness's own output files."""
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


def build_cells() -> list[PointSpec]:
    cells: list[PointSpec] = []

    def arms(base_kwargs):
        cells.append(PointSpec(**base_kwargs, weights="pretrained",
                               pretrained_id=HF_IDS[base_kwargs["model"]],
                               seed=PRETRAINED_INPUT_SEED))
        for sd in INIT_SEEDS:
            cells.append(PointSpec(**base_kwargs, weights="random", seed=sd))

    arms(dict(model="GPT-2", arch="decoder_only", phase="prefill",
              seq_len=512, precision="fp16", attn_kind="flash"))
    arms(dict(model="GPT-2", arch="decoder_only", phase="decode",
              seq_len=512, precision="fp16", attn_kind="flash",
              decode_tokens=64, decode_mode="growing"))
    arms(dict(model="BERT-base", arch="encoder_only", phase="encode",
              seq_len=512, precision="fp16", attn_kind="flash"))
    return cells


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


def analyze(cells: list[PointSpec], latest: dict[str, dict]) -> int:
    out("EXP-002 representativeness report (pre-registered; band 0.33)")
    out(f"generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    out(f"data: {OUT}")
    out("note: decoder-only decode context is spec.seq_len (=512); the key's")
    out("ctx128 segment is the unused tgt_ctx default (see module docstring).")
    missing = [c.key() for c in cells if not _done(latest, c.key())]
    if missing:
        out(f"INCOMPLETE: {len(missing)} cells not resume-complete; re-run to finish:")
        for k in missing:
            out(f"  - {k}")
        REPORT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        return 1

    # Group cells (model, phase) -> pretrained cell + random cells by seed.
    # Keys always come from the cell objects (never hand-built).
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for c in cells:
        gk = (c.model, c.phase)
        if gk not in groups:
            groups[gk] = {"pre": None, "rand": {}}
            order.append(gk)
        if c.weights == "pretrained":
            groups[gk]["pre"] = c
        else:
            groups[gk]["rand"][c.seed] = c

    rows = []
    worst = 0.0
    out()
    out("point                      seed   E_pre(J)    E_rand(J)   ratio     ratio(per_unit)")
    for gk in order:
        model, phase = gk
        label = f"{model}/{phase}/s512"
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
            worst = max(worst, ratio)
            rows.append({"point": label, "seed": sd, "e_pretrained_j": e_pre,
                         "e_random_j": e_rnd, "ratio": ratio,
                         "ratio_per_unit": ratio_u,
                         "within_band": bool(ratio <= BAND)})
            out(f"{label:26s} {sd:5d}  {e_pre:9.4g}  {e_rnd:9.4g}  "
                f"{ratio:7.4f}  {ratio_u:7.4f}")
        m = statistics.mean(rnd_ys)
        cv = (statistics.pstdev(rnd_ys) / m) if m else 0.0
        out(f"{label:26s} random-arm init-seed CV: {100 * cv:.2f}%")
    verdict = "PASS" if worst <= BAND else "FAIL"
    out()
    out(f"VERDICT: max ratio {worst:.4f} vs band {BAND} -> {verdict}")
    if verdict == "FAIL":
        out("  Representativeness NOT established for random-init energies; the")
        out("  sweep is flagged per the pre-registration. Logged as a finding.")
    else:
        out("  Random-init energies are representative within the pre-registered")
        out("  band; the sweep's Fork-1 premise stands. Logged as a finding.")

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "band": BAND,
        "init_seeds": list(INIT_SEEDS),
        "pretrained_input_seed": PRETRAINED_INPUT_SEED,
        "rows": rows,
        "max_ratio": worst,
        "verdict": verdict,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_TXT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT_TXT}")
    print(f"wrote {REPORT_JSON}")
    return 0


def main() -> int:
    git_clean_or_die()
    cells = build_cells()
    with ephemeral_hf_repos(sorted(set(HF_IDS.values()))):
        latest = measure_all(cells)
    return analyze(cells, latest)


if __name__ == "__main__":
    raise SystemExit(main())
