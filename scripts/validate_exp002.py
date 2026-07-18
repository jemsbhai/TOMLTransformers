#!/usr/bin/env python
"""Read-only data-quality report for the EXP-002 sweep output.

Run after each nightly chunk, from the repo root:

    python scripts/validate_exp002.py

Optional overrides:

    python scripts/validate_exp002.py --data <energy.jsonl> --summary <sweep_summary.json>

What it checks:
  1.  File integrity: parse errors, schema versions, git commits seen,
      duplicate keys (last-write-wins dedupe).
  2.  Outcomes on the latest record per key (ok / failed / oom / short
      window), whether earlier failed or short-window records were
      cleared by a later re-run, runner notes, and the regime-aware
      repeat protocol (5 forward, 10 decode-like).
  3.  Cross-check against sweep_summary.json.
  4.  Coverage by arch, phase, precision, attention kind, model.
  5.  Pace from record timestamps (gaps over 2 h are treated as chunk
      boundaries and excluded) plus a projection for the remaining grid.
  6.  Instrument agreement: A-B split by phase class against prior
      observed bands, and B-C.
  7.  Repeat noise: CV of instrument B, cv_exceeded frequencies by phase.
  8.  Window sizing: inner_iters and measured wall-time floor.
  9.  Thermal and clocks: settle temps, max temps, median SM clocks with
      the first sample per point dropped (known idle artifact), idle power.
  10. Physics sanity: per-forward energy monotone in seq_len for prefill
      and encode (ViT excluded; seq_len is ignored for vision, so ViT
      points are checked for invariance instead), fp32 > fp16 at matched
      shape, decode per-unit contamination flags.

This script never modifies energy.jsonl. It prints the report and writes
two derived, regenerable artifacts next to the data:
validation_report.txt and validation_report.json.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "experiments" / "exp_002_size_sweep"
DEFAULT_DATA = RUN_DIR / "energy.jsonl"
DEFAULT_SUMMARY = RUN_DIR / "sweep_summary.json"
DEFAULT_JSON = RUN_DIR / "validation_report.json"
DEFAULT_TXT = RUN_DIR / "validation_report.txt"

FORWARD_PHASES = {"prefill", "encode"}
DECODE_PHASES = {"decode", "decoder_prefill"}

# Attention thresholds. These are prior observations from the probes and
# the smoke run, not pre-registered acceptance bands. A WARN is a prompt
# for human inspection, never an automatic pass/fail.
A_B_UPPER_FORWARD = 0.12   # probes showed ~5-10% on prefill/encode
A_B_UPPER_DECODE = 0.17    # probes showed ~13-15%, smoke decode ~5%
B_C_UPPER = 0.03           # usually bit-identical; encoder probe 1.2%
CV_B_UPPER = 0.075         # smoke showed 0.85-4.3%
INNER_ITERS_MIN = 3
WALL_FLOOR_S = 3.9         # min_window_s is 4.0; small float tolerance
IDLE_W_LOW, IDLE_W_HIGH = 1.0, 15.0  # laptop GPU idles ~4-5 W
TEMP_ATTENTION_C = 83.0
MONO_TOL = 0.02            # allow 2% wiggle before calling non-monotone
VIT_INVARIANCE_CV = 0.05
CHUNK_GAP_S = 7200.0

LINES = []
WARNINGS = []
INFOS = []


def out(s=""):
    LINES.append(s)
    print(s)


def warn(s):
    WARNINGS.append(s)


def note(s):
    INFOS.append(s)


def klass(phase):
    return "forward" if phase in FORWARD_PHASES else "decode-like"


def pctile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def dist(name, vals, fmt):
    """Print a five-number summary and return it as a dict (or None)."""
    vals = [v for v in vals if v is not None]
    if not vals:
        out("  {}: (no data)".format(name))
        return None
    sv = sorted(vals)
    d = {
        "n": len(sv),
        "min": sv[0],
        "p25": pctile(sv, 0.25),
        "med": pctile(sv, 0.50),
        "p75": pctile(sv, 0.75),
        "max": sv[-1],
    }
    out("  {}: n={}  min={}  p25={}  med={}  p75={}  max={}".format(
        name, d["n"], fmt.format(d["min"]), fmt.format(d["p25"]),
        fmt.format(d["med"]), fmt.format(d["p75"]), fmt.format(d["max"])))
    return d


def main():
    ap = argparse.ArgumentParser(description="EXP-002 energy.jsonl data-quality report")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--json", dest="json_out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--txt", dest="txt_out", type=Path, default=DEFAULT_TXT)
    args = ap.parse_args()

    if not args.data.exists():
        print("ERROR: data file not found: {}".format(args.data))
        return 2

    J = {"dists": {}}

    records = []
    parse_errors = 0
    with open(args.data, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1
                warn("line {}: JSON parse error".format(lineno))

    out("EXP-002 sweep data-quality report")
    out("generated: {}".format(datetime.now().astimezone().isoformat(timespec="seconds")))
    out("data: {}".format(args.data))

    # ------------------------------------------------------------------
    out()
    out("== 1. File integrity ==")
    schemas = Counter(r.get("schema", "<missing>") for r in records)
    commits = Counter(str(r.get("git_commit", "<missing>"))[:9] for r in records)
    keyed = [r for r in records if r.get("spec", {}).get("key")]
    unkeyed = len(records) - len(keyed)

    history = defaultdict(list)
    for r in keyed:
        history[r["spec"]["key"]].append(r)
    latest = {k: hist[-1] for k, hist in history.items()}
    superseded = len(keyed) - len(latest)

    cleared_sw = 0
    cleared_fail = 0
    for k, hist in history.items():
        if len(hist) < 2:
            continue
        last = hist[-1]
        if any(x.get("short_window") for x in hist[:-1]) and last.get("ok") and not last.get("short_window"):
            cleared_sw += 1
        if any(not x.get("ok") for x in hist[:-1]) and last.get("ok"):
            cleared_fail += 1

    out("  records: {} lines parsed, {} parse errors, {} without spec.key".format(
        len(records), parse_errors, unkeyed))
    out("  schema versions: {}".format(dict(schemas)))
    out("  git commits: {}".format(dict(commits)))
    out("  unique keys: {}  superseded records (last-write-wins): {}".format(
        len(latest), superseded))
    if unkeyed:
        warn("{} records without spec.key".format(unkeyed))
    if len(schemas) > 1:
        warn("multiple schema versions present: {}".format(dict(schemas)))
    if len(commits) > 1:
        note("data spans multiple git commits: {} (expected across chunks if code changed between them)".format(dict(commits)))

    # ------------------------------------------------------------------
    out()
    out("== 2. Outcomes (latest record per key) ==")
    lat = list(latest.values())
    ok = [r for r in lat if r.get("ok")]
    oom = [r for r in lat if r.get("oom_skipped")]
    failed = [r for r in lat if not r.get("ok") and not r.get("oom_skipped")]
    sw = [r for r in lat if r.get("short_window")]
    done = [r for r in lat if r.get("ok") and not r.get("short_window")]

    out("  ok: {}   failed: {}   oom_skipped: {}   short_window: {}".format(
        len(ok), len(failed), len(oom), len(sw)))
    out("  resume-complete (ok and not short_window): {}".format(len(done)))
    out("  earlier short-window records cleared by re-run: {}".format(cleared_sw))
    out("  earlier failed records cleared by re-run: {}".format(cleared_fail))
    for r in failed:
        warn("failed point: {} ({})".format(
            r["spec"]["key"], r.get("skip_reason") or "no skip_reason"))
    for r in oom:
        note("oom_skipped point: {}".format(r["spec"]["key"]))
    for r in sw:
        note("short-window point pending re-run: {}".format(r["spec"]["key"]))

    all_notes = Counter()
    for r in lat:
        for n in (r.get("notes") or []):
            all_notes[str(n)] += 1
    if all_notes:
        out("  runner notes seen:")
        for msg, c in all_notes.most_common():
            out("    [{}x] {}".format(c, msg))

    out("  n_repeats by phase class (protocol: forward=5, decode-like=10):")
    for cls in ("forward", "decode-like"):
        c = Counter(r.get("n_repeats") for r in ok if klass(r["spec"]["phase"]) == cls)
        out("    {:12s} {}".format(cls, dict(c)))
        expected = 5 if cls == "forward" else 10
        for v, cnt in c.items():
            if v != expected:
                warn("{} {} points have n_repeats={} (expected {})".format(cnt, cls, v, expected))

    # ------------------------------------------------------------------
    out()
    out("== 3. sweep_summary.json cross-check ==")
    total_grid = None
    if args.summary.exists():
        try:
            summ = json.loads(args.summary.read_text(encoding="utf-8"))
            prog = summ.get("progress", {})
            total_grid = prog.get("total")
            expected_done = (prog.get("ok") or 0) + (prog.get("skipped_done") or 0)
            out("  summary: total={} measured={} skipped_done={} ok={} failed={} oom={} short_window={} stopped_early={}".format(
                prog.get("total"), prog.get("measured"), prog.get("skipped_done"),
                prog.get("ok"), prog.get("failed"), prog.get("oom"),
                prog.get("short_window"), prog.get("stopped_early")))
            out("  summary elapsed: {:.2f} h   commit: {}".format(
                (summ.get("elapsed_s") or 0) / 3600.0, str(summ.get("git_commit"))[:9]))
            if expected_done == len(done):
                out("  consistent: summary ok+skipped_done == file resume-complete == {}".format(len(done)))
            else:
                note("summary ok+skipped_done={} vs file resume-complete={} (fine if summary predates latest data)".format(
                    expected_done, len(done)))
        except Exception as e:
            warn("could not read sweep_summary.json: {}".format(e))
    else:
        out("  (no sweep_summary.json found)")

    # ------------------------------------------------------------------
    out()
    out("== 4. Coverage (ok points) ==")
    by_arch_phase = Counter(
        (r["spec"].get("arch", "?"), r["spec"].get("phase", "?")) for r in ok)
    for (arch, phase), c in sorted(by_arch_phase.items()):
        out("  {:18s} {:18s} {:4d}".format(arch, phase, c))
    by_prec = Counter(r["spec"].get("precision", "?") for r in ok)
    by_attn = Counter(r["spec"].get("attn_kind", "?") for r in ok)
    by_model = Counter(r["spec"].get("model", "?") for r in ok)
    out("  precision: {}".format(dict(sorted(by_prec.items()))))
    out("  attn_kind: {}".format(dict(sorted(by_attn.items()))))
    out("  models: " + ", ".join(
        "{} {}".format(m, c) for m, c in sorted(by_model.items())))
    if total_grid:
        out("  grid progress: {}/{} resume-complete ({:.1f}%)".format(
            len(done), total_grid, 100.0 * len(done) / total_grid))

    # ------------------------------------------------------------------
    out()
    out("== 5. Pace and projection ==")
    times = []
    for r in keyed:
        ts = r.get("timestamp")
        if ts:
            try:
                times.append(datetime.fromisoformat(ts))
            except ValueError:
                pass
    times.sort()
    gaps = []
    for a, b in zip(times, times[1:]):
        g = (b - a).total_seconds()
        if 0 < g <= CHUNK_GAP_S:
            gaps.append(g)
    gd = None
    eta_h = None
    if gaps:
        gd = dist("seconds per point (chunk-internal gaps)", gaps, "{:.0f}")
        J["dists"]["pace_s"] = gd
        if total_grid is not None and gd is not None:
            remaining = total_grid - len(done)
            eta_h = remaining * gd["med"] / 3600.0
            out("  remaining: {} points -> ~{:.1f} h at median pace (~{:.1f} chunks of 8 h, ~{:.1f} chunks of 6 h)".format(
                remaining, eta_h, eta_h / 8.0, eta_h / 6.0))
    else:
        out("  (not enough timestamps for pace)")

    # ------------------------------------------------------------------
    out()
    out("== 6. Instrument agreement (ok points) ==")
    ab = defaultdict(list)
    ab_points = []
    bc = []
    for r in ok:
        ag = r.get("agreement") or {}
        cls = klass(r["spec"]["phase"])
        v = ag.get("A-B")
        if v is not None:
            ab[cls].append(v)
            ab_points.append((v, cls, r["spec"]["key"]))
        w = ag.get("B-C")
        if w is not None:
            bc.append((w, r["spec"]["key"]))
    J["dists"]["ab_forward_pct"] = dist(
        "A-B forward (prior obs ~5-10%)", [100 * v for v in ab["forward"]], "{:.2f}%")
    J["dists"]["ab_decode_pct"] = dist(
        "A-B decode-like (prior obs ~5-15%)", [100 * v for v in ab["decode-like"]], "{:.2f}%")
    for v, cls, key in sorted(ab_points, reverse=True)[:5]:
        out("    highest A-B: {:.2f}%  [{}]  {}".format(100 * v, cls, key))
    for v, cls, key in ab_points:
        upper = A_B_UPPER_FORWARD if cls == "forward" else A_B_UPPER_DECODE
        if v > upper:
            warn("A-B {:.1f}% above {:.0f}% attention line ({} {})".format(
                100 * v, 100 * upper, cls, key))
    J["dists"]["bc_pct"] = dist("B-C", [100 * v for v, _ in bc], "{:.3f}%")
    for v, key in bc:
        if v > B_C_UPPER:
            warn("B-C {:.2f}% above {:.0f}% ({})".format(100 * v, 100 * B_C_UPPER, key))

    # ------------------------------------------------------------------
    out()
    out("== 7. Repeat noise ==")
    cvb = []
    for r in ok:
        v = (r.get("per_execution_cv") or {}).get("B")
        if v is not None:
            cvb.append((v, r["spec"]["key"], r["spec"]["phase"]))
    J["dists"]["cvb_forward_pct"] = dist(
        "CV(B) forward", [100 * v for v, k, p in cvb if klass(p) == "forward"], "{:.2f}%")
    J["dists"]["cvb_decode_pct"] = dist(
        "CV(B) decode-like", [100 * v for v, k, p in cvb if klass(p) == "decode-like"], "{:.2f}%")
    for v, k, p in sorted(cvb, reverse=True)[:5]:
        out("    highest CV(B): {:.2f}%  {}".format(100 * v, k))
    for v, k, p in cvb:
        if v > CV_B_UPPER:
            warn("CV(B) {:.1f}% above {:.1f}% ({})".format(100 * v, 100 * CV_B_UPPER, k))
    flg = defaultdict(Counter)
    for r in ok:
        lab = "+".join(sorted(r.get("cv_exceeded") or [])) or "none"
        flg[r["spec"]["phase"]][lab] += 1
    out("  cv_exceeded by phase (A-only on forward and A+B+C on decode are the known patterns):")
    for phase in sorted(flg):
        out("    {:18s} {}".format(phase, dict(flg[phase].most_common())))

    # ------------------------------------------------------------------
    out()
    out("== 8. Window sizing ==")
    for cls in ("forward", "decode-like"):
        vals = [r.get("inner_iters") for r in ok
                if klass(r["spec"]["phase"]) == cls]
        key_name = "inner_iters_{}".format(cls.replace("-", "_"))
        J["dists"][key_name] = dist("inner_iters {}".format(cls), vals, "{:.0f}")
    for r in ok:
        ii = r.get("inner_iters")
        if ii is not None and ii < INNER_ITERS_MIN:
            warn("inner_iters={} very low ({})".format(ii, r["spec"]["key"]))
    J["dists"]["wall_s"] = dist(
        "wall_time_s_median", [r.get("wall_time_s_median") for r in ok], "{:.2f}")
    for r in ok:
        wt = r.get("wall_time_s_median")
        if wt is not None and wt < WALL_FLOOR_S and not r.get("short_window"):
            warn("wall {:.2f}s below the 4s floor but not short_window-flagged ({})".format(
                wt, r["spec"]["key"]))

    # ------------------------------------------------------------------
    out()
    out("== 9. Thermal and clocks ==")
    settle = [(r.get("thermal") or {}).get("temp_c") for r in ok]
    J["dists"]["settle_temp_c"] = dist("settle temp (C)", settle, "{:.0f}")
    unsettled = [r["spec"]["key"] for r in ok
                 if r.get("thermal") and not r["thermal"].get("settled")]
    for k in unsettled:
        warn("thermal not settled: {}".format(k))

    maxtemps = []
    for r in ok:
        temps = r.get("temps_c") or []
        if temps:
            maxtemps.append((max(temps), r["spec"]["key"]))
    J["dists"]["max_temp_c"] = dist(
        "max temp during repeats (C)", [t for t, _ in maxtemps], "{:.0f}")
    for t, k in sorted(maxtemps, reverse=True)[:5]:
        out("    hottest: {:.0f} C  {}".format(t, k))
    for t, k in maxtemps:
        if t > TEMP_ATTENTION_C:
            warn("max temp {:.0f} C above {:.0f} C attention line ({})".format(
                t, TEMP_ATTENTION_C, k))

    med_clocks = []
    for r in ok:
        clocks = r.get("clocks_mhz") or []
        sm = [c.get("sm") for c in clocks[1:]
              if isinstance(c, dict) and c.get("sm") is not None]
        if not sm:
            sm = [c.get("sm") for c in clocks
                  if isinstance(c, dict) and c.get("sm") is not None]
        if sm:
            med_clocks.append((pctile(sorted(sm), 0.5), r["spec"]["key"]))
    J["dists"]["median_sm_clock_mhz"] = dist(
        "median SM clock, first sample dropped (MHz)",
        [c for c, _ in med_clocks], "{:.0f}")
    for c, k in sorted(med_clocks)[:5]:
        out("    lowest clocks: {:.0f} MHz  {}".format(c, k))

    J["dists"]["idle_w"] = dist(
        "idle power (W)", [r.get("idle_power_w") for r in ok], "{:.2f}")
    for r in ok:
        ip = r.get("idle_power_w")
        if ip is not None and not (IDLE_W_LOW <= ip <= IDLE_W_HIGH):
            warn("idle power {:.2f} W outside [{:.0f}, {:.0f}] ({})".format(
                ip, IDLE_W_LOW, IDLE_W_HIGH, r["spec"]["key"]))

    # ------------------------------------------------------------------
    out()
    out("== 10. Physics sanity ==")
    cont = defaultdict(Counter)
    for r in ok:
        cont[r["spec"]["phase"]][bool(r.get("per_unit_contaminated"))] += 1
    out("  per_unit_contaminated by phase:")
    for phase in sorted(cont):
        c = cont[phase]
        out("    {:18s} True={} False={}".format(phase, c[True], c[False]))
        if phase == "decode" and c[False]:
            warn("{} decode points not flagged per_unit_contaminated".format(c[False]))
        if phase in FORWARD_PHASES and c[True]:
            warn("{} {} points unexpectedly flagged contaminated".format(c[True], phase))

    mono_groups = defaultdict(list)
    for r in ok:
        s = r["spec"]
        if s["phase"] in FORWARD_PHASES and not str(s.get("model", "")).startswith("ViT"):
            pu = (r.get("per_unit_j") or {}).get("B")
            if pu is not None and s.get("seq_len") is not None:
                g = (s.get("model"), s.get("phase"), s.get("precision"),
                     s.get("attn_kind"), s.get("weights"), s.get("batch_size"))
                mono_groups[g].append((s["seq_len"], pu))
    checked = 0
    violations = 0
    for g, pts in sorted(mono_groups.items()):
        pts = sorted(pts)
        if len(pts) < 2:
            continue
        checked += 1
        for (s1, e1), (s2, e2) in zip(pts, pts[1:]):
            if e2 < e1 * (1.0 - MONO_TOL):
                violations += 1
                warn("non-monotone energy in {}: s{}={:.4g} J/forward > s{}={:.4g} J/forward".format(
                    "/".join(str(x) for x in g[:4]), s1, e1, s2, e2))
    out("  seq_len monotonicity (prefill/encode, non-ViT): {} series checked, {} violations".format(
        checked, violations))

    vit_groups = defaultdict(list)
    for r in ok:
        s = r["spec"]
        if s["phase"] == "encode" and str(s.get("model", "")).startswith("ViT"):
            pu = (r.get("per_unit_j") or {}).get("B")
            if pu is not None:
                vit_groups[(s.get("model"), s.get("precision"), s.get("attn_kind"))].append(pu)
    for g, vals in sorted(vit_groups.items()):
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        cv = (math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) / m) if m else 0.0
        out("  ViT seq_len-invariance {}: n={} CV={:.2f}% (seq_len ignored for vision; doubles as a replicate check)".format(
            "/".join(str(x) for x in g), len(vals), 100 * cv))
        if cv > VIT_INVARIANCE_CV:
            warn("ViT invariance CV {:.1f}% above {:.0f}% ({})".format(
                100 * cv, 100 * VIT_INVARIANCE_CV, g))

    pairs = defaultdict(dict)
    for r in ok:
        s = r["spec"]
        pu = (r.get("per_unit_j") or {}).get("B")
        if pu is None:
            continue
        pk = tuple(sorted((k2, v2) for k2, v2 in s.items()
                          if k2 not in ("precision", "key")))
        pairs[pk][s.get("precision")] = (pu, s.get("key"), s.get("phase"))
    ratios_fwd = []
    inversions = 0
    npairs = 0
    for pk, d in pairs.items():
        if "fp16" in d and "fp32" in d:
            npairs += 1
            e16, k16, ph = d["fp16"]
            e32, _, _ = d["fp32"]
            if e32 <= e16:
                inversions += 1
                warn("fp32 <= fp16 at matched shape: {} (fp32 {:.4g} vs fp16 {:.4g} J/unit)".format(
                    k16, e32, e16))
            if ph in FORWARD_PHASES and e16 > 0:
                ratios_fwd.append(e32 / e16)
    out("  fp32/fp16 matched-shape pairs: {}   inversions: {}".format(npairs, inversions))
    J["dists"]["fp32_over_fp16_forward"] = dist(
        "fp32/fp16 per-unit energy ratio (forward phases)", ratios_fwd, "{:.2f}")

    # ------------------------------------------------------------------
    out()
    out("== Verdict ==")
    if WARNINGS:
        out("  WARNINGS ({}):".format(len(WARNINGS)))
        for w in WARNINGS:
            out("    WARN: {}".format(w))
    else:
        out("  no warnings: no systematic anomalies detected in the data so far")
    if INFOS:
        out("  notes ({}):".format(len(INFOS)))
        for n in INFOS:
            out("    note: {}".format(n))

    J.update({
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_path": str(args.data),
        "lines": len(records),
        "parse_errors": parse_errors,
        "unique_keys": len(latest),
        "superseded": superseded,
        "outcomes": {
            "ok": len(ok), "failed": len(failed), "oom_skipped": len(oom),
            "short_window": len(sw), "resume_complete": len(done),
            "cleared_short_window": cleared_sw, "cleared_failed": cleared_fail,
        },
        "grid_total": total_grid,
        "eta_hours_remaining": eta_h,
        "monotonicity": {"series_checked": checked, "violations": violations},
        "fp_pairs": {"n": npairs, "inversions": inversions},
        "warnings": WARNINGS,
        "notes": INFOS,
    })
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(J, indent=2), encoding="utf-8")
    args.txt_out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print()
    print("wrote {}".format(args.txt_out))
    print("wrote {}".format(args.json_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
