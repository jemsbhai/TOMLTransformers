#!/usr/bin/env python
"""EXP-002 paper figures (Step 8a).

Draws the eight figures for the UEMCON manuscript from the frozen datasets
and the committed fit artifacts. This script FITS NOTHING. Every predicted
value it plots was computed once, by the fit scripts, and lives in a
committed artifact:

  4090 confirmatory + exploratory   scripts/fit_exp002.py        (db1f984)
  A100 confirmatory (T0-T3)         scripts/fit_exp002_a100.py   (176183d)
  A100 exploratory (M8p)            scripts/explore_exp002_a100.py (4e774c7)

Before any figure is written, the lineage gate in
tomltransformers.figures.data reproduces the per-point quantities and checks
them against those artifacts. On any mismatch the run aborts and writes
nothing, so a figure can never disagree with a recorded verdict.

CONFIRMATORY vs EXPLORATORY. The A100 pre-registered verdicts are T0 PASS,
T1 FAIL, T2 FAIL, T3 FAIL (a100_amendment.md sections 3, 9, 16). The
precision-split model M8p is EXPLORATORY, pre-registered as post-verdict work
in amendment section 16.5(a), and carries no verdict authority. Wherever M8p
appears (F6, F7) it is labeled as such in the legend, and the caller is
expected to repeat the label in the caption.

Run from the repo root with the code committed, so the manifest records the
commit the figures were drawn at:

    python scripts/make_figures.py

Writes paper/figures/*.pdf, *.png and figures_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tomltransformers.figures import data as fd   # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:   # pragma: no cover - environment guard
    raise SystemExit(
        "matplotlib is required for the figures and is not installed.\n"
        "    pip install matplotlib\n"
        f"(import error: {exc})")

OUTDIR = REPO / "paper" / "figures"

# IEEE two-column widths, inches.
W1 = 3.4
W2 = 7.0

PHASE_ORDER = ("prefill", "decode", "encode", "decoder_prefill")
PHASE_MARKER = {"prefill": "o", "decode": "s", "encode": "^",
                "decoder_prefill": "D"}
PRECISION_MARKER = {"fp16": "o", "fp32": "s"}

FIGURES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")

# Set by --no-titles. See save().
STRIP_TITLES = False


# ----------------------------------------------------------------------
# Plumbing
# ----------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return True


def style() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "lines.markersize": 3.5,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
    })


def save(fig, name: str, outdir: Path, formats) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    if STRIP_TITLES:
        # Camera-ready: the description lives in the LaTeX caption, not in
        # the image. Titles stay on by default for review passes.
        for ax in fig.get_axes():
            ax.set_title("")
    written = []
    for ext in formats:
        path = outdir / f"{name}.{ext}"
        fig.savefig(path)
        written.append(str(path.relative_to(REPO)))
    plt.close(fig)
    return written


def identity_line(ax, lo: float, hi: float, bands=()) -> None:
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.8,
            color="0.35", zorder=1)
    for frac in bands:
        ax.plot([lo, hi], [lo * (1 + frac), hi * (1 + frac)],
                linestyle=":", linewidth=0.6, color="0.6", zorder=1)
        ax.plot([lo, hi], [lo * (1 - frac), hi * (1 - frac)],
                linestyle=":", linewidth=0.6, color="0.6", zorder=1)


def short_label(key: str) -> str:
    parts = key.split("|")
    model = parts[1] if len(parts) > 1 else key
    phase = parts[2] if len(parts) > 2 else ""
    seq = ctx = None
    for p in parts:
        if p.startswith("ctx") and p[3:].isdigit():
            ctx = p[3:]
        elif p.startswith("s") and p[1:].isdigit():
            seq = p[1:]
    tag = {"prefill": "pf", "decode": "dec", "encode": "enc",
           "decoder_prefill": "dpf"}.get(phase, phase)
    if phase == "decode" and ctx:
        return f"{model}\n{tag} ctx{ctx}"
    return f"{model}\n{tag} s{seq}" if seq else f"{model}\n{tag}"


def r2_of(pairs) -> float:
    pts = list(pairs)
    ys = [y for y, _ in pts]
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - yh) ** 2 for y, yh in pts)
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def signed_pct(y: float, yhat: float) -> float:
    return 100.0 * (yhat - y) / y


# ----------------------------------------------------------------------
# F1: instrument agreement, both platforms
# ----------------------------------------------------------------------

def fig_f1(outdir, formats):
    y_by_key = {}
    for row in fd.load_4090_predictions():
        y_by_key[row["key"]] = row["y_j"]
    for row in fd.load_a100_predictions():
        y_by_key[row["key"]] = row["y_j"]

    fig, ax = plt.subplots(figsize=(W1, 2.5))
    for label, path, marker in (("RTX 4090", fd.R4090_ENERGY, "o"),
                                ("A100 SXM4", fd.A100_ENERGY, "^")):
        rows = fd.ab_percentages(fd.load_records(path))
        xs, ys = [], []
        for r in rows:
            y = y_by_key.get(r["key"])
            if y is None:
                continue
            xs.append(y)
            ys.append(100.0 * r["ab"])
        med = statistics.median(ys)
        ax.scatter(xs, ys, s=6, marker=marker, alpha=0.55, linewidths=0,
                   label=f"{label} (n={len(ys)}, median {med:.2f}%)")

    ax.axhline(5.0, linestyle="--", linewidth=0.8, color="0.35")
    ax.text(0.98, 0.95, "pre-registered target: pooled median 5%",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured energy per composite execution (J)")
    ax.set_ylabel("Instrument A vs B difference (%)")
    ax.legend(loc="lower left", frameon=False)
    return save(fig, "f1_instrument_agreement", outdir, formats)


# ----------------------------------------------------------------------
# F2: 4090 measured vs predicted, absolute (confirmatory) and R1
# ----------------------------------------------------------------------

def fig_f2(outdir, formats):
    rows = fd.load_4090_predictions()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1), sharex=True, sharey=True)

    panels = (
        (axes[0], "yhat_full_j", "(a) absolute NNLS (confirmatory)"),
        (axes[1], "yhat_r1_j", "(b) R1 relative NNLS (exploratory)"),
    )
    lo = min(min(r["y_j"] for r in rows),
             min(min(r["yhat_full_j"], r["yhat_r1_j"]) for r in rows)) * 0.6
    hi = max(max(r["y_j"] for r in rows),
             max(max(r["yhat_full_j"], r["yhat_r1_j"]) for r in rows)) * 1.6

    for ax, col, title in panels:
        identity_line(ax, lo, hi)
        for phase in PHASE_ORDER:
            sel = [r for r in rows if r["phase"] == phase]
            if not sel:
                continue
            ax.scatter([r[col] for r in sel], [r["y_j"] for r in sel],
                       s=7, marker=PHASE_MARKER[phase], alpha=0.6,
                       linewidths=0, label=phase)
        pairs = [(r["y_j"], r[col]) for r in rows]
        ax.set_title(f"{title}\n$R^2$ = {r2_of(pairs):.3f}, "
                     f"MAPE = {fd.mape_of(pairs):.1f}%")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Predicted energy (J)")
    axes[0].set_ylabel("Measured energy (J)")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.2)
    return save(fig, "f2_4090_measured_vs_predicted", outdir, formats)


# ----------------------------------------------------------------------
# F3: A100 measured vs predicted (T3 all-84 fit), by precision
# ----------------------------------------------------------------------

def fig_f3(outdir, formats):
    rows = [r for r in fd.load_a100_predictions() if r["stratum"] != "spot"]
    shared = [r for r in rows if r["stratum"] == "shared"]
    ext = [r for r in rows if r["stratum"] == "extension"]
    col = "yhat_t3_all84_fit_j"

    fig, ax = plt.subplots(figsize=(W1, 3.0))
    lo = min(min(r["y_j"], r[col]) for r in rows) * 0.6
    hi = max(max(r["y_j"], r[col]) for r in rows) * 1.6
    identity_line(ax, lo, hi, bands=(0.30,))

    for prec in ("fp16", "fp32"):
        sel = [r for r in shared if r["precision"] == prec]
        if sel:
            ax.scatter([r[col] for r in sel], [r["y_j"] for r in sel],
                       s=9, marker=PRECISION_MARKER[prec], alpha=0.65,
                       linewidths=0, label=f"shared grid, {prec}")
    if ext:
        ax.scatter([r[col] for r in ext], [r["y_j"] for r in ext],
                   s=30, marker="*", label="7B extension (fp16)",
                   edgecolors="none")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted energy (J)")
    ax.set_ylabel("Measured energy (J)")
    ax.set_title("A100, M8 fitted on the 84 shared points\n"
                 f"in-sample MAPE {fd.mape_of([(r['y_j'], r[col]) for r in shared]):.1f}%, "
                 f"7B extrapolation {fd.mape_of([(r['y_j'], r[col]) for r in ext]):.1f}%")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2)
    return save(fig, "f3_a100_measured_vs_predicted", outdir, formats)


# ----------------------------------------------------------------------
# F4: fp32/fp16 ratio by platform against the model-implied ceiling
# ----------------------------------------------------------------------

def fig_f4(outdir, formats):
    r4090 = fd.forward_ratios(fd.precision_pairs(fd.load_records(fd.R4090_ENERGY)))
    ra100 = fd.forward_ratios(fd.precision_pairs(fd.load_records(fd.A100_ENERGY)))

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.boxplot([r4090, ra100], widths=0.5, showfliers=False,
               medianprops={"linewidth": 1.2})
    # Deterministic fixed-width jitter (golden angle), so both swarms are the
    # same width regardless of n.
    for i, vals in enumerate((r4090, ra100), start=1):
        xs = [i + 0.24 + 0.05 * math.sin(j * 2.399963) for j in range(len(vals))]
        ax.scatter(xs, vals, s=5, alpha=0.5, linewidths=0)

    ax.axhline(fd.MODEL_IMPLIED_RATIO_CEILING, linestyle="--", linewidth=0.9,
               color="0.35")
    ax.text(0.04, 0.95,
            f"model-implied ceiling {fd.MODEL_IMPLIED_RATIO_CEILING:g}\n"
            r"(any $\beta \geq 0$ under the frozen priors)",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color="0.3")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"RTX 4090\n(n={len(r4090)})",
                        f"A100 SXM4\n(n={len(ra100)})"])
    ax.set_ylabel("Measured fp32 / fp16 energy ratio")
    ax.set_title("Matched-shape precision ratio, forward phases")
    return save(fig, "f4_precision_ratio_by_platform", outdir, formats)


# ----------------------------------------------------------------------
# F5: MCER by architecture and phase, three fits
# ----------------------------------------------------------------------

MCER_GROUPS = (
    ("encoder_only", "encode", "encoder\nencode"),
    ("encoder_decoder", "encode", "enc-dec\nencode"),
    ("decoder_only", "prefill", "decoder\nprefill"),
    ("encoder_decoder", "decoder_prefill", "enc-dec\ndec-prefill"),
    ("encoder_decoder", "decode", "enc-dec\ndecode"),
    ("decoder_only", "decode", "decoder\ndecode"),
)


def fig_f5(outdir, formats):
    a100 = fd.load_a100_fit()["mcer"]["summary"]
    r4090 = fd.load_4090_fit()["mcer_summary"]

    labels, s_a100, s_4090_r1, s_4090_abs = [], [], [], []
    for arch, phase, label in MCER_GROUPS:
        # Both artifacts key this map FLAT, as "arch/phase", not nested.
        entry = a100[f"{arch}/{phase}"]
        labels.append(label)
        s_a100.append(entry["median"])
        s_4090_r1.append(entry["median_4090_r1"])
        s_4090_abs.append(r4090[f"{arch}/{phase}"]["median"])

    x = range(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(W2, 2.7))
    ax.bar([i - width for i in x], s_4090_abs, width,
           label="RTX 4090, absolute NNLS (confirmatory)")
    ax.bar(list(x), s_4090_r1, width, label="RTX 4090, R1 (exploratory)")
    ax.bar([i + width for i in x], s_a100, width,
           label="A100, R1 (confirmatory primary)")

    ax.axhline(1.0, linestyle="--", linewidth=0.9, color="0.35")
    ax.text(len(labels) - 0.45, 1.08, "MCER = 1", fontsize=6.5, color="0.3",
            ha="right")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("MCER (memory / compute energy)")
    ax.set_title("Prefill and decode separate by 4x to 40x on both platforms; "
                 "the crossing of unity is estimator dependent")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    return save(fig, "f5_mcer_phase_separation", outdir, formats)


# ----------------------------------------------------------------------
# F6: residual vs operating point
#
# Deliberately NOT a scatter against clock. Six of the twelve rows sit at
# 1410 MHz and three at 1095 MHz, and at 1410 MHz alone the M8 residual spans
# -70.9% (eager) to +6.0% (fp16 flash forward). Clock does not organise that
# spread; precision and the eager path do. Plotting against clock would
# visually charge mechanism 1 (precision datapath) and mechanism 3 (eager) to
# mechanism 2 (operating point), which is the exact misattribution the
# 2026-08-18 correction in findings.md exists to undo. Rows are therefore
# labeled with their full operating point and ordered by clock, so the
# operating-point effect reads WITHIN the comparable fp16 flash rows.
# ----------------------------------------------------------------------

PHASE_SHORT = {"forward": "fwd", "decode_like": "dec"}


def fig_f6(outdir, formats):
    table = sorted(fd.load_a100_explore()["b_operating_point_table"],
                   key=lambda r: (r["median_clock_mhz"], r["median_p_total_w"]))
    rows = list(range(len(table)))
    labels = [
        f"{r['median_clock_mhz']:.0f} MHz, {r['median_p_total_w']:.0f} W   "
        f"{r['precision']} {r['attn_kind']} "
        f"{PHASE_SHORT.get(r['phase_class'], r['phase_class'])} (n={r['n']})"
        for r in table]

    fig, ax = plt.subplots(figsize=(W2, 4.2))

    # Connector showing what the precision split moves, drawn under the markers.
    for i, r in enumerate(table):
        ax.plot([r["m8_mean_signed_pct"], r["m8p_mean_signed_pct"]], [i, i],
                linewidth=0.8, color="0.6", zorder=1)

    for mkey, marker, label in (("m8", "o", "M8 (confirmatory form)"),
                                ("m8p", "D",
                                 "M8p precision split (EXPLORATORY)")):
        xs = [r[f"{mkey}_mean_signed_pct"] for r in table]
        lo = [r[f"{mkey}_mean_signed_pct"] - r[f"{mkey}_min"] for r in table]
        hi = [r[f"{mkey}_max"] - r[f"{mkey}_mean_signed_pct"] for r in table]
        ax.errorbar(xs, rows, xerr=[lo, hi], fmt="none", elinewidth=0.7,
                    capsize=1.5, alpha=0.55, zorder=2)
        ax.scatter(xs, rows, s=22, marker=marker, label=label, zorder=3,
                   linewidths=0)

    ax.axvline(0.0, linestyle="--", linewidth=0.8, color="0.35")
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Mean signed residual (%), whiskers span the cell group")
    ax.set_title("Residual by operating point and execution path\n"
                 "(rows ordered by SM clock; whiskers are min to max)")
    ax.legend(loc="upper left", frameon=False)
    return save(fig, "f6_operating_point_residual", outdir, formats)


# ----------------------------------------------------------------------
# F7: 7B extrapolation, per point
# ----------------------------------------------------------------------

def fig_f7(outdir, formats):
    m8 = {r["key"]: r for r in fd.load_a100_fit()["t3"]["per_point"]}
    m8p = {r["key"]: r for r in
           fd.load_a100_explore()["a2_t3_repeat"]["per_point"]}
    keys = list(m8)
    missing = [k for k in keys if k not in m8p]
    if missing:
        raise SystemExit(f"F7: exploratory artifact is missing {missing}")

    def sort_key(k):
        r = m8[k]
        parts = k.split("|")
        phase = parts[2]
        seq = next((int(p[1:]) for p in parts
                    if p.startswith("s") and p[1:].isdigit()), 0)
        return (0 if phase == "prefill" else 1, seq, r["key"])

    keys.sort(key=sort_key)
    x = range(len(keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.bar([i - width / 2 for i in x],
           [signed_pct(m8[k]["y_j"], m8[k]["yhat_j"]) for k in keys],
           width, label="M8 (confirmatory, T3 FAIL 42.7%)")
    ax.bar([i + width / 2 for i in x],
           [signed_pct(m8p[k]["y_j"], m8p[k]["yhat_j"]) for k in keys],
           width, label="M8p precision split (EXPLORATORY, 11.0%)")

    for band in (30.0, -30.0):
        ax.axhline(band, linestyle=":", linewidth=0.7, color="0.5")
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.35")
    ax.text(len(keys) - 0.4, 31, "pre-registered band 30%", fontsize=6.5,
            color="0.3", ha="right", va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels([short_label(k) for k in keys], fontsize=6)
    ax.set_ylabel("Signed prediction error (%)")
    ax.set_title("Extrapolation to 7B-class models: decode transfers, "
                 "prefill does not")
    ax.legend(loc="lower left", frameon=False)
    return save(fig, "f7_extrapolation_7b", outdir, formats)


# ----------------------------------------------------------------------
# F8: baseline comparison, both platforms
# ----------------------------------------------------------------------

BASELINE_ORDER = ("M8 (this work)", "M0 FLOPs", "roofline", "layerwise")


def fig_f8(outdir, formats):
    f4090 = fd.load_4090_fit()
    fa100 = fd.load_a100_fit()

    b4090_abs = f4090["baselines"]
    r4090_abs = [b4090_abs["winner_mape"],
                 b4090_abs["comparisons"]["M0_flops"]["mape"],
                 b4090_abs["comparisons"]["roofline"]["mape"],
                 b4090_abs["comparisons"]["layerwise"]["mape"]]
    e4090 = b4090_abs["exploratory_r1"]
    r4090_r1 = [e4090["winner_mape"],
                e4090["baseline_mape"]["M0_flops_R1"],
                e4090["baseline_mape"]["roofline_R1"],
                e4090["baseline_mape"]["layerwise_R1"]]

    def a100_set(estimator):
        d = fa100["d7_baselines"][estimator]
        return [d["winner_mape"],
                d["comparisons"]["M0_flops"]["mape"],
                d["comparisons"]["roofline"]["mape"],
                d["comparisons"]["layerwise"]["mape"]]

    panels = (
        ("(a) RTX 4090, main test split", r4090_abs, r4090_r1,
         f"roofline fitted $P_{{avg}}$ = "
         f"{b4090_abs['roofline_p_avg_w']:.0f} W on a 150 W part"),
        ("(b) A100, T1 test split", a100_set("absolute"), a100_set("R1"),
         f"roofline fitted $P_{{avg}}$ = "
         f"{fa100['d7_baselines']['R1']['roofline_p_avg_w']:.0f} W "
         f"on a {fa100['d7_baselines']['R1']['tdp_w']:.0f} W part"),
    )

    x = range(len(BASELINE_ORDER))
    width = 0.38
    # sharey: these are both MAPE, and a reader will compare across panels.
    # Independent scales would make the 4090 bars look larger than they are.
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), sharey=True)
    for ax, (title, absolute, relative, note) in zip(axes, panels):
        ax.bar([i - width / 2 for i in x], absolute, width,
               label="absolute NNLS")
        ax.bar([i + width / 2 for i in x], relative, width, label="R1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(BASELINE_ORDER, rotation=20, ha="right")
        # The note lives in the title so it cannot collide with the bars or
        # the legend.
        ax.set_title(f"{title}\n{note}")
    axes[0].set_ylabel("Held-out MAPE (%)")
    axes[0].legend(loc="upper left", frameon=False)
    return save(fig, "f8_baselines_by_platform", outdir, formats)


BUILDERS = {"F1": fig_f1, "F2": fig_f2, "F3": fig_f3, "F4": fig_f4,
            "F5": fig_f5, "F6": fig_f6, "F7": fig_f7, "F8": fig_f8}


# ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", default=str(OUTDIR),
                   help="output directory (default paper/figures)")
    p.add_argument("--only", default=None,
                   help="comma separated figure ids, e.g. F4,F5")
    p.add_argument("--formats", default="pdf,png",
                   help="comma separated output formats (default pdf,png)")
    p.add_argument("--gate-only", action="store_true",
                   help="run the lineage gate and exit without drawing")
    p.add_argument("--no-titles", action="store_true",
                   help="strip axes titles for camera-ready (captions carry "
                        "the description)")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    print("Lineage gate (recomputation vs committed artifacts)")
    results = fd.run_lineage_gate(strict=True)
    print(f"  {len(results)} checks passed")
    if args.gate_only:
        for r in results:
            print(r.line())
        return 0

    which = FIGURES if args.only is None else tuple(
        s.strip().upper() for s in args.only.split(","))
    unknown = [w for w in which if w not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown figure ids: {unknown}; known {list(FIGURES)}")

    outdir = Path(args.outdir)
    formats = [s.strip() for s in args.formats.split(",") if s.strip()]

    global STRIP_TITLES
    STRIP_TITLES = bool(args.no_titles)

    style()
    commit = git_commit()
    dirty = git_dirty()
    if dirty:
        print("  NOTE: working tree is dirty; the manifest records that.")

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "titles_stripped": STRIP_TITLES,
        "lineage_gate_checks": len(results),
        "lineage_gate_all_passed": True,
        "sources": {
            "rtx4090_fit": str(fd.R4090_FIT.relative_to(REPO)),
            "rtx4090_predictions": str(fd.R4090_PREDICTIONS.relative_to(REPO)),
            "a100_fit": str(fd.A100_FIT.relative_to(REPO)),
            "a100_predictions": str(fd.A100_PREDICTIONS.relative_to(REPO)),
            "a100_exploratory": str(fd.A100_EXPLORE.relative_to(REPO)),
        },
        "figures": {},
    }

    print(f"\nWriting {len(which)} figures to {outdir}")
    for fid in which:
        written = BUILDERS[fid](outdir, formats)
        manifest["figures"][fid] = written
        print(f"  {fid}: " + ", ".join(written))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest: {(outdir / 'figures_manifest.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
