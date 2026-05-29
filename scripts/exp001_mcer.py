"""EXP-001: prefill vs decode MCER phase transition (exploratory, pre-calibration).

Deterministic. Reproduces the table logged in LOGBOOK.md EXP-001 and writes the
numbers (with the full per-phase feature records) to
experiments/exp_001_mcer/results/mcer.json so every value traces to a results
file rather than only to stdout.

These are prior-weighted TO ratios, NOT calibrated energy. See findings.md.
"""

from __future__ import annotations

import json
import os

from tomltransformers.architectures import configs as cf
from tomltransformers.architectures import decoder as dec

OUT_DIR = os.path.join("experiments", "exp_001_mcer", "results")
MODELS = ("GPT-2", "LLaMA-7B", "Mistral-7B")
SEQ_LENS = (512, 2048)
ATTN_SEQ_LENS = (2048, 8192, 16384)
DEVICE = "rtx4090"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {
        "experiment": "EXP-001",
        "status": "exploratory",
        "device": DEVICE,
        "precision": "fp16",
        "metric": "MCER = (to_sram + to_hbm) / (to_mac + to_nonlinear), dimensionless",
        "caveat": "prior-weighted TO ratios, NOT calibrated energy",
        "phase_transition": [],
        "attention_compare": [],
    }

    for s in SEQ_LENS:
        for name in MODELS:
            cfg = cf.get(name)
            pf = dec.prefill(cfg, s, device=DEVICE)
            ds = dec.decode_step(cfg, s, device=DEVICE)
            results["phase_transition"].append({
                "model": name,
                "seq_len": s,
                "prefill_mcer": pf.mcer,
                "decode_mcer": ds.mcer,
                "decode_over_prefill": ds.mcer / pf.mcer,
                "prefill_record": pf.record(),
                "decode_record": ds.record(),
            })

    for s in ATTN_SEQ_LENS:
        flash = dec.prefill(cf.LLAMA_7B, s, device=DEVICE, attn_kind="flash").mcer
        standard = dec.prefill(cf.LLAMA_7B, s, device=DEVICE, attn_kind="standard").mcer
        results["attention_compare"].append({
            "model": "LLaMA-7B",
            "seq_len": s,
            "flash_mcer": flash,
            "standard_mcer": standard,
            "standard_over_flash": standard / flash,
        })

    path = os.path.join(OUT_DIR, "mcer.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"wrote {path}\n")
    for r in results["phase_transition"]:
        print(f"{r['model']:12s} s={r['seq_len']:>5d}  prefill={r['prefill_mcer']:.4f}  "
              f"decode={r['decode_mcer']:.2f}  ratio={r['decode_over_prefill']:.0f}")


if __name__ == "__main__":
    main()
