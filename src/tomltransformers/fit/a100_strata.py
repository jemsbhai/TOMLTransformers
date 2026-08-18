"""Strata of the frozen A100 grid, resolved mechanically by pass name.

The A100 grid (configs/exp_002_a100.yaml) is a multi-pass config expanded by
sweep/grid_passes.expand_passes; the passes are named after the amendment's
strata (shared_main, shared_enc_dec_fp16, shared_enc_dec_fp32_anchor,
extension_7b, spot_random, spot_random_v, spot_ported, spot_pretrained_hf).
expand_passes returns a flat, seeded, globally deduplicated list and does not
tag points with their pass, so the fit needs the same enumeration grouped by
pass: shared (84 points, T1/T2/T3 fits), extension (10, T3 targets), spot (4,
descriptive only; excluded from every fit and test).

strata_by_pass replays exactly the expand_passes loop (same expand_grid call,
same first-occurrence-wins dedup on the seed-less key, in file order) and
records which pass each seed-less key belongs to. tests/test_a100_strata.py
asserts the union equals expand_passes' enumeration and pins the per-pass
sizes, so the two can never disagree silently. Pure CPU.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from ..sweep.grid import expand_grid
from .a100_calibration import seedless_key

STRATUM_PREFIXES = {"shared": "shared_", "extension": "extension_", "spot": "spot_"}


def strata_by_pass(cfg: Mapping) -> Dict[str, List[str]]:
    """Pass name -> ordered seed-less keys, replaying expand_passes' dedup."""
    passes = cfg.get("passes")
    if not isinstance(passes, list) or not passes:
        raise ValueError("multi-pass config needs a non-empty 'passes' list")
    seen: set[str] = set()
    out: Dict[str, List[str]] = {}
    for i, p in enumerate(passes):
        name = str(p.get("name", f"pass{i}"))
        sub = p.get("config")
        if not isinstance(sub, dict):
            raise ValueError(f"pass '{name}' needs a 'config' dict")
        specs = expand_grid(
            sub,
            enc_dec_anchor=int(p.get("enc_dec_anchor", 1024)),
            include_attention_compare=bool(p.get("include_attention_compare", True)),
            weights=str(p.get("weights", "random")),
        )
        pid = p.get("pretrained_id")
        keys: List[str] = []
        for ps in specs:
            if pid:
                ps.pretrained_id = str(pid)
            k = ps.key()          # seed-less: ps.seed is None here
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)
        if name in out:
            raise ValueError(f"duplicate pass name {name!r}")
        out[name] = keys
    return out


def stratum_of_pass(pass_name: str) -> str:
    for stratum, prefix in STRATUM_PREFIXES.items():
        if pass_name.startswith(prefix):
            return stratum
    raise ValueError(f"pass name {pass_name!r} matches no stratum prefix "
                     f"{sorted(STRATUM_PREFIXES.values())}")


def key_to_stratum(cfg: Mapping) -> Dict[str, str]:
    """Seed-less key -> 'shared' | 'extension' | 'spot'."""
    out: Dict[str, str] = {}
    for name, keys in strata_by_pass(cfg).items():
        s = stratum_of_pass(name)
        for k in keys:
            out[k] = s
    return out


def partition_records(records: Sequence[Mapping], cfg: Mapping) -> Dict[str, List[dict]]:
    """Group sweep records into the three strata by their seed-less key.

    Every record must resolve to a frozen-grid key; a record outside the
    enumeration raises (the dataset is exactly the frozen grid, so anything
    else is a bug to surface). Record order is preserved within each stratum.
    """
    k2s = key_to_stratum(cfg)
    out: Dict[str, List[dict]] = {s: [] for s in STRATUM_PREFIXES}
    for r in records:
        k = seedless_key(r["spec"]["key"])
        if k not in k2s:
            raise ValueError(f"record key not in the frozen A100 enumeration: {r['spec']['key']}")
        out[k2s[k]].append(dict(r))
    return out
