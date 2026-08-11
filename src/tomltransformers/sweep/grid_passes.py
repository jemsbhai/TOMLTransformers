"""Multi-pass grid expander: the A100 grid of the approved Step 5 amendment.

The frozen 4090 grid is ONE expand_grid() pass over one config. The A100 grid
(experiments/exp_002_size_sweep/a100_amendment.md, APPROVED 2026-08-10) cannot
be expressed as a single pass: the enc-dec cells are restricted to the anchor
stratum (12 fp16 cells covering both sub-sweep arms plus 6 fp32 anchor cells),
the 7B extension carries its own sequence lists, and the four spot cells carry
per-pass weights arms with a pretrained_id. This module composes MULTIPLE
expand_grid passes mechanically: each pass is a full expand_grid config plus
per-pass expander arguments; passes are expanded by the SAME frozen expander
the 4090 sweep used, concatenated, and deduplicated globally on the seed-less
key (first occurrence wins). No hand-built cells.

Per-point seeds (amendment section 11): every surviving point receives an
EXPLICIT init seed derived deterministically from the master seed and the
point's seed-less key:

    seed = int(sha256(f"{master}|{seedless_key}")[:4]) mod 2^31

The derivation is pure Python (no torch), stable across platforms and re-runs,
so a re-expansion reproduces identical seeds and hence identical final keys
(the seed joins the key; see sweep/point.py), which is what the resumable
driver's done-key matching requires.

The config may carry `expected_points`; when present, expansion hard-fails on
any mismatch, so an accidental edit to the frozen grid file cannot silently
change the enumeration.

Pure and deterministic (no torch, no GPU); unit-tested against the exact
frozen enumeration in tests/test_grid_passes.py.

Pass schema (YAML):

    seed:
      master: 42                    # seed.json master_seed, embedded so the
                                    # frozen grid file is self-contained
    expected_points: 98             # optional integrity guard
    passes:
      - name: shared_main           # documentation only
        weights: random             # expand_grid weights arm for the pass
        include_attention_compare: true
        enc_dec_anchor: 1024
        pretrained_id: null         # optional; set on every point of the pass
        config: {...}               # a standard expand_grid config dict
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from .grid import expand_grid
from .point import PointSpec

_SEED_MOD = 2 ** 31


def derive_seed(master: int, seedless_key: str) -> int:
    """Deterministic per-point init seed from the master seed and seed-less key.

    First 4 bytes of sha256(f"{master}|{seedless_key}"), big-endian, mod 2^31
    (a non-negative int32, valid for torch.manual_seed). Pure Python so the
    derivation is identical on Windows, Linux, and any future platform.
    """
    digest = hashlib.sha256(f"{master}|{seedless_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _SEED_MOD


def expand_passes(cfg: dict) -> List[PointSpec]:
    """Expand a multi-pass config into the ordered, seeded list of PointSpecs.

    Order: passes in file order, expand_grid order within each pass. Global
    dedup on the seed-less key keeps the FIRST occurrence. Every returned point
    carries an explicit derived seed (see module docstring); the seed is
    derived from the seed-less key BEFORE it joins the key, so dedup identity
    and seed derivation cannot disagree.

    Raises ValueError on a missing/empty `passes` list, a pass without a
    `config` dict, a missing/non-int `seed.master`, or an `expected_points`
    mismatch.
    """
    passes = cfg.get("passes")
    if not isinstance(passes, list) or not passes:
        raise ValueError("multi-pass config needs a non-empty 'passes' list")
    seed_cfg = cfg.get("seed") or {}
    master = seed_cfg.get("master")
    if isinstance(master, bool) or not isinstance(master, int):
        raise ValueError(
            "multi-pass config needs seed.master (int): every A100 point "
            "carries an explicit derived init seed (a100_amendment.md, "
            "section 11)"
        )

    out: List[PointSpec] = []
    seen: Dict[str, str] = {}  # seed-less key -> pass name (first occurrence wins)
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
        for ps in specs:
            if pid:
                ps.pretrained_id = str(pid)
            k = ps.key()  # seed-less: ps.seed is still None here
            if k in seen:
                continue
            seen[k] = name
            ps.seed = derive_seed(master, k)
            out.append(ps)

    expected = cfg.get("expected_points")
    if expected is not None and len(out) != int(expected):
        raise ValueError(
            f"frozen grid integrity: expanded {len(out)} points but the config "
            f"declares expected_points={expected}; the grid file or the "
            f"expander changed"
        )
    return out
