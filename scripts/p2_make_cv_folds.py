#!/usr/bin/env python3
"""4-fold surgery-level cross-validation manifests (DESIGN.md §5, amendment 4.4).

Folds: a permutation of the 32 surgery ids with seed 2027, cut into four
groups of eight. In fold k the held-out group is relabelled ``heldout``, three
of the remaining 24 surgeries (seed 2027 + k) become ``val`` for checkpoint
selection and calibration, and the other 21 are ``train``. Run after the
single test evaluation, which it does not replace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 2027


def folds(surgeries):
    perm = np.random.default_rng(SEED).permutation(sorted(surgeries)).tolist()
    out = []
    for k in range(4):
        held = sorted(perm[8 * k: 8 * k + 8])
        rest = [s for s in perm if s not in held]
        val = sorted(np.random.default_rng(SEED + k).choice(rest, 3, replace=False).tolist())
        out.append({"fold": k, "heldout": held, "val": val,
                    "train": sorted(s for s in rest if s not in val)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    definition = folds({r["surgery"] for r in manifest["records"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cv_folds_v1.json").write_text(json.dumps({"seed": SEED, "folds": definition}, indent=1))
    for f in definition:
        role = {s: name for name in ("train", "val", "heldout") for s in f[name]}
        m = dict(manifest)
        m["records"] = [r | {"split": role[r["surgery"]]} for r in manifest["records"]]
        m["split"] = {name: f[name] for name in ("train", "val", "heldout")}
        m["cv_fold"] = f["fold"]
        (args.out_dir / f"fold{f['fold']}_manifest.json").write_text(json.dumps(m))
        print(f)


if __name__ == "__main__":
    main()
