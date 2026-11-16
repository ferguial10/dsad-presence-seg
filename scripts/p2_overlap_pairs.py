#!/usr/bin/env python3
"""Fix the overlap pairs exempt from the other-mask negative rule (DESIGN.md §4).

A pair is exempt if, in training frames where both structures carry a mask,
the intersection covers >= 1% of the smaller mask in >= 5% of those frames.
Masks are read at 512x640 from the resized cache; only train frames are used.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from davinci.data.dsad import STRUCTURES
from davinci.data.dsad_cache import cache_dir, read_entry

MIN_COVER = 0.01
MIN_FRAMES = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    directory = cache_dir(args.cache_root, (512, 640))
    stats = {p: [] for p in itertools.combinations(STRUCTURES, 2)}
    for r in manifest["records"]:
        if r["split"] != "train" or len(r["masks"]) < 2:
            continue
        _, masks = read_entry(directory, r)
        for a, b in itertools.combinations([s for s in STRUCTURES if s in masks], 2):
            smaller = min(masks[a].sum(), masks[b].sum())
            inter = np.logical_and(masks[a], masks[b]).sum()
            stats[(a, b)].append(float(inter / smaller) if smaller else 0.0)
    pairs, table = [], []
    for (a, b), covers in stats.items():
        if not covers:
            continue
        frac = float(np.mean(np.asarray(covers) >= MIN_COVER))
        exempt = frac >= MIN_FRAMES
        table.append({"pair": [a, b], "frames_both_masked": len(covers),
                      "fraction_frames_cover_ge_1pct": frac,
                      "median_cover": float(np.median(covers)), "exempt": exempt})
        if exempt:
            pairs.append([a, b])
    out = {"rule": {"min_cover_of_smaller_mask": MIN_COVER, "min_fraction_of_frames": MIN_FRAMES},
           "split": "train", "size_hw": [512, 640], "overlap_pairs": pairs, "table": table,
           "test_accessed": False}
    args.output.write_text(json.dumps(out, indent=1))
    for row in sorted(table, key=lambda t: -t["fraction_frames_cover_ge_1pct"]):
        print(f"{row['pair'][0]:>27s} x {row['pair'][1]:<27s} n={row['frames_both_masked']:4d} "
              f"frac={row['fraction_frames_cover_ge_1pct']:.3f} exempt={row['exempt']}")


if __name__ == "__main__":
    main()
