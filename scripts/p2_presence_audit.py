#!/usr/bin/env python3
"""Spot-check DSAD presence labels against pixel masks (DESIGN.md §9).

For every train/val frame and every folder row that describes it, a structure
with a non-empty mask should have presence 1. The disagreement rate estimates
how often a presence 0 (used as a whole-image negative) is wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from davinci.data.dsad import FOLDERS_WITHOUT_PRESENCE, STRUCTURES, read_weak_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    weak_cache = {}
    counts = {s: {"rows": 0, "presence_0": 0, "empty_masks": 0} for s in STRUCTURES}
    examples = []
    for r in manifest["records"]:
        if r["split"] == "test":
            continue
        for source in r["sources"]:
            folder, index = source.split("/")
            if folder == "multilabel" or folder in FOLDERS_WITHOUT_PRESENCE:
                continue
            key = (folder, r["surgery"])
            if key not in weak_cache:
                weak_cache[key] = read_weak_labels(root / folder / f"{r['surgery']:02d}" / "weak_labels.csv")
            row = weak_cache[key].get(int(index))
            if row is None:
                continue
            for s, rel in r["masks"].items():
                with Image.open(root / rel) as m:
                    if not (np.asarray(m.convert("L")) > 127).any():
                        counts[s]["empty_masks"] += 1
                        continue
                counts[s]["rows"] += 1
                if row[s] == 0:
                    counts[s]["presence_0"] += 1
                    if len(examples) < 50:
                        examples.append({"frame_md5": r["frame_md5"], "row_from": source, "structure": s})
    for c in counts.values():
        c["disagreement"] = c["presence_0"] / c["rows"] if c["rows"] else None
    total_rows = sum(c["rows"] for c in counts.values())
    out = {"splits": ["train", "val"], "per_structure": counts,
           "overall_disagreement": sum(c["presence_0"] for c in counts.values()) / total_rows,
           "examples": examples, "test_accessed": False}
    args.output.write_text(json.dumps(out, indent=1))
    for s, c in counts.items():
        print(f"{s:28s} rows={c['rows']:5d} presence0={c['presence_0']:4d} "
              f"rate={c['disagreement'] or 0:.4f} empty_masks={c['empty_masks']}")
    print("overall", round(out["overall_disagreement"], 4))


if __name__ == "__main__":
    main()
