#!/usr/bin/env python3
"""Merge per-fold score or calibration files, keying each fold as seed ``f<k>``.

Held-out surgeries are disjoint across folds, so the merged file feeds
``p2_analyze.py`` like a single arm whose surgeries each come from one fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs=4, type=Path, help="fold 0..3 in order")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    parts = [json.loads(f.read_text()) for f in args.files]
    merged = {k: v for k, v in parts[0].items() if k != "seeds"}
    merged["seeds"] = {f"f{k}": next(iter(p["seeds"].values())) for k, p in enumerate(parts)}
    if merged.get("split") == "heldout":
        merged["split"] = "cv_heldout"
    merged["merged_from"] = [str(f) for f in args.files]
    args.output.write_text(json.dumps(merged))


if __name__ == "__main__":
    main()
