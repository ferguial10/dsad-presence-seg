#!/usr/bin/env python3
"""Select the A5 encoder on validation (DESIGN.md §4, amendment 1.8) and write A5 specs.

Candidates A3, A4, C1: 3-seed mean of validation critical macro IoU; a
difference below 0.005 is a tie, broken by fewer trainable parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CANDIDATES = {"A3": "dinov2_base_frozen", "A4": "dinov2_large_lora16", "C1": "sam2_baseplus_lora16"}
TIE = 0.005


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=Path("runs/p2"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--common", required=True, help="arguments shared by all A5 runs")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    table = {}
    for arm, model in CANDIDATES.items():
        results = [json.loads((args.runs_root / f"{arm}_seed{s}" / "result.json").read_text())
                   for s in (11, 23, 37)]
        table[arm] = {"model": model, "seeds": [r["best_val"] for r in results],
                      "mean": float(np.mean([r["best_val"] for r in results])),
                      "trainable_parameters": results[0]["trainable_parameters"],
                      "best_epochs": [r["best_epoch"] for r in results]}
    best = max(table.values(), key=lambda t: t["mean"])["mean"]
    tied = [a for a, t in table.items() if best - t["mean"] < TIE]
    chosen = min(tied, key=lambda a: table[a]["trainable_parameters"])
    model = CANDIDATES[chosen]
    batch = "--batch-size 4 --accum-steps 2" if "large" in model else ""
    lines = [f"A5_seed{s} {args.common} --model {model}_ph --use-absent {batch} --seed {s}".replace("  ", " ")
             for s in (11, 23, 37)]
    args.specs.write_text("\n".join(lines) + "\n")
    out = {"rule": "max 3-seed mean val critical macro IoU; tie < 0.005 -> fewer trainable parameters",
           "candidates": table, "tied": tied, "selected": chosen, "a5_model": model + "_ph",
           "test_accessed": False}
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
