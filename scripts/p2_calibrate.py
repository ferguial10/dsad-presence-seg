#!/usr/bin/env python3
"""Calibrate per-structure presence thresholds on validation (amendments 2.3 and 4).

Input: a ``p2_score.py`` output on the validation split. For each seed and
structure, an absent image hallucinates at threshold lam when the predicted
area exceeds 0.5% and its presence score is >= lam. The smallest lam whose
hallucination rate is certified <= alpha (Learn-then-Test, Hoeffding-Bentkus,
confidence 1 - delta) is kept; lam = 0 means no gating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from davinci.metrics.dsad_eval import HALLUCINATION_FRACTION
from davinci.metrics.risk_control import calibrate_threshold

PRESENCE_GRID = list(np.round(np.arange(0, 1.0 + 1e-9, 0.005), 3)) + [1.0001]
AREA_GRID = list(np.logspace(np.log10(HALLUCINATION_FRACTION), 0, 200)) + [1.0001]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scores", type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.scores.read_text())
    assert data["split"] == "val", "thresholds are calibrated on validation only"
    out = {"arm": data["arm"], "alpha": args.alpha, "delta": args.delta,
           "source": str(args.scores), "test_accessed": False, "seeds": {}}
    for seed, block in data["seeds"].items():
        rows, res = block["images"], {}
        for s in block["structures"]:
            absent = [r for r in rows if f"area/{s}" in r and f"iou/{s}" not in r]
            head = any(f"presence/{s}" in r for r in absent)
            if head:
                scores = [r[f"presence/{s}"] if r[f"area/{s}"] > HALLUCINATION_FRACTION else -1.0
                          for r in absent]
                grid, kind = PRESENCE_GRID, "presence_probability"
            else:
                scores = [r[f"area/{s}"] if r[f"area/{s}"] > HALLUCINATION_FRACTION else -1.0
                          for r in absent]
                grid, kind = AREA_GRID, "area_fraction"
            res[s] = calibrate_threshold(scores, args.alpha, args.delta, grid) | {"score": kind}
        out["seeds"][seed] = res
        print(seed, {s[:10]: v["lambda"] for s, v in res.items()}, flush=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
