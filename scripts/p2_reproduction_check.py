#!/usr/bin/env python3
"""Acceptance check of a reproduced arm against Kolbinger et al. validation IoU.

Pass rule (DESIGN.md §4, reference corrected in amendment 1): every structure
within +-0.05 of the published validation mean image IoU. The mean signed
difference is reported as a diagnostic of systematic bias, not as a criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from davinci.data.dsad import STRUCTURES
from davinci.data.dsad_reference import KOLBINGER_IOU

TOLERANCE = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_json", type=Path)
    ap.add_argument("--prefix", required=True, help="run-name prefix, e.g. A1_")
    ap.add_argument("--kind", choices=("specialist", "combined"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    runs = {k: v for k, v in json.loads(args.eval_json.read_text())["runs"].items()
            if k.startswith(args.prefix)}
    ref_val = KOLBINGER_IOU[("deeplabv3", args.kind, "val")]
    ref_test = KOLBINGER_IOU[("deeplabv3", args.kind, "test")]
    rows = {}
    for s in STRUCTURES:
        image_means = [r["iou_image_mean"][s] for r in runs.values() if r["iou_image_mean"].get(s) is not None]
        by_surgery = {}
        for r in runs.values():
            for surgery, v in r["iou_by_surgery"].get(s, {}).items():
                by_surgery.setdefault(surgery, []).append(v)
        n_images = next(r["images_by_surgery"][s] for r in runs.values() if s in r["images_by_surgery"])
        mean = float(np.mean(image_means))
        rows[s] = {"seeds": len(image_means), "iou_image_mean": mean,
                   "seed_sd": float(np.std(image_means)),
                   "iou_by_surgery": {k: float(np.mean(v)) for k, v in sorted(by_surgery.items())},
                   "images_by_surgery": n_images,
                   "kolbinger_val": ref_val[s], "diff_val": mean - ref_val[s],
                   "kolbinger_test": ref_test[s],
                   "within_tolerance": abs(mean - ref_val[s]) <= TOLERANCE}
    bias = float(np.mean([r["diff_val"] for r in rows.values()]))
    out = {"prefix": args.prefix, "kind": args.kind, "tolerance": TOLERANCE,
           "mean_signed_diff": bias,
           "structures_within": sum(r["within_tolerance"] for r in rows.values()),
           "passed": all(r["within_tolerance"] for r in rows.values()),
           "test_accessed": False, "structures": rows}
    args.output.write_text(json.dumps(out, indent=1))
    print(f"{'structure':28s} {'ours':>6s} {'sd':>5s} {'K-val':>6s} {'diff':>6s}  by surgery (n images)")
    for s, r in rows.items():
        surg = "  ".join(f"{k}:{v:.2f}({r['images_by_surgery'][k]})" for k, v in r["iou_by_surgery"].items())
        flag = "" if r["within_tolerance"] else " *"
        print(f"{s:28s} {r['iou_image_mean']:6.3f} {r['seed_sd']:5.3f} {r['kolbinger_val']:6.2f} "
              f"{r['diff_val']:+6.3f}{flag:2s} {surg}")
    print(f"within {out['structures_within']}/11, mean signed diff {bias:+.3f}, passed={out['passed']}")


if __name__ == "__main__":
    main()
