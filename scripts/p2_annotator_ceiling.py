#!/usr/bin/env python3
"""Annotator ceiling on the DSAD test frames (DESIGN.md §5, secondary).

For every test frame and structure with a published mask, each of the three
annotator masks is compared with the strict majority of the other two (their
intersection; pre-registered definition), and, as sensitivity, with their
union, with each other annotator (pairwise) and with the published mask.
Values are averaged over frames within surgery, then over surgeries (the same
aggregation as the model IoU in p2_analyze.py) and reported next to the model
IoU of the test analysis.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from davinci.data.dsad import CRITICAL, STRUCTURES
from davinci.metrics.dense import annotator_agreement

KEYS = ("loo_majority", "loo_union", "pairwise", "vs_published")
MODEL_ARMS = ("A5p", "A5p_ungated", "A1", "A1_cal", "A4", "C2")


def _mask(root: Path, rel: str) -> np.ndarray:
    with Image.open(root / rel) as m:
        return np.asarray(m.convert("L")) > 127


def _job(args):
    root, surgery, s, published, annos = args
    vals = annotator_agreement([_mask(root, a) for a in annos], _mask(root, published))
    return surgery, s, vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True, help="p2_test_analysis_v1.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    jobs, skipped = [], 0
    for r in manifest["records"]:
        if r["split"] != "test":
            continue
        for s, rel in r["masks"].items():
            annos = r["annotators"].get(s, [])
            if len(annos) != 3:
                skipped += 1
                continue
            jobs.append((root, r["surgery"], s, rel, annos))
    acc: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for surgery, s, vals in ex.map(_job, jobs, chunksize=16):
            for k, v in vals.items():
                acc.setdefault(k, {}).setdefault(s, {}).setdefault(surgery, []).append(v)
    per_surgery = {k: {s: {g: float(np.mean(v)) for g, v in sorted(d.items())} for s, d in ks.items()}
                   for k, ks in acc.items()}
    ceiling = {k: {s: float(np.mean(list(d.values()))) for s, d in ks.items()} for k, ks in per_surgery.items()}
    n_frames = {s: sum(len(v) for v in acc["pairwise"].get(s, {}).values()) for s in STRUCTURES}

    analysis = json.loads(args.analysis.read_text())
    models = {a: analysis["arms"][a]["summary"]["iou"] for a in MODEL_ARMS if a in analysis["arms"]}
    table = {}
    for s in STRUCTURES:
        row = {k: ceiling[k].get(s) for k in KEYS} | {"n_frame_masks": n_frames[s]}
        for a, iou in models.items():
            row[f"model/{a}"] = iou.get(s)
            ref = ceiling["loo_majority"].get(s)
            if ref and iou.get(s) is not None:
                row[f"ratio_to_loo_majority/{a}"] = iou[s] / ref
        table[s] = row
    macro = {}
    for name, group in (("critical", CRITICAL), ("all11", STRUCTURES)):
        macro[name] = {k: float(np.mean([ceiling[k][s] for s in group])) for k in KEYS}
        macro[name] |= {f"model/{a}": float(np.mean([iou[s] for s in group])) for a, iou in models.items()}
    out = {"split": "test", "test_accessed": True, "aggregation": "frame -> surgery mean -> mean over surgeries",
           "definition": "loo_majority = IoU(annotator, intersection of the other two) [pre-registered]",
           "masks_skipped_without_3_annotators": skipped, "per_structure": table, "macro": macro,
           "per_surgery": per_surgery}
    args.output.write_text(json.dumps(out, indent=1))
    print(f"{'structure':28s} {'loo_maj':>8s} {'pair':>6s} {'vs_pub':>6s} {'A5p':>6s} {'A1':>6s}  n")
    for s, row in table.items():
        print(f"{s:28s} {row['loo_majority']:8.3f} {row['pairwise']:6.3f} {row['vs_published']:6.3f} "
              f"{row.get('model/A5p', float('nan')):6.3f} {row.get('model/A1', float('nan')):6.3f}  "
              f"{row['n_frame_masks']}")
    for name, m in macro.items():
        print(name, {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    main()
