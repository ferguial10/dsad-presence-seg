#!/usr/bin/env python3
"""Re-evaluate saved DSAD checkpoints with per-surgery and per-image detail.

Each run is scored with its own training configuration (structures, input
size, frame selection). Only ``--split val`` is allowed until the candidates
are frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from davinci.data.dsad_torch import DSADDataset
from davinci.metrics.dsad_eval import evaluate
from davinci.models.seg import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=("val",))
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    device = torch.device("cuda")
    out = {"split": args.split, "manifest": str(args.manifest), "test_accessed": False, "runs": {}}
    for run in args.runs:
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        a, structures = ck["args"], tuple(ck["structures"])
        specialist = len(structures) == 1
        kwargs = dict(structures=structures, use_absent=a["use_absent"],
                      require_mask=structures[0] if specialist else None,
                      other_mask_negatives=not specialist, cache_root=args.cache_root)
        if a["model"].startswith("dinov2"):
            kwargs["size_hw"] = (518, 644)
        ds = DSADDataset(manifest, args.split, train=False, **kwargs)
        dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=args.workers, pin_memory=True)
        model = build_model(a["model"], len(structures))
        model.load_state_dict(ck["model"])
        model.to(device)
        res = evaluate(model, dl, structures, device, per_image=True)
        res["best_epoch"] = ck["epoch"]
        out["runs"][run.name] = res
        print(run.name, {s: res["iou_by_surgery"][s] for s in structures}, flush=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
