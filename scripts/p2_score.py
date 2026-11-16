#!/usr/bin/env python3
"""Final P2 scoring at native resolution (DESIGN.md §4-5).

Logits are upsampled bilinearly to 1280x1024 and thresholded at 0.5. Per image
and structure the raw (ungated) prediction is stored: IoU and Dice on frames
where the structure has a mask, clDice / HD95 / NSD (10 px at native size, i.e.
5 px at 640x512) for the ureter and vessels, and on mask and absent frames the
predicted area fraction and, for models with a presence head, the presence
probability. Presence gating, hallucination (area > 0.5% on absent frames) and
risk-controlled thresholds are applied afterwards by the analysis script.

Runs are grouped by seed. A seed whose runs are all single-structure
specialists is scored as their 11-structure ensemble (A1).

``--split test`` requires ``CANDIDATES_FROZEN.json`` (run directory ->
sha256 of best.pt) and refuses any run that is not listed or whose checkpoint
has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from davinci.data.dsad import STRUCTURES
from davinci.data.dsad_torch import DSADDataset
from davinci.metrics import boundary
from davinci.models.seg import build_model

NATIVE_HW = (1024, 1280)
THIN = ("ureter", "inferior_mesenteric_artery", "intestinal_veins")
NSD_TAU = 10.0
LOCK = Path("papers/P2_anatomy_segmentation/CANDIDATES_FROZEN.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Ensemble(nn.Module):
    """Concatenate single-structure models into one 11-channel output."""

    def __init__(self, models: dict[str, nn.Module]):
        super().__init__()
        self.order = [s for s in STRUCTURES if s in models]
        self.models = nn.ModuleDict(models)

    def forward(self, x):
        return torch.cat([self.models[s](x) for s in self.order], 1)


def load_run(run: Path):
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["args"]["model"], len(ck["structures"]))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def native_mask(root: Path, rel: str) -> np.ndarray:
    with Image.open(root / rel) as m:
        return np.asarray(m.convert("L")) > 127


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test", "heldout"), default="val",
                    help="heldout = cross-validation fold manifests (after the test evaluation)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.split == "test":
        lock = json.loads(LOCK.read_text())
        for run in args.runs:
            if lock.get(str(run)) != sha256(run / "best.pt"):
                raise SystemExit(f"{run} is not a frozen candidate (or its checkpoint changed)")
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    device = torch.device("cuda")

    seeds: dict[int, list] = {}
    for run in args.runs:
        model, ck = load_run(run)
        seeds.setdefault(ck["args"]["seed"], []).append((run, model, ck))
    out = {"arm": args.arm, "split": args.split, "native_hw": NATIVE_HW, "nsd_tau_px": NSD_TAU,
           "test_accessed": args.split == "test", "seeds": {}}
    for seed, items in sorted(seeds.items()):
        if all(len(ck["structures"]) == 1 for _, _, ck in items):
            model = Ensemble({ck["structures"][0]: m for _, m, ck in items})
            structures = tuple(model.order)
        else:
            assert len(items) == 1, f"seed {seed}: several multi-structure runs"
            model, structures = items[0][1], tuple(items[0][2]["structures"])
        model_name = items[0][2]["args"]["model"]
        size_hw = (518, 644) if model_name.startswith("dinov2") else (512, 640)
        ds = DSADDataset(manifest, args.split, structures=structures, size_hw=size_hw,
                         train=False, cache_root=None)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=args.workers)
        model.to(device)
        rows = []
        with torch.no_grad():
            for batch in dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(batch["image"].to(device))
                logits, presence = o if isinstance(o, tuple) else (o, None)
                logits = F.interpolate(logits.float(), size=NATIVE_HW, mode="bilinear",
                                       align_corners=False)
                pred = (logits > 0).cpu().numpy()
                prob = None if presence is None else torch.sigmoid(presence.float()).cpu().numpy()
                for b, idx in enumerate(batch["index"].tolist()):
                    r = ds.records[idx]
                    row = {"surgery": r["surgery"], "frame_md5": r["frame_md5"]}
                    for c, s in enumerate(structures):
                        p = pred[b, c]
                        state = r["label_state"][s]
                        if state in ("mask", "absent"):
                            row[f"area/{s}"] = float(p.mean())
                            if prob is not None:
                                row[f"presence/{s}"] = float(prob[b, c])
                        if state == "mask":
                            t = native_mask(root, r["masks"][s])
                            inter, union = (p & t).sum(), (p | t).sum()
                            if not union:
                                continue
                            row[f"iou/{s}"] = float(inter / union)
                            row[f"dice/{s}"] = float(2 * inter / (p.sum() + t.sum()))
                            if s in THIN:
                                row[f"cldice/{s}"] = boundary.cldice(p, t)
                                row[f"hd95/{s}"] = boundary.hd95(p, t)
                                row[f"nsd/{s}"] = boundary.nsd(p, t, NSD_TAU)
                    rows.append(row)
        model.cpu()
        out["seeds"][str(seed)] = {"runs": [str(run) for run, _, _ in items],
                                   "structures": list(structures), "images": rows}
        print(f"seed {seed}: {len(rows)} images scored", flush=True)
    args.output.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
