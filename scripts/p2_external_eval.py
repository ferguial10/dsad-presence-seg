#!/usr/bin/env python3
"""Zero-shot external evaluation of frozen P2 arms (amendment 7).

CholecSeg8k: letter-box to 5:4, predict, crop back. Shared targets: abdominal
wall, liver, gastrointestinal tract (union of predicted colon, small intestine
and stomach). Impossible structures (ureter, IMA, intestinal veins, vesicular
glands, spleen) are hallucinations on every frame.
EndoVis18 (development sequences only): intestine = union of predicted colon and
small intestine.

Per frame and operating point it stores IoU on frames where the target is in
the dense mask, and whether the target (or an impossible structure) is claimed
(> 0.5% of the frame) where it is absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from davinci.data.dsad_torch import MEAN, STD
from davinci.metrics.dsad_eval import HALLUCINATION_FRACTION
from p2_score import Ensemble, LOCK, load_run, sha256

# Watershed grey value -> class id, verified visually on frames before scoring
# (amendment 7). 255 marks thin lines at the image border and 0 a few isolated
# pixels; both are background.
CHOLEC_WATERSHED = {50: 0, 11: 1, 21: 2, 13: 3, 12: 4, 31: 5, 23: 6, 24: 7, 25: 8, 32: 9, 22: 10, 33: 11, 5: 12,
                    255: 0, 0: 0}
CHOLEC_TARGETS = {"abdominal_wall": ({"abdominal_wall"}, 1), "liver": ({"liver"}, 2),
                  "gi_tract": ({"colon", "small_intestine", "stomach"}, 3)}
IMPOSSIBLE = ("ureter", "inferior_mesenteric_artery", "intestinal_veins", "vesicular_glands", "spleen")
ENDOVIS_INTESTINE = (124, 155, 5)
ENDOVIS_TARGETS = {"intestine": ({"colon", "small_intestine"}, None)}


def cholec_frames(root: Path):
    for img in sorted(root.glob("video*/video*_*/frame_*_endo.png")):
        yield {"unit": img.parts[-3], "image": img,
               "mask": img.with_name(img.name.replace("_endo.png", "_endo_watershed_mask.png"))}


def cholec_labels(path: Path) -> np.ndarray:
    grey = np.asarray(Image.open(path).convert("L"))
    out = np.full(grey.shape, -1, np.int16)
    for value, cls in CHOLEC_WATERSHED.items():
        out[grey == value] = cls
    if (out < 0).any():
        raise ValueError(f"{path}: unmapped grey values {np.unique(grey[out < 0])[:10]}")
    return out


def endovis_frames(manifest: Path):
    m = json.loads(manifest.read_text())
    for split in ("train", "validation"):
        for r in m["records"][split]:
            yield {"unit": r["unit"], "image": Path(r["image"]), "mask": Path(r["mask"])}


def to_input(image: np.ndarray, size_hw, letterbox: bool):
    """Returns the model input and the crop box of the original frame in the padded frame."""
    h, w = image.shape[:2]
    top = 0
    if letterbox:  # pad height to the 5:4 aspect ratio of DSAD
        ph = round(w * 4 / 5)
        top = (ph - h) // 2
        padded = np.zeros((ph, w, 3), np.uint8)
        padded[top:top + h] = image
        image = padded
    resized = np.asarray(Image.fromarray(image).resize(size_hw[::-1], Image.BILINEAR), np.float32) / 255
    x = (torch.from_numpy(resized).permute(2, 0, 1) - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    return x, image.shape[:2], (top, top + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", choices=("cholecseg8k", "endovis18"), required=True)
    ap.add_argument("--root", type=Path, required=True, help="CholecSeg8k root or EndoVis18 dev manifest")
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = json.loads(LOCK.read_text())
    for run in args.runs:
        assert lock.get(str(run)) == sha256(run / "best.pt"), f"{run} is not a frozen candidate"
    cal = json.loads(args.calibration.read_text()) if args.calibration else None
    frames = list(cholec_frames(args.root) if args.dataset == "cholecseg8k" else endovis_frames(args.root))
    targets = CHOLEC_TARGETS if args.dataset == "cholecseg8k" else ENDOVIS_TARGETS
    device = torch.device("cuda")

    seeds: dict[int, list] = {}
    for run in args.runs:
        model, ck = load_run(run)
        seeds.setdefault(ck["args"]["seed"], []).append((run, model, ck))
    out = {"arm": args.arm, "dataset": args.dataset, "seeds": {}}
    for seed, items in sorted(seeds.items()):
        if all(len(ck["structures"]) == 1 for _, _, ck in items):
            model = Ensemble({ck["structures"][0]: m for _, m, ck in items})
            structures = list(model.order)
        else:
            model, structures = items[0][1], list(items[0][2]["structures"])
        size_hw = (518, 644) if items[0][2]["args"]["model"].startswith("dinov2") else (512, 640)
        ops = {"default": {s: None for s in structures}}
        if cal is not None:
            ops["calibrated"] = {s: cal["seeds"][str(seed)][s]["lambda"] for s in structures}
        model.to(device).eval()
        rows = []
        with torch.no_grad():
            for fr in frames:
                image = np.asarray(Image.open(fr["image"]).convert("RGB"))
                x, padded_hw, (y0, y1) = to_input(image, size_hw, args.dataset == "cholecseg8k")
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(x[None].to(device))
                logits, presence = o if isinstance(o, tuple) else (o, None)
                logits = F.interpolate(logits.float(), size=padded_hw, mode="bilinear", align_corners=False)
                pred = (logits[0, :, y0:y1] > 0).cpu().numpy()
                prob = None if presence is None else torch.sigmoid(presence.float())[0].cpu().numpy()
                if args.dataset == "cholecseg8k":
                    lab = cholec_labels(fr["mask"])
                    gt = {t: lab == cls for t, (_, cls) in targets.items()}
                else:
                    rgb = np.asarray(Image.open(fr["mask"]).convert("RGB"))
                    gt = {"intestine": (rgb == ENDOVIS_INTESTINE).all(-1)}
                row = {"unit": fr["unit"], "frame": str(fr["image"].name)}
                for op, lams in ops.items():
                    keep = {}
                    for c, s in enumerate(structures):
                        lam = lams[s]
                        if lam is None:
                            keep[s] = True
                        elif prob is not None:
                            keep[s] = prob[structures.index(s)] >= lam
                        else:
                            keep[s] = pred[c].mean() >= lam
                    masks = {s: pred[c] & keep[s] for c, s in enumerate(structures)}
                    for t, (members, _) in targets.items():
                        p = np.any([masks[s] for s in members if s in masks], 0)
                        g = gt[t]
                        if g.any():
                            row[f"{op}/iou/{t}"] = float((p & g).sum() / (p | g).sum())
                        else:
                            row[f"{op}/halluc/{t}"] = float(p.mean() > HALLUCINATION_FRACTION)
                    if args.dataset == "cholecseg8k":
                        for s in IMPOSSIBLE:
                            if s in masks:
                                row[f"{op}/impossible/{s}"] = float(masks[s].mean() > HALLUCINATION_FRACTION)
                rows.append(row)
        model.cpu()
        out["seeds"][str(seed)] = {"runs": [str(r) for r, _, _ in items], "images": rows}
        print(f"seed {seed}: {len(rows)} frames", flush=True)
    args.output.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
