#!/usr/bin/env python3
"""Qualitative figure: A1 specialists vs A5' (seed 11) on randomly drawn frames.

Frames are drawn with seed 2027, not chosen by hand: three DSAD test frames
that contain a critical structure and one CholecSeg8k frame. Each panel lists
the structures the model claims (> 0.5% of the frame).
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from davinci.data.dsad import CRITICAL, STRUCTURES  # noqa: E402
from davinci.metrics.dsad_eval import HALLUCINATION_FRACTION  # noqa: E402
from p2_external_eval import to_input  # noqa: E402
from p2_score import Ensemble, load_run  # noqa: E402

SHORT = {"abdominal_wall": "abd. wall", "colon": "colon", "inferior_mesenteric_artery": "IMA",
         "intestinal_veins": "int. veins", "liver": "liver", "pancreas": "pancreas",
         "small_intestine": "small int.", "spleen": "spleen", "stomach": "stomach", "ureter": "ureter",
         "vesicular_glands": "ves. glands"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7f5bd6", "#d23a3a",
          "#1c5cab", "#8a6a00", "#4fa3a5"]


def overlay(image, masks, alpha=0.45):
    out = image.astype(np.float32) / 255
    for s, m in masks.items():
        c = np.array(matplotlib.colors.to_rgb(COLORS[STRUCTURES.index(s)]))
        out[m] = (1 - alpha) * out[m] + alpha * c
    return out


def predict(model, image, size_hw, letterbox, lams=None):
    x, padded_hw, (y0, y1) = to_input(image, size_hw, letterbox)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = model(x[None].cuda())
    logits, presence = o if isinstance(o, tuple) else (o, None)
    logits = F.interpolate(logits.float(), size=padded_hw, mode="bilinear", align_corners=False)
    pred = (logits[0, :, y0:y1] > 0).cpu().numpy()
    prob = None if presence is None else torch.sigmoid(presence.float())[0].cpu().numpy()
    masks = {}
    for c, s in enumerate(STRUCTURES):
        keep = True if lams is None or prob is None else prob[c] >= lams[s]
        if keep and pred[c].mean() > HALLUCINATION_FRACTION:
            masks[s] = pred[c]
    return masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cholec-root", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    rng = random.Random(2027)
    test = [r for r in manifest["records"]
            if r["split"] == "test" and any(r["label_state"][s] == "mask" for s in CRITICAL)]
    frames = [("DSAD test", root / r["image"], r) for r in rng.sample(test, 3)]
    cholec = sorted(args.cholec_root.glob("video*/video*_*/frame_*_endo.png"))
    frames.append(("CholecSeg8k", rng.choice(cholec), None))

    a1 = Ensemble({ck["structures"][0]: m for m, ck in
                   (load_run(p) for p in sorted(Path("runs/p2").glob("A1_*_seed11")))}).cuda().eval()
    a5, _ = load_run(Path("runs/p2/A5p_seed11"))
    a5 = a5.cuda().eval()
    lams = json.loads(args.calibration.read_text())["seeds"]["11"]
    lams = {s: v["lambda"] for s, v in lams.items()}

    fig, axes = plt.subplots(len(frames), 3, figsize=(7.0, 2.0 * len(frames)))
    for i, (source, path, rec) in enumerate(frames):
        image = np.asarray(Image.open(path).convert("RGB"))
        letterbox = source == "CholecSeg8k"
        if rec is not None:
            ref = {s: np.asarray(Image.open(root / p).convert("L")) > 127 for s, p in rec["masks"].items()}
            ref_txt = "reference: " + ", ".join(SHORT[s] for s in ref)
        else:
            ref, ref_txt = {}, "no DSAD structure annotated"
        panels = [(overlay(image, ref), f"{source} — {ref_txt}"),
                  (overlay(image, m1 := predict(a1, image, (512, 640), letterbox)),
                   "A1 claims: " + (", ".join(SHORT[s] for s in m1) or "none")),
                  (overlay(image, m5 := predict(a5, image, (518, 644), letterbox, lams)),
                   "A5$'$ claims: " + (", ".join(SHORT[s] for s in m5) or "none"))]
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title("\n".join(textwrap.wrap(title, 42)), fontsize=6, loc="left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    print("frames:", [str(p) for _, p, _ in frames])


if __name__ == "__main__":
    main()
