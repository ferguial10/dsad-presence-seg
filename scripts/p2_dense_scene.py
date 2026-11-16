#!/usr/bin/env python3
"""Dense scene evaluation on the multilabel/ test frames (DESIGN.md §5, secondary).

The 365 test frames of ``multilabel/`` carry masks for all seven organs. Each
frozen candidate is used as a full scene parser: logits of the seven organ
channels are upsampled to 1280x1024, channels switched off by the arm's
operating point are dropped, and each pixel takes the organ with the largest
logit, or background when that logit is <= 0. Specialists (A1, A1N) are
combined by the same rule. GT pixels claimed by two organs are ignored.

Operating points follow p2_analyze.py: ``default``/``ungated`` keep every
channel, ``gate0.5`` keeps channels with presence >= 0.5, and
``calibrated:<json>`` applies the frozen per-seed, per-structure threshold to
the presence probability (models with a presence head) or to the predicted
area fraction (models without one).

Per surgery and seed, organ IoU = sum of intersections / sum of unions over
frames (organs with zero union are skipped); scene mIoU is the mean over the
seven organs. Seeds are averaged within surgery, surgeries are the unit
(n = 5: surgeries 2, 7, 11, 18 and 20 have dense frames). Scene hallucination
= share of (frame, organ) pairs with an empty GT mask whose predicted area
exceeds 0.5% of the frame. Arms trained with multilabel/ frames are refused.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from davinci.data.dsad import MULTILABEL_ORGANS, STRUCTURES
from davinci.data.dsad_cache import load_resized
from davinci.data.dsad_torch import MEAN, STD
from davinci.metrics import dense, stats
from davinci.metrics.dsad_eval import HALLUCINATION_FRACTION
from p2_score import LOCK, NATIVE_HW, Ensemble, load_run, sha256

IGNORE = 255
ORGANS = MULTILABEL_ORGANS
CONTRASTS = (("A5p", "A1"), ("A5p", "A1_cal"), ("A5p", "A4"), ("A5p", "A1N"), ("A5p", "A2"),
             ("A5p", "C2"), ("A5p_ungated", "A1"))


class DenseFrames(torch.utils.data.Dataset):
    def __init__(self, manifest: dict, size_hw):
        self.root = Path(manifest["root"])
        self.records = [r for r in manifest["records"] if r["split"] == "test" and r["multilabel_masks"]]
        self.size_hw = size_hw

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        image, _ = load_resized(self.root, {"image": r["image"], "masks": {}}, self.size_hw)
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        masks = []
        for s in ORGANS:
            rel = r["multilabel_masks"].get(s)
            if rel is None:
                masks.append(np.zeros(NATIVE_HW, bool))
                continue
            with Image.open(self.root / rel) as m:
                masks.append(np.asarray(m.convert("L")) > 127)
        gt, overlap = dense.gt_labels(masks, IGNORE)
        return {"image": x, "gt": torch.from_numpy(gt), "overlap": overlap, "index": i}


def keep_matrix(op: str, cal, seed: str, area, prob) -> torch.Tensor:
    """(B, organs) bool: channels kept by the operating point (mirrors p2_analyze)."""
    if op in ("default", "ungated"):
        return torch.ones_like(area, dtype=torch.bool)
    if op == "gate0.5":
        return prob >= 0.5 if prob is not None else torch.ones_like(area, dtype=torch.bool)
    lam = torch.tensor([math.inf if cal["seeds"][seed][s]["lambda"] is None else cal["seeds"][seed][s]["lambda"]
                        for s in ORGANS], device=area.device)
    score = prob if prob is not None else area
    return score >= lam


def counts_torch(pred, gt, n):
    valid = gt != IGNORE
    out = {"inter": [], "union": [], "gt_area": [], "pred_area": []}
    for c in range(n + 1):
        pc, gc = (pred == c) & valid, (gt == c) & valid
        out["inter"].append(int((pc & gc).sum()))
        out["union"].append(int((pc | gc).sum()))
        out["gt_area"].append(int(gc.sum()))
        out["pred_area"].append(int(pc.sum()))
    out["n_pixels"], out["correct"] = int(valid.sum()), int(((pred == gt) & valid).sum())
    return out


def score_arm(scores_path: Path, ops: list[str], manifest: dict, device, workers: int) -> dict:
    scores = json.loads(scores_path.read_text())
    lock = json.loads(LOCK.read_text())
    cals = {}
    for op in ops:
        if op.startswith("calibrated:"):
            p = op.split(":", 1)[1]
            assert lock["_calibrations"].get(p) == sha256(Path(p)), f"{p} not frozen or changed"
            cals[op] = json.loads(Path(p).read_text())
    rows = {op: {} for op in ops}
    for seed, block in scores["seeds"].items():
        runs = [Path(r) for r in block["runs"]]
        items = []
        for run in runs:
            if lock.get(str(run)) != sha256(run / "best.pt"):
                raise SystemExit(f"{run} is not a frozen candidate (or its checkpoint changed)")
            model, ck = load_run(run)
            if ck["args"].get("multilabel"):
                raise SystemExit(f"{run} was trained with multilabel/ frames (excluded by protocol)")
            items.append((model, ck))
        if all(len(ck["structures"]) == 1 for _, ck in items):
            model = Ensemble({ck["structures"][0]: m for m, ck in items})
            structures = tuple(model.order)
        else:
            assert len(items) == 1
            model, structures = items[0][0], tuple(items[0][1]["structures"])
        chans = [structures.index(s) for s in ORGANS]
        size_hw = (518, 644) if items[0][1]["args"]["model"].startswith("dinov2") else (512, 640)
        ds = DenseFrames(manifest, size_hw)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=workers)
        model.to(device).eval()
        checked = False
        with torch.no_grad():
            for batch in dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(batch["image"].to(device))
                logits, presence = o if isinstance(o, tuple) else (o, None)
                logits = F.interpolate(logits[:, chans].float(), size=NATIVE_HW, mode="bilinear",
                                       align_corners=False)
                area = (logits > 0).float().mean((2, 3))
                prob = None if presence is None else torch.sigmoid(presence[:, chans].float())
                gt = batch["gt"].to(device)
                for op in ops:
                    keep = keep_matrix(op.split(":", 1)[0], cals.get(op), seed, area, prob)
                    masked = logits.masked_fill(~keep[:, :, None, None], -math.inf)
                    mx, arg = masked.max(1)
                    pred = torch.where(mx > 0, arg + 1, torch.zeros_like(arg)).to(torch.uint8)
                    if not checked:  # GPU rule == tested numpy rule on one frame
                        ref = dense.scene_labels(logits[0].cpu().numpy(), keep[0].cpu().numpy())
                        assert np.array_equal(ref, pred[0].cpu().numpy())
                        checked = True
                    for b, idx in enumerate(batch["index"].tolist()):
                        r = ds.records[idx]
                        c = counts_torch(pred[b], gt[b], len(ORGANS))
                        c |= {"surgery": r["surgery"], "frame_md5": r["frame_md5"],
                              "gt_overlap_px": int(batch["overlap"][b]),
                              "kept": keep[b].cpu().tolist()}
                        rows[op].setdefault(seed, []).append(c)
        model.cpu()
        print(f"{scores_path.name} seed {seed}: {len(ds)} dense frames", flush=True)
    return rows


def aggregate(rows_by_seed: dict) -> dict:
    """Per-surgery metrics averaged over seeds, plus means over surgeries."""
    acc: dict = {}
    for seed, rows in rows_by_seed.items():
        by: dict = {}
        for r in rows:
            by.setdefault(r["surgery"], []).append(r)
        for g, rs in by.items():
            inter = np.sum([r["inter"] for r in rs], 0)
            union = np.sum([r["union"] for r in rs], 0)
            gt_area = np.sum([r["gt_area"] for r in rs], 0)
            organ_iou = {s: float(inter[c + 1] / union[c + 1]) for c, s in enumerate(ORGANS) if union[c + 1]}
            npx = sum(r["n_pixels"] for r in rs)
            empty = [(r["gt_area"][c + 1] == 0, r["pred_area"][c + 1] / r["n_pixels"] > HALLUCINATION_FRACTION)
                     for r in rs for c in range(len(ORGANS))]
            m = {"miou_organs": float(np.mean(list(organ_iou.values()))),
                 # Post hoc (added after v1 showed spleen absent from all dense test frames):
                 # mean only over organs whose GT is non-empty in this surgery; false positives
                 # of absent organs are counted by scene_hallucination instead.
                 "miou_gt_present": float(np.mean([organ_iou[s] for c, s in enumerate(ORGANS)
                                                   if gt_area[c + 1] and s in organ_iou])),
                 "miou_with_background": float(np.mean(list(organ_iou.values()) + [inter[0] / union[0]])),
                 "pixel_accuracy": sum(r["correct"] for r in rs) / npx,
                 "scene_hallucination": float(np.mean([h for e, h in empty if e])) if any(e for e, _ in empty) else None}
            m |= {f"iou/{s}": v for s, v in organ_iou.items()}
            m |= {f"gt_frames/{s}": float(sum(r["gt_area"][c + 1] > 0 for r in rs)) for c, s in enumerate(ORGANS)}
            for k, v in m.items():
                if v is not None:
                    acc.setdefault(k, {}).setdefault(g, []).append(v)
    per_surgery = {k: {g: float(np.mean(v)) for g, v in sorted(d.items())} for k, d in acc.items()}
    return {"per_surgery": per_surgery,
            "summary": {k: float(np.mean(list(d.values()))) for k, d in per_surgery.items()},
            "n_frames": len(next(iter(rows_by_seed.values()))),
            "gt_overlap_px_total": sum(r["gt_overlap_px"] for r in next(iter(rows_by_seed.values())))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "SCORES", "OP"), required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text())
    device = torch.device("cuda")
    by_scores: dict = {}
    for name, path, op in args.arm:
        by_scores.setdefault(path, []).append((name, op))
    arms = {}
    for path, pairs in by_scores.items():
        rows = score_arm(Path(path), sorted({op for _, op in pairs}), manifest, device, args.workers)
        for name, op in pairs:
            arms[name] = {"op": op, "scores": path} | aggregate(rows[op])
    contrasts = {}
    for a, b in CONTRASTS:
        if {a, b} <= arms.keys():
            for metric in ("miou_organs", "miou_gt_present", "scene_hallucination"):
                pa, pb = arms[a]["per_surgery"][metric], arms[b]["per_surgery"][metric]
                common = sorted(set(pa) & set(pb))
                contrasts[f"{metric}/{a}_minus_{b}"] = stats.paired_contrast(
                    [pa[g] for g in common], [pb[g] for g in common]) | {"surgeries": common}
    out = {"split": "test", "test_accessed": True, "organs": list(ORGANS), "ignore_overlap_pixels": True,
           "note": "n = 5 surgeries: the smallest attainable two-sided exact sign-flip p is 2/32 = 0.0625",
           "arms": arms, "contrasts": contrasts}
    args.output.write_text(json.dumps(out, indent=1))
    for name, a in arms.items():
        s = a["summary"]
        print(f"{name:12s} {a['op'][:10]:10s} mIoU={s['miou_organs']:.3f} gtp={s['miou_gt_present']:.3f} +bg={s['miou_with_background']:.3f} "
              f"acc={s['pixel_accuracy']:.3f} halluc={s.get('scene_hallucination', float('nan')):.3f}")
    for k, v in contrasts.items():
        print(k, round(v["mean_diff"], 3), [round(x, 3) for x in v["ci95"]], v["p_sign_flip"])


if __name__ == "__main__":
    main()
