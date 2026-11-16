"""DSAD evaluation at training resolution: per-image IoU and hallucination.

IoU is computed per image on the images where the structure has a mask;
hallucination is whether more than 0.5% of an image with presence label 0 is
predicted as the structure. Both are aggregated per surgery first.

Models with a presence head are scored after presence gating (a structure's
mask is emptied when its presence probability is below 0.5); the ungated
scores are kept under ``"ungated"``.
"""
from __future__ import annotations

import numpy as np
import torch

from davinci.data.dsad import CRITICAL

HALLUCINATION_FRACTION = 0.005


def _mean(values):
    return float(np.mean(values)) if values else None


@torch.no_grad()
def predict(model, loader, device):
    """Yield (batch, binary predictions, gated predictions or None)."""
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch["image"].to(device, non_blocking=True))
        logits, presence = out if isinstance(out, tuple) else (out, None)
        pred = (logits.float() > 0).cpu()
        gated = None if presence is None else pred & (presence.float() > 0).cpu()[:, :, None, None]
        yield batch, pred, gated


def evaluate(model, loader, structures, device, per_image: bool = False) -> dict:
    model.eval()
    raw, gated = _Accumulator(loader.dataset, structures), _Accumulator(loader.dataset, structures)
    has_gate = False
    for batch, pred, gate in predict(model, loader, device):
        raw.add(batch, pred)
        if gate is not None:
            has_gate = True
            gated.add(batch, gate)
    model.train()
    if not has_gate:
        return raw.result(per_image)
    out = gated.result(per_image)
    out["ungated"] = raw.result(False)
    return out


class _Accumulator:
    def __init__(self, ds, structures):
        self.ds, self.structures = ds, structures
        self.iou = {s: {} for s in structures}
        self.halluc = {s: {} for s in structures}
        self.images = []

    def add(self, batch, pred) -> None:
        ds, structures, iou, halluc = self.ds, self.structures, self.iou, self.halluc
        target = batch["target"].bool()
        for b, idx in enumerate(batch["index"].tolist()):
            r = ds.records[idx]
            row = {"surgery": r["surgery"], "frame_md5": r["frame_md5"]}
            for c, s in enumerate(structures):
                state = r["label_state"][s]
                p, t = pred[b, c], target[b, c]
                if state == "mask":
                    union = (p | t).sum().item()
                    if union:
                        v = (p & t).sum().item() / union
                        iou[s].setdefault(r["surgery"], []).append(v)
                        row[f"iou/{s}"] = v
                elif state == "absent":
                    v = float(p.float().mean() > HALLUCINATION_FRACTION)
                    halluc[s].setdefault(r["surgery"], []).append(v)
                    row[f"halluc/{s}"] = v
            self.images.append(row)

    def result(self, per_image: bool) -> dict:
        structures, iou, halluc = self.structures, self.iou, self.halluc
        out = {"iou_image_mean": {}, "iou_surgery_mean": {}, "hallucination_surgery_mean": {},
               "iou_by_surgery": {}, "images_by_surgery": {}}
        for s in structures:
            out["iou_image_mean"][s] = _mean([v for vs in iou[s].values() for v in vs])
            by = {str(k): float(np.mean(v)) for k, v in sorted(iou[s].items())}
            out["iou_by_surgery"][s] = by
            out["images_by_surgery"][s] = {str(k): len(v) for k, v in sorted(iou[s].items())}
            out["iou_surgery_mean"][s] = _mean(list(by.values()))
            out["hallucination_surgery_mean"][s] = _mean([np.mean(v) for v in halluc[s].values()])
        crit = [out["iou_surgery_mean"][s] for s in CRITICAL if s in structures]
        allv = [v for v in out["iou_surgery_mean"].values() if v is not None]
        out["critical_macro_iou"] = float(np.mean(crit)) if crit and None not in crit else None
        out["macro_iou"] = _mean(allv)
        if per_image:
            out["images"] = self.images
        return out
