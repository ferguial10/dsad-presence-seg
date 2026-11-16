"""Pure helpers for two secondary P2 analyses (DESIGN.md §5).

* Annotator ceiling: agreement of each DSAD annotator with the other two.
* Dense scene evaluation: per-pixel argmax over organ channels plus background.
"""
from __future__ import annotations

import itertools

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    """IoU of two boolean masks; None when both are empty."""
    union = np.logical_or(a, b).sum()
    if not union:
        return None
    return float(np.logical_and(a, b).sum() / union)


def annotator_agreement(annos: list[np.ndarray], published: np.ndarray | None = None) -> dict:
    """Per-frame agreement of k annotator masks (k = 3 in DSAD).

    ``loo_majority``: mean over annotators of IoU(annotator, strict majority of
    the others). With two others the strict majority is their intersection
    (the pre-registered definition). ``loo_union``: same against the union of
    the others (lenient bound). ``pairwise``: mean IoU over annotator pairs.
    ``vs_published``: mean IoU of each annotator with the published mask (which
    is itself the majority of all three, so this is optimistic).
    Pairs where both masks are empty are skipped; a key is absent if all were.
    """
    annos = [np.asarray(a, bool) for a in annos]
    k = len(annos)
    out: dict[str, list[float]] = {"loo_majority": [], "loo_union": [], "pairwise": [], "vs_published": []}
    for i in range(k):
        others = np.stack([annos[j] for j in range(k) if j != i])
        need = others.shape[0] // 2 + 1
        for key, ref in (("loo_majority", others.sum(0) >= need), ("loo_union", others.any(0))):
            v = iou(annos[i], ref)
            if v is not None:
                out[key].append(v)
        if published is not None:
            v = iou(annos[i], np.asarray(published, bool))
            if v is not None:
                out["vs_published"].append(v)
    for i, j in itertools.combinations(range(k), 2):
        v = iou(annos[i], annos[j])
        if v is not None:
            out["pairwise"].append(v)
    return {key: float(np.mean(v)) for key, v in out.items() if v}


def scene_labels(logits: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Label map from per-organ logits (C, H, W): 0 = background, c + 1 = organ c.

    A pixel takes the organ with the largest logit among the kept channels and
    is background when that logit is <= 0 (sigmoid <= 0.5) or no channel is kept.
    """
    logits = np.where(np.asarray(keep, bool)[:, None, None], logits, -np.inf)
    best = logits.argmax(0)
    return np.where(logits.max(0) > 0, best + 1, 0).astype(np.uint8)


def gt_labels(masks: list[np.ndarray], ignore: int = 255) -> tuple[np.ndarray, int]:
    """Label map from per-organ GT masks; pixels claimed by >1 organ -> ``ignore``.

    Returns the map and the number of overlap pixels.
    """
    stack = np.stack([np.asarray(m, bool) for m in masks])
    count = stack.sum(0)
    lab = np.where(count == 0, 0, stack.argmax(0) + 1).astype(np.uint8)
    lab[count > 1] = ignore
    return lab, int((count > 1).sum())


def confusion_counts(pred: np.ndarray, gt: np.ndarray, n_organs: int, ignore: int = 255) -> dict:
    """Per-class intersection/union/areas (class 0 = background) on non-ignored pixels."""
    valid = gt != ignore
    p, g = pred[valid], gt[valid]
    out = {"inter": [], "union": [], "gt_area": [], "pred_area": [], "n_pixels": int(valid.sum()),
           "correct": int((p == g).sum())}
    for c in range(n_organs + 1):
        pc, gc = p == c, g == c
        out["inter"].append(int((pc & gc).sum()))
        out["union"].append(int((pc | gc).sum()))
        out["gt_area"].append(int(gc.sum()))
        out["pred_area"].append(int(pc.sum()))
    return out
