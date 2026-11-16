"""Topology and boundary metrics for thin structures: clDice, HD95, NSD.

All functions take boolean masks of equal shape. When the prediction or the
reference is empty, distances are undefined: HD95 returns ``nan`` and NSD and
clDice return 0 (both empty is handled by the caller, which skips the image).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def cldice(pred: np.ndarray, ref: np.ndarray) -> float:
    if not pred.any() or not ref.any():
        return 0.0
    sp, sr = skeletonize(pred), skeletonize(ref)
    tprec = (sp & ref).sum() / max(sp.sum(), 1)
    tsens = (sr & pred).sum() / max(sr.sum(), 1)
    return float(2 * tprec * tsens / (tprec + tsens)) if tprec + tsens else 0.0


def _border(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def _surface_distances(pred: np.ndarray, ref: np.ndarray):
    bp, br = _border(pred), _border(ref)
    to_ref = ndimage.distance_transform_edt(~br)
    to_pred = ndimage.distance_transform_edt(~bp)
    return to_ref[bp], to_pred[br]


def hd95(pred: np.ndarray, ref: np.ndarray) -> float:
    if not pred.any() or not ref.any():
        return float("nan")
    a, b = _surface_distances(pred, ref)
    return float(np.percentile(np.concatenate([a, b]), 95))


def nsd(pred: np.ndarray, ref: np.ndarray, tau: float) -> float:
    """Normalised surface Dice at tolerance ``tau`` pixels."""
    if not pred.any() or not ref.any():
        return 0.0
    a, b = _surface_distances(pred, ref)
    return float(((a <= tau).sum() + (b <= tau).sum()) / (len(a) + len(b)))
