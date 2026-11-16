"""Resized DSAD cache: lossless copies of the exact tensors the loader would build.

For each frame, ``<md5>.png`` holds the image resized with PIL bilinear and
``<md5>_masks.png`` a 16-bit bitmask whose bit ``i`` is the nearest-resized,
thresholded mask of ``STRUCTURES[i]``. Reading the cache therefore gives the
same arrays as decoding the 1280x1024 originals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from davinci.data.dsad import STRUCTURES


def cache_dir(root: Path, size_hw: tuple[int, int]) -> Path:
    return Path(root) / f"dsad_{size_hw[0]}x{size_hw[1]}"


def load_mask(root: Path, rel: str, size_hw: tuple[int, int]) -> np.ndarray:
    """One binary mask, nearest-resized to (H, W) and thresholded, as uint8."""
    h, w = size_hw
    with Image.open(root / rel) as m:
        m = m.convert("L")
        if m.size != (w, h):
            m = m.resize((w, h), Image.NEAREST)
        return (np.asarray(m) > 127).astype(np.uint8)


def load_resized(root: Path, record: dict, size_hw: tuple[int, int]):
    """Image (H, W, 3) uint8 and {structure: (H, W) uint8} from the originals."""
    h, w = size_hw
    with Image.open(root / record["image"]) as im:
        image = np.array(im.convert("RGB").resize((w, h), Image.BILINEAR))
    masks = {s: load_mask(root, rel, size_hw) for s, rel in record["masks"].items()}
    return image, masks


def pack_masks(masks: dict, size_hw) -> np.ndarray:
    bits = np.zeros(size_hw, np.uint16)
    for s, m in masks.items():
        bits |= m.astype(np.uint16) << STRUCTURES.index(s)
    return bits


def write_entry(directory: Path, record: dict, image, masks) -> None:
    md5 = record["frame_md5"]
    Image.fromarray(image).save(directory / f"{md5}.png", compress_level=1)
    Image.fromarray(pack_masks(masks, image.shape[:2])).save(
        directory / f"{md5}_masks.png", compress_level=1)


def read_entry(directory: Path, record: dict):
    md5 = record["frame_md5"]
    with Image.open(directory / f"{md5}.png") as im:
        image = np.array(im)
    with Image.open(directory / f"{md5}_masks.png") as m:
        bits = np.asarray(m).astype(np.uint16)
    masks = {s: ((bits >> STRUCTURES.index(s)) & 1).astype(np.uint8) for s in record["masks"]}
    return image, masks
