"""PyTorch dataset over the fused DSAD manifest.

Each item returns the image, one binary target channel per structure and a
per-pixel loss weight for every channel:

* state ``mask``: the mask supervises every pixel (weight 1);
* state ``absent``: every pixel is a negative when ``use_absent`` is on;
* state ``unknown``: weight 0, except pixels that carry another structure's
  mask, which are negatives unless the pair may overlap (Kolbinger et al.).

A specialist (``require_mask``) can add ``negative_ratio`` times as many
frames whose presence label for its structure is 0, drawn once with
``negative_seed`` (Kolbinger et al., CVPRW 2024); they need ``use_absent``.

``soft_targets`` replaces each mask by the mean of its three annotator masks
(A6); ``use_multilabel`` adds the dense ``multilabel/`` organ masks, where an
organ without a non-empty mask is absent (ablation).

It also returns the image-level presence labels (``presence``) and whether
each is known (``presence_weight``); manifests before v3 have none.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from davinci.data.dsad import MULTILABEL_ORGANS, STRUCTURES
from davinci.data.dsad_cache import cache_dir, load_mask, load_resized, read_entry

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class DSADDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: dict, split: str, *, structures=STRUCTURES,
                 size_hw=(512, 640), train=False, use_absent=False,
                 other_mask_negatives=True, overlap_pairs=(), require_mask=None,
                 surgeries=None, cache_root=None, negative_ratio=0.0, negative_seed=0,
                 soft_targets=False, use_multilabel=False):
        self.root = Path(manifest["root"])
        self.structures = tuple(structures)
        records = [r for r in manifest["records"] if r["split"] == split]
        if surgeries is not None:
            records = [r for r in records if r["surgery"] in set(surgeries)]
        if require_mask is not None:
            positives = [r for r in records if r["label_state"][require_mask] == "mask"]
            negatives = [r for r in records if r["label_state"][require_mask] == "absent"]
            k = min(len(negatives), round(negative_ratio * len(positives)))
            pick = np.random.default_rng(negative_seed).choice(len(negatives), k, replace=False)
            records = positives + [negatives[i] for i in sorted(pick)]
        else:
            records = [r for r in records if r["masks"]]
        self.records = records
        self.size_hw = tuple(size_hw)
        self.cache = cache_dir(cache_root, self.size_hw) if cache_root else None
        self.train = train
        self.soft_targets = soft_targets
        self.use_multilabel = use_multilabel
        self.use_absent = use_absent
        self.other_mask_negatives = other_mask_negatives
        self.overlap = {frozenset(p) for p in overlap_pairs}
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(degrees=15, scale=(0.75, 1.25)),
        ]) if train else None
        self.jitter = v2.ColorJitter(0.2, 0.2, 0.2, 0.02) if train else None
        self.normalize = v2.Normalize(MEAN, STD)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        r = self.records[index]
        h, w = self.size_hw
        if self.cache is not None:
            image, all_masks = read_entry(self.cache, r)
        else:
            image, all_masks = load_resized(self.root, r, self.size_hw)
        k = len(self.structures)
        target = np.zeros((k, h, w), np.float32)
        weight = np.zeros((k, h, w), np.uint8)
        has_mask = [r["label_state"][s] == "mask" for s in self.structures]
        for c, s in enumerate(self.structures):
            state = r["label_state"][s]
            if state == "mask":
                target[c] = all_masks[s]
                weight[c] = 1
                if self.soft_targets and r["annotators"].get(s):
                    target[c] = np.mean([load_mask(self.root, a, self.size_hw)
                                         for a in r["annotators"][s]], 0)
            elif self.use_multilabel and s in MULTILABEL_ORGANS and r["multilabel_masks"]:
                # Dense frames annotate every visible organ: missing or empty means absent.
                rel = r["multilabel_masks"].get(s)
                m = load_mask(self.root, rel, self.size_hw) if rel else None
                weight[c] = 1
                if m is not None and m.any():
                    target[c], has_mask[c] = m, True
            elif state == "absent" and self.use_absent:
                weight[c] = 1
            elif self.other_mask_negatives:
                for other, m in all_masks.items():
                    if other != s and frozenset((s, other)) not in self.overlap:
                        weight[c] |= m
        img = tv_tensors.Image(torch.from_numpy(image).permute(2, 0, 1))
        tgt = tv_tensors.Mask(torch.from_numpy(target))
        wgt = tv_tensors.Mask(torch.from_numpy(weight))
        if self.augment is not None:
            img, tgt, wgt = self.augment(img, tgt, wgt)  # out-of-frame weight -> 0
            img = self.jitter(img)
        img = self.normalize(v2.functional.to_dtype(img, torch.float32, scale=True))
        presence = [r.get("presence", {}).get(s) for s in self.structures]
        return {"image": img.as_subclass(torch.Tensor),
                "target": tgt.as_subclass(torch.Tensor).float(),
                "weight": wgt.as_subclass(torch.Tensor).float(),
                "presence": torch.tensor([float(p or 0) for p in presence]),
                "presence_weight": torch.tensor([float(p is not None) for p in presence]),
                "has_mask": torch.tensor(has_mask),
                "index": index}
