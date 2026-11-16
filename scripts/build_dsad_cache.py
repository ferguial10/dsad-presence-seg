#!/usr/bin/env python3
"""Write the resized DSAD cache for the given splits and verify it on a sample.

Only train and val are cached until the candidates are frozen; ``--splits test``
is refused unless ``--allow-test`` is given.
"""
from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from davinci.data.dsad_cache import cache_dir, load_resized, read_entry, write_entry

_STATE = {}


def _init(root, directory, size_hw):
    _STATE.update(root=Path(root), directory=Path(directory), size_hw=size_hw)


def _work(record):
    if (_STATE["directory"] / f"{record['frame_md5']}_masks.png").exists():
        return 0
    image, masks = load_resized(_STATE["root"], record, _STATE["size_hw"])
    write_entry(_STATE["directory"], record, image, masks)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--size", default="512x640", help="HxW")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify", type=int, default=200)
    args = ap.parse_args()
    splits = args.splits.split(",")
    if "test" in splits and not args.allow_test:
        raise SystemExit("test split is locked until the candidates are frozen")
    size_hw = tuple(int(v) for v in args.size.split("x"))
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    records = [r for r in manifest["records"] if r["split"] in splits and r["masks"]]
    directory = cache_dir(args.cache_root, size_hw)
    directory.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, _init, (root, directory, size_hw)) as pool:
        written = sum(pool.imap_unordered(_work, records, chunksize=16))
    sample = random.Random(0).sample(records, min(args.verify, len(records)))
    for r in sample:
        image, masks = load_resized(root, r, size_hw)
        c_image, c_masks = read_entry(directory, r)
        assert np.array_equal(image, c_image), r["frame_md5"]
        assert masks.keys() == c_masks.keys()
        assert all(np.array_equal(masks[s], c_masks[s]) for s in masks), r["frame_md5"]
    info = {"size_hw": size_hw, "splits": splits, "frames": len(records),
            "written": written, "verified_identical": len(sample)}
    (directory / f"cache_info_{'_'.join(splits)}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
