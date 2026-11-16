#!/usr/bin/env python3
"""Check downloaded checkpoints against the hashes frozen before the test evaluation.

Usage: python scripts/verify_checkpoints.py runs/p2
Every ``<run>/best.pt`` found under the folder is compared with
``papers/P2_anatomy_segmentation/CANDIDATES_FROZEN.json`` (key ``runs/p2/<run>``).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "papers/P2_anatomy_segmentation/CANDIDATES_FROZEN.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/p2")
    lock = json.loads(LOCK.read_text())
    bad = 0
    found = sorted(folder.glob("*/best.pt"))
    for ck in found:
        key = f"runs/p2/{ck.parent.name}"
        ok = lock.get(key) == sha256(ck)
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {key}")
    print(f"{len(found) - bad}/{len(found)} checkpoints match the frozen hashes")
    sys.exit(1 if bad or not found else 0)


if __name__ == "__main__":
    main()
