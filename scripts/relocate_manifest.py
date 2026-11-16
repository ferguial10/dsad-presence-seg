#!/usr/bin/env python3
"""Copy a DSAD frame manifest with ``root`` pointing to a local DSAD copy.

All other fields (records, split, labels, builder hash) are kept unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--root", type=Path, required=True, help="local DSAD folder (contains abdominal_wall/, ...)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    m = json.loads(args.manifest.read_text())
    root = args.root.resolve()
    missing = [r["image"] for r in m["records"][:50] if not (root / r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of the first 50 images not found under {root}, e.g. {missing[0]}")
    m["root"] = str(root)
    args.output.write_text(json.dumps(m))
    print(f"{args.output}: {len(m['records'])} records, root={root}")


if __name__ == "__main__":
    main()
