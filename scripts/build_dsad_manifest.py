#!/usr/bin/env python3
"""Build the fused DSAD frame manifest (no pixels are decoded, only hashed)."""
import argparse
import hashlib
import json
from pathlib import Path

from davinci.data.dsad import build_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
manifest = build_manifest(args.root)
manifest["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as f:
    json.dump(manifest, f, indent=1)
print(json.dumps(manifest["summary"], indent=1))
