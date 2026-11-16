#!/usr/bin/env python3
"""Freeze the P2 candidates before test access (DESIGN.md §3, amendment 4).

Writes ``CANDIDATES_FROZEN.json``: SHA-256 of every ``best.pt`` that may be
scored on test, of every calibration file, and the git commit. ``p2_score.py``
refuses test scoring for any run that is missing or changed. The file is
written once; it is never overwritten.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from p2_score import LOCK, sha256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--calibrations", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    if LOCK.exists():
        raise FileExistsError(LOCK)
    entries = {str(run): sha256(run / "best.pt") for run in args.runs}
    for run in args.runs:
        assert json.loads((run / "result.json").read_text())["test_accessed"] is False, run
    lock = entries | {"_calibrations": {str(c): sha256(c) for c in args.calibrations},
                      "_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "_frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
                      "_n_runs": len(entries)}
    with LOCK.open("x") as f:
        json.dump(lock, f, indent=1)
    print(f"froze {len(entries)} runs and {len(args.calibrations)} calibration files")


if __name__ == "__main__":
    main()
