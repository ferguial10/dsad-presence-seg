#!/usr/bin/env python3
"""Summarise the external evaluation (amendment 7): unit-level means and bootstrap CIs.

Unit = CholecSeg8k video or EndoVis18 sequence. Frames are averaged within unit
and seed, seeds within unit; contrasts are A5′ (calibrated) minus A1 and A4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from davinci.metrics.stats import bootstrap_ci


def by_unit(data: dict, op: str) -> dict:
    """{metric/key: {unit: value}} averaged over frames, then seeds."""
    acc: dict = {}
    for block in data["seeds"].values():
        per: dict = {}
        for r in block["images"]:
            for k, v in r.items():
                if k.startswith(op + "/"):
                    per.setdefault(k[len(op) + 1:], {}).setdefault(r["unit"], []).append(v)
        for k, units in per.items():
            for u, vs in units.items():
                acc.setdefault(k, {}).setdefault(u, []).append(float(np.mean(vs)))
    return {k: {u: float(np.mean(v)) for u, v in units.items()} for k, units in acc.items()}


def macro(units: dict, prefix: str) -> dict:
    keys = [k for k in units if k.startswith(prefix)]
    common = sorted(set.intersection(*(set(units[k]) for k in keys))) if keys else []
    return {u: float(np.mean([units[k][u] for k in keys])) for u in common}


def contrast(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in common])
    lo, hi = bootstrap_ci(d)
    return {"n_units": len(common), "mean_diff": float(d.mean()), "ci95": [lo, hi]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    out = {}
    for ds in ("cholecseg8k", "endovis18"):
        arms = {}
        for arm, op in (("a1", "default"), ("a1n", "default"), ("a2", "default"), ("a4", "default"),
                        ("a5p", "calibrated"), ("a5p", "default")):
            data = json.loads((args.results / f"{arm}_{ds}_external_v1.json").read_text())
            name = arm.upper().replace("A5P", "A5p") + ("_ungated" if arm == "a5p" and op == "default" else "")
            u = by_unit(data, op)
            arms[name] = {"units": u,
                          "summary": {k: float(np.mean(list(v.values()))) for k, v in u.items()}}
            for prefix in ("iou/", "halluc/", "impossible/"):
                m = macro(u, prefix)
                if m:
                    arms[name]["summary"][prefix + "macro"] = float(np.mean(list(m.values())))
                    arms[name][prefix + "macro_units"] = m
        res = {"arms": {k: v["summary"] for k, v in arms.items()}, "contrasts": {}}
        for other in ("A1", "A4", "A1N", "A2"):
            for prefix in ("iou/", "halluc/", "impossible/"):
                key = prefix + "macro_units"
                if key in arms["A5p"] and key in arms[other]:
                    res["contrasts"][f"{prefix}macro A5p - {other}"] = contrast(arms["A5p"][key], arms[other][key])
        out[ds] = res
    args.output.write_text(json.dumps(out, indent=1))
    for ds, res in out.items():
        print("==", ds)
        for name, s in res["arms"].items():
            print(f"  {name:12s} " + " ".join(f"{k}={v:.3f}" for k, v in sorted(s.items())))
        for k, c in res["contrasts"].items():
            print(f"  {k:28s} n={c['n_units']} diff={c['mean_diff']:+.3f} CI=[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
