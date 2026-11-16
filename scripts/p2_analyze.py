#!/usr/bin/env python3
"""P2 endpoints from p2_score outputs (DESIGN.md §5-6, amendments 2-4).

Each arm is a scores JSON plus an operating point:
  ``default``   no gating; hallucination = predicted area > 0.5% (comparators)
  ``gate0.5``   presence probability >= 0.5 (fixed gate of amendment 1.5)
  ``ungated``   presence head ignored
  ``calibrated:<calibration.json>``  per-seed, per-structure risk-controlled threshold

Metrics are averaged per surgery within each seed, then over seeds within
surgery; contrasts use the surgery as unit (exact sign-flip, bootstrap CI,
Holm over H1 and H2, non-inferiority margin 0.03).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from davinci.data.dsad import CRITICAL, STRUCTURES
from davinci.data.dsad_reference import KOLBINGER_IOU
from davinci.metrics import stats
from davinci.metrics.dsad_eval import HALLUCINATION_FRACTION
from p2_score import LOCK, sha256

THIN = ("ureter", "inferior_mesenteric_artery", "intestinal_veins")
ALPHA = 0.05


def _threshold(op, cal, seed, s):
    if op in ("default", "ungated"):
        return None
    if op == "gate0.5":
        return 0.5
    lam = cal["seeds"][seed][s]["lambda"]
    return math.inf if lam is None else lam


def per_surgery(scores: dict, op: str) -> dict:
    """{metric: {structure: {surgery: value}}} averaged over seeds."""
    cal = json.loads(Path(op.split(":", 1)[1]).read_text()) if op.startswith("calibrated:") else None
    op = op.split(":", 1)[0]
    acc: dict = {}
    for seed, block in scores["seeds"].items():
        for s in block["structures"]:
            lam = _threshold(op, cal, seed, s)
            by: dict = {}
            for r in block["images"]:
                if f"area/{s}" not in r:
                    continue
                area = r[f"area/{s}"]
                if lam is None:
                    keep = True
                elif f"presence/{s}" in r:
                    keep = r[f"presence/{s}"] >= lam
                else:
                    keep = area >= lam
                claimed = keep and area > HALLUCINATION_FRACTION
                d = by.setdefault(r["surgery"], {})
                if f"iou/{s}" in r:
                    d.setdefault("iou", []).append(r[f"iou/{s}"] if keep else 0.0)
                    d.setdefault("dice", []).append(r[f"dice/{s}"] if keep else 0.0)
                    d.setdefault("miss", []).append(float(not claimed))
                    if s in THIN:
                        d.setdefault("cldice", []).append(r[f"cldice/{s}"] if keep else 0.0)
                        d.setdefault("nsd", []).append(r[f"nsd/{s}"] if keep else 0.0)
                        hd = r[f"hd95/{s}"] if keep else float("nan")
                        if not math.isnan(hd):
                            d.setdefault("hd95", []).append(hd)
                else:
                    d.setdefault("halluc", []).append(float(claimed))
            for surgery, metrics in by.items():
                for m, v in metrics.items():
                    acc.setdefault(m, {}).setdefault(s, {}).setdefault(surgery, []).append(float(np.mean(v)))
    return {m: {s: {g: float(np.mean(v)) for g, v in d.items()} for s, d in ms.items()}
            for m, ms in acc.items()}


def macro_by_surgery(ps: dict, metric: str, structures) -> dict:
    surgeries = sorted({g for s in structures for g in ps.get(metric, {}).get(s, {})})
    return {g: float(np.mean([ps[metric][s][g] for s in structures if g in ps[metric].get(s, {})]))
            for g in surgeries}


def summary(ps: dict) -> dict:
    out = {}
    for m, ms in ps.items():
        out[m] = {s: float(np.mean(list(d.values()))) for s, d in ms.items()}
    crit = [s for s in CRITICAL if s in out.get("iou", {})]
    out["critical_macro_iou"] = float(np.mean([out["iou"][s] for s in crit]))
    out["macro_iou"] = float(np.mean(list(out["iou"].values())))
    out["hallucination_macro"] = float(np.mean(list(out["halluc"].values())))
    return out


def contrast(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    return stats.paired_contrast([a[g] for g in common], [b[g] for g in common]) | {"surgeries": common}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "SCORES", "OP"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arms, split = {}, None
    for name, path, op in args.arm:
        scores = json.loads(Path(path).read_text())
        split = split or scores["split"]
        assert scores["split"] == split, "all arms must come from the same split"
        if scores["split"] == "test" and op.startswith("calibrated:"):
            cal_path = op.split(":", 1)[1]
            frozen = json.loads(LOCK.read_text())["_calibrations"]
            assert frozen.get(cal_path) == sha256(Path(cal_path)), f"{cal_path} not frozen or changed"
        ps = per_surgery(scores, op)
        arms[name] = {"op": op, "per_surgery": ps, "summary": summary(ps)}
    out = {"split": split, "arms": {k: {"op": v["op"], "summary": v["summary"]} for k, v in arms.items()}}

    def crit(name):
        return macro_by_surgery(arms[name]["per_surgery"], "iou", CRITICAL)

    def all11(name):
        return macro_by_surgery(arms[name]["per_surgery"], "iou", STRUCTURES)

    def hal(name):
        return macro_by_surgery(arms[name]["per_surgery"], "halluc", STRUCTURES)

    if {"A5p", "A1", "A4"} <= arms.keys():
        h1, h2 = contrast(crit("A5p"), crit("A1")), contrast(hal("A4"), hal("A5p"))
        adj = stats.holm({"H1": h1["p_sign_flip"], "H2": h2["p_sign_flip"]})
        ni = contrast(all11("A5p"), all11("A1"))
        out["primary"] = {
            "H1_critical_iou_A5p_minus_A1": h1 | {"p_holm": adj["H1"]},
            "H2_hallucination_A4_minus_A5p": h2 | {"p_holm": adj["H2"]},
            "noninferiority_macro11_A5p_minus_A1": ni | {"margin": 0.03,
                                                       "non_inferior": stats.non_inferior(ni, 0.03)},
        }
    secondary = {}
    for other in ("A1", "A1N", "A2", "C3", "A4", "C1", "C2"):
        if {"A5p", other} <= arms.keys():
            secondary[f"critical_iou_A5p_minus_{other}"] = contrast(crit("A5p"), crit(other))
            secondary[f"hallucination_{other}_minus_A5p"] = contrast(hal(other), hal("A5p"))
    if split == "test":
        secondary["kolbinger_published_test"] = {
            name: {s: {"ours": arms[name]["summary"]["iou"][s],
                       "kolbinger_specialist": KOLBINGER_IOU[("deeplabv3", "specialist", "test")][s],
                       "kolbinger_combined": KOLBINGER_IOU[("deeplabv3", "combined", "test")][s]}
                   for s in STRUCTURES}
            for name in arms if name in ("A1", "A2", "A5p")}
    out["secondary_descriptive"] = secondary
    rq5 = {}
    for name, a in arms.items():
        if not a["op"].startswith("calibrated"):
            continue
        h = a["per_surgery"]["halluc"]
        rq5[name] = {s: {"pooled_rate": float(np.mean(list(h[s].values()))),
                         "surgeries_above_alpha": int(sum(v > ALPHA for v in h[s].values())),
                         "n_surgeries": len(h[s]),
                         "miss_rate": a["summary"]["miss"][s], "iou": a["summary"]["iou"][s]}
                     for s in h}
    out["rq5"] = rq5
    args.output.write_text(json.dumps(out, indent=1))
    for name, a in arms.items():
        sm = a["summary"]
        print(f"{name:14s} {a['op'][:10]:10s} crit={sm['critical_macro_iou']:.3f} "
              f"macro11={sm['macro_iou']:.3f} halluc={sm['hallucination_macro']:.3f}")
    for k, v in out.get("primary", {}).items():
        print(k, {x: v[x] for x in v if x in ("mean_diff", "ci95", "p_sign_flip", "p_holm", "non_inferior")})


if __name__ == "__main__":
    main()
