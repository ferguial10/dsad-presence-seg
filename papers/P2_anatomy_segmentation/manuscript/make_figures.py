#!/usr/bin/env python3
"""Figures for the P2 manuscript, computed from the frozen per-image result files.

Colours: validated categorical palette (blue A5', orange A1, aqua A4, yellow
A1N); every series is also direct-labelled because aqua and yellow are below
3:1 contrast on white.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from p2_analyze import macro_by_surgery, per_surgery  # noqa: E402

from davinci.data.dsad import CRITICAL, STRUCTURES  # noqa: E402

RES = HERE.parent / "results"
OUT = HERE / "figures"
COL = {"A5p": "#2a78d6", "A1": "#eb6834", "A4": "#1baf7a", "A1N": "#eda100", "ink": "#2b2b2b",
       "muted": "#8a8a85", "grid": "#e6e6e3"}
LABEL = {"A5p": "A5$'$ (presence)", "A1": "A1 specialists", "A4": "A4 masks only", "A1N": "A1N specialists + neg."}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "font.family": "sans-serif",
                     "axes.edgecolor": COL["muted"], "axes.linewidth": 0.6, "xtick.color": COL["ink"],
                     "ytick.color": COL["ink"], "axes.labelcolor": COL["ink"], "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/{name}.pdf")


def paired(ax, a, b, title):
    """Per-surgery critical IoU, A1 vs A5' joined by a line."""
    common = sorted(set(a) & set(b), key=lambda g: b[g] - a[g])
    for i, g in enumerate(common):
        ax.plot([0, 1], [a[g], b[g]], color=COL["grid"] if b[g] > a[g] else COL["muted"], lw=1, zorder=1)
    ax.scatter(np.zeros(len(common)), [a[g] for g in common], s=14, color=COL["A1"], zorder=2,
               edgecolor="white", linewidth=0.5)
    ax.scatter(np.ones(len(common)), [b[g] for g in common], s=14, color=COL["A5p"], zorder=2,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks([0, 1], ["A1\nspecialists", "A5$'$"])
    ax.set_xlim(-0.4, 1.4)
    better = sum(b[g] > a[g] for g in common)
    ax.set_title(f"{title}: {better} of {len(common)} surgeries improve", loc="left")
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)


def fig_paired():
    test_a1 = per_surgery(load("a1_test_native_v1.json"), "default")
    test_a5 = per_surgery(load("a5p_test_native_v1.json"), "calibrated:" + str(RES / "a5p_calibration_v1.json"))
    cv = RES / "cv"
    cv_a1 = per_surgery(json.loads((cv / "cv_a1_heldout_merged_v1.json").read_text()), "default")
    cv_a5 = per_surgery(json.loads((cv / "cv_a5p_heldout_merged_v1.json").read_text()),
                        "calibrated:" + str(cv / "cv_a5p_calibration_merged_v1.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    paired(axes[0], macro_by_surgery(test_a1, "iou", CRITICAL), macro_by_surgery(test_a5, "iou", CRITICAL),
           "Test set")
    paired(axes[1], macro_by_surgery(cv_a1, "iou", CRITICAL), macro_by_surgery(cv_a5, "iou", CRITICAL),
           "4-fold cross-validation")
    axes[0].set_ylabel("Critical-structure IoU (per surgery)")
    save(fig, "fig_paired_surgeries")


def sweep(scores, use_presence, grid):
    """Critical IoU and hallucination macro for one global threshold per value of grid."""
    crit, hall = [], []
    for lam in grid:
        iou_s, hal_s = {}, {}
        for block in scores["seeds"].values():
            for s in STRUCTURES:
                for r in block["images"]:
                    if f"area/{s}" not in r:
                        continue
                    q = r.get(f"presence/{s}") if use_presence else r[f"area/{s}"]
                    keep = q >= lam
                    if f"iou/{s}" in r:
                        iou_s.setdefault(s, []).append(r[f"iou/{s}"] if keep else 0.0)
                    else:
                        hal_s.setdefault(s, []).append(float(keep and r[f"area/{s}"] > 0.005))
        crit.append(np.mean([np.mean(iou_s[s]) for s in CRITICAL]))
        hall.append(np.mean([np.mean(v) for v in hal_s.values()]))
    return np.array(hall), np.array(crit)


def fig_tradeoff():
    """Image-pooled sweep of one global threshold on the test set (descriptive)."""
    a5, a1 = load("a5p_test_native_v1.json"), load("a1_test_native_v1.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    h, c = sweep(a5, True, np.linspace(0, 0.995, 60))
    ax.plot(100 * h, c, color=COL["A5p"], lw=2)
    ax.text(100 * h[0] + 1.8, c[0] - 0.01, "A5$'$, presence threshold", color=COL["ink"], va="center", fontsize=7)
    h1, c1 = sweep(a1, False, np.concatenate([[0], np.logspace(np.log10(0.005), 0, 60)]))
    ax.plot(100 * h1, c1, color=COL["A1"], lw=2)
    ax.text(100 * h1[0] - 0.6, c1[0] + 0.01, "A1, area threshold", color=COL["ink"], ha="right", fontsize=7)
    ax.axvline(5, color=COL["muted"], lw=0.8, ls=(0, (3, 2)))
    ax.text(5.4, 0.30, "5% target", color=COL["muted"], fontsize=7)
    ax.set_xlabel("Hallucination rate (%, macro over structures)")
    ax.set_ylabel("Critical-structure IoU")
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_tradeoff")


def fig_budget():
    a = load("p2_test_analysis_v1.json")["arms"]
    xs = [5, 10, 21]
    ys = [a[k]["summary"]["critical_macro_iou"] for k in ("AB_budget5", "AB_budget10", "A5p")]
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.plot(xs, ys, color=COL["A5p"], lw=2, marker="o", ms=5, markeredgecolor="white")
    for x, y in zip(xs, ys):
        ax.text(x + 0.6, y + 0.006, f"{y:.3f}", ha="left", va="top", fontsize=7, color=COL["ink"])
    ref = a["A1"]["summary"]["critical_macro_iou"]
    ax.axhline(ref, color=COL["A1"], lw=1.2, ls=(0, (4, 2)))
    ax.text(4.6, ref + 0.03, f"A1 specialists,\n21 surgeries ({ref:.3f})", fontsize=7, color=COL["ink"], va="bottom")
    ax.set_xlim(3.5, 24)
    ax.set_xticks(xs)
    ax.set_xlabel("Training surgeries")
    ax.set_ylabel("Critical-structure IoU (test)")
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_label_budget")


def fig_external():
    e = load("p2_external_analysis_v1.json")["cholecseg8k"]["arms"]
    names = {"ureter": "Ureter", "inferior_mesenteric_artery": "IMA", "intestinal_veins": "Intestinal veins",
             "vesicular_glands": "Vesicular glands", "spleen": "Spleen"}
    arms = ("A1", "A4", "A5p")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    y = np.arange(len(names))
    hgt = 0.26
    for i, arm in enumerate(arms):
        vals = [100 * e[arm][f"impossible/{s}"] for s in names]
        ax.barh(y + (i - 1) * (hgt + 0.02), vals, height=hgt, color=COL[arm], label=LABEL[arm])
        for yy, v in zip(y + (i - 1) * (hgt + 0.02), vals):
            ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=6, color=COL["ink"])
    ax.set_yticks(y, list(names.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Frames where the structure is claimed (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handlelength=1.2)
    ax.xaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig_external_impossible")


if __name__ == "__main__":
    fig_paired()
    fig_tradeoff()
    fig_budget()
    fig_external()
