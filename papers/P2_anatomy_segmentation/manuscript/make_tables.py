#!/usr/bin/env python3
"""LaTeX tables for the P2 manuscript, generated from the frozen analysis JSON.

Every number in the tables comes from ``results/p2_test_analysis_v1.json`` (and
the validation files named below); nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "tables"

NAMES = {
    "abdominal_wall": "Abdominal wall", "colon": "Colon", "inferior_mesenteric_artery": "IMA",
    "intestinal_veins": "Intestinal veins", "liver": "Liver", "pancreas": "Pancreas",
    "small_intestine": "Small intestine", "spleen": "Spleen", "stomach": "Stomach",
    "ureter": "Ureter", "vesicular_glands": "Vesicular glands",
}
CRITICAL = ("ureter", "inferior_mesenteric_artery", "intestinal_veins", "pancreas")
ARMS = [  # (key in analysis, label, group)
    ("A1", "A1 specialists (Kolbinger)", "Baselines"),
    ("A1N", "A1N specialists + presence negatives", "Baselines"),
    ("A2", "A2 combined, 11 decoders (Kolbinger)", "Baselines"),
    ("A3", "A3 DINOv2-B frozen", "Single model, masks only"),
    ("C1", "C1 SAM2.1 Hiera-B+ LoRA", "Single model, masks only"),
    ("A4", "A4 DINOv2-L LoRA", "Single model, masks only"),
    ("C3", "C3 A2 + our recipe", "Single model, presence supervision"),
    ("A6", "A6 A5$'$ + annotator soft targets", "Single model, presence supervision"),
    ("A5p", "\\textbf{A5$'$ DINOv2-L LoRA + presence}", "Single model, presence supervision"),
    ("C2", "C2 SurgeNetXL DINOv2-L$^\\dagger$", "Test-contaminated comparator"),
]


def f(x, d=3):
    return f"{x:.{d}f}"


def pct(x):
    return f"{100 * x:.1f}"


def ci(c, scale=1.0, d=3):
    return f"[{c[0] * scale:.{d}f}, {c[1] * scale:.{d}f}]"


def main_table(a):
    arms = a["arms"]
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Group & Arm & Critical IoU & 11-structure IoU & Hallucination (\\%) \\\\", "\\midrule"]
    last = None
    for key, label, group in ARMS:
        s = arms[key]["summary"]
        g = group if group != last else ""
        last = group
        lines.append(f"{g} & {label} & {f(s['critical_macro_iou'])} & {f(s['macro_iou'])} & "
                     f"{pct(s['hallucination_macro'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def primary_table(a, gate):
    p, q = a["primary"], gate["primary"]
    rows = [("H1: critical IoU, A5$'$ $-$ A1", p["H1_critical_iou_A5p_minus_A1"], 1, 3),
            ("H2: hallucination (pp), A4 $-$ A5$'$", p["H2_hallucination_A4_minus_A5p"], 100, 1),
            ("Non-inferiority: 11-structure IoU, A5$'$ $-$ A1", p["noninferiority_macro11_A5p_minus_A1"], 1, 3),
            ("H1 at the fixed 0.5 gate (secondary)", q["H1_critical_iou_A5p_minus_A1"], 1, 3)]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Contrast & $n$ & Difference & 95\\% CI & $p$ (Holm) \\\\", "\\midrule"]
    for label, c, sc, d in rows:
        p_holm = c.get("p_holm")
        ptxt = f"{c['p_sign_flip']:.3f}" + (f" ({p_holm:.3f})" if p_holm is not None and "fixed" not in label else "")
        lines.append(f"{label} & {c['n']} & {c['mean_diff'] * sc:+.{d}f} & {ci(c['ci95'], sc, d)} & {ptxt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def structure_table(a):
    arms = a["arms"]
    k = a["secondary_descriptive"]["kolbinger_published_test"]["A1"]
    lines = ["\\begin{tabular}{lcccccc|ccc}", "\\toprule",
             " & \\multicolumn{6}{c|}{IoU} & \\multicolumn{3}{c}{Hallucination (\\%)} \\\\",
             "Structure & K-spec & K-comb & A1 & A2 & A4 & A5$'$ & A1 & A4 & A5$'$ \\\\", "\\midrule"]
    for s, name in NAMES.items():
        iou = [arms[x]["summary"]["iou"][s] for x in ("A1", "A2", "A4", "A5p")]
        best = max(iou)
        cells = [f"\\textbf{{{f(v)}}}" if v == best else f(v) for v in iou]
        hal = [pct(arms[x]["summary"]["halluc"][s]) for x in ("A1", "A4", "A5p")]
        star = "$^*$" if s in CRITICAL else ""
        lines.append(f"{name}{star} & {f(k[s]['kolbinger_specialist'], 2)} & {f(k[s]['kolbinger_combined'], 2)} & "
                     + " & ".join(cells) + " & " + " & ".join(hal) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def rq5_table(a):
    arms, rq5 = a["arms"], a["rq5"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Arm (risk-controlled) & Critical IoU & Structures $\\le 5\\%$ & Surgery$\\times$structure $>5\\%$ & Max rate (\\%) \\\\",
             "\\midrule"]
    for key, label in (("A1_cal", "A1 specialists"), ("A1N_cal", "A1N specialists + negatives"),
                       ("A4_cal", "A4 masks only"), ("AB_nohead", "A5$'$ without presence head (area score)"),
                       ("A5p", "\\textbf{A5$'$ (presence score)}")):
        r = rq5[key]
        ok = sum(v["pooled_rate"] <= 0.05 for v in r.values())
        cells = sum(v["surgeries_above_alpha"] for v in r.values())
        total = sum(v["n_surgeries"] for v in r.values())
        mx = max(v["pooled_rate"] for v in r.values())
        lines.append(f"{label} & {f(arms[key]['summary']['critical_macro_iou'])} & {ok}/{len(r)} & "
                     f"{cells}/{total} & {pct(mx)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def ablation_table(a):
    arms = a["arms"]
    rows = [("A5$'$ (21 training surgeries)", "A5p"), ("10 training surgeries", "AB_budget10"),
            ("5 training surgeries", "AB_budget5"), ("without LoRA (frozen encoder)", "AB_nolora"),
            ("without presence head, ungated", "AB_nohead_def"),
            ("without presence head, risk-controlled (area)", "AB_nohead"),
            ("+ dense \\texttt{multilabel/} frames", "AB_multilabel"),
            ("+ annotator soft targets (A6)", "A6"), ("fixed 0.5 gate", "A5p_gate"),
            ("ungated", "A5p_ungated"), ("empty-target Dice (v1.2 A5)", "A5_v12")]
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             "Variant & Critical IoU & 11-structure IoU & Hallucination (\\%) \\\\", "\\midrule"]
    for label, key in rows:
        s = arms[key]["summary"]
        lines.append(f"{label} & {f(s['critical_macro_iou'])} & {f(s['macro_iou'])} & {pct(s['hallucination_macro'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def thin_table(a):
    arms = a["arms"]
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Structure & Arm & clDice & NSD (10 px) & HD95 (px) \\\\", "\\midrule"]
    for s in ("ureter", "inferior_mesenteric_artery", "intestinal_veins"):
        for i, key in enumerate(("A1", "A4", "A5p")):
            sm = arms[key]["summary"]
            name = NAMES[s] if i == 0 else ""
            lines.append(f"{name} & {key.replace('A5p', 'A5$' + chr(39) + '$')} & {f(sm['cldice'][s])} & "
                         f"{f(sm['nsd'][s])} & {sm['hd95'][s]:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def ceiling_table(c):
    rows = c["per_structure"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "& \\multicolumn{2}{c}{Annotator vs.\\ others} & \\multicolumn{2}{c}{Model vs.\\ published mask} \\\\",
             "Structure & majority & pairwise & A1 & A5$'$ \\\\", "\\midrule"]
    order = list(CRITICAL) + [s for s in NAMES if s not in CRITICAL]
    for i, s in enumerate(order):
        if i == len(CRITICAL):
            lines.append("\\midrule")
        r = rows[s]
        lines.append(f"{NAMES[s]} & {f(r['loo_majority'])} & {f(r['pairwise'])} & "
                     f"{f(r['model/A1'])} & {f(r['model/A5p'])} \\\\")
    lines.append("\\midrule")
    for key, label in (("critical", "Critical macro"), ("all11", "11-structure macro")):
        m = c["macro"][key]
        lines.append(f"{label} & {f(m['loo_majority'])} & {f(m['pairwise'])} & "
                     f"{f(m['model/A1'])} & {f(m['model/A5p'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def dense_table(d):
    arms = d["arms"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Arm & mIoU (7 organs) & mIoU (organs present) & Pixel acc. & Hallucination (\\%) \\\\",
             "\\midrule"]
    for key, label in (("A1", "A1 specialists"), ("A1N", "A1N specialists + negatives"),
                       ("A2", "A2 combined (Kolbinger)"), ("C1", "C1 SAM2.1 LoRA"), ("A4", "A4 DINOv2-L LoRA"),
                       ("C3", "C3 A2 + our recipe"), ("A6", "A6 soft targets"),
                       ("A5p", "\\textbf{A5$'$}"), ("A5p_ungated", "A5$'$ ungated"),
                       ("C2", "C2 SurgeNetXL$^\\dagger$")):
        s = arms[key]["summary"]
        lines.append(f"{label} & {f(s['miou_organs'])} & {f(s['miou_gt_present'])} & "
                     f"{f(s['pixel_accuracy'])} & {pct(s['scene_hallucination'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    a = json.loads((RESULTS / "p2_test_analysis_v1.json").read_text())
    gate = json.loads((RESULTS / "p2_test_analysis_gate05_secondary_v1.json").read_text())
    assert a["split"] == "test"
    OUT.mkdir(exist_ok=True)
    for name, text in (("main", main_table(a)), ("primary", primary_table(a, gate)),
                       ("structures", structure_table(a)), ("rq5", rq5_table(a)),
                       ("ablations", ablation_table(a)), ("thin", thin_table(a)),
                       ("ceiling", ceiling_table(json.loads((RESULTS / "p2_annotator_ceiling_v1.json").read_text()))),
                       ("dense", dense_table(json.loads((RESULTS / "p2_dense_scene_v2.json").read_text())))):
        (OUT / f"{name}.tex").write_text(text + "\n")
        print(f"tables/{name}.tex")


if __name__ == "__main__":
    main()
