# P2 external evaluation on CholecSeg8k and EndoVis18 — proposed amendment 7

Status: **draft, 2026-11-10**. To be adopted as amendment 7 once the author
approves it and has downloaded the dataset, and before any CholecSeg8k image is
passed through a model. Nothing here is tuned on CholecSeg8k.

## Why

DSAD is the only public dataset with these eleven structures. A reviewer will ask
whether the findings depend on one dataset, one centre and one procedure.
CholecSeg8k (Hong et al., arXiv 2012.12453; 8,080 frames from 17 Cholec80
laparoscopic cholecystectomies; 13 dense classes; CC BY-NC-SA 4.0) shares three
structures with DSAD and is a different procedure, centre, platform
(conventional laparoscopy, not robotic) and resolution. It therefore tests the
two claims that matter most under domain shift: accuracy on shared structures
and hallucination of structures that cannot be in view.

A second external domain, **EndoVis 2018 Robotic Scene Segmentation**
(already on Supercom, `~/datasets/EndoVis2018_RSS/release`), is the da Vinci
platform of this thesis: porcine robotic nephrectomy, 1280×1024 frames (the DSAD
resolution), 12 dense classes including *intestine* (class 10). It tests the
shared structure on the robotic platform under a species shift.

## Design

- **Zero-shot.** The frozen checkpoints of A1, A1N, A2, A4 and A5′ (all seeds)
  are applied without any training or tuning. C2 is excluded: SurgeNetXL was
  pretrained on Cholec80, the source of CholecSeg8k.
- **Pre-processing.** Frames (854×480) are letter-boxed with black borders to the
  5:4 DSAD aspect ratio, resized to the model input, and predictions are cropped
  back to the original frame before scoring.
- **Shared structures.** DSAD abdominal wall ↔ CholecSeg8k *Abdominal Wall*;
  liver ↔ *Liver*; the union of predicted colon, small intestine and stomach ↔
  *Gastrointestinal Tract*.
- **Accuracy.** Per-image IoU on frames where the class is present in the dense
  mask; video-level mean (unit = video, n = 17).
- **Hallucination, two kinds.**
  1. Shared structures on frames where the dense mask shows them absent.
  2. Structures that cannot be in a cholecystectomy field of view — ureter,
     inferior mesenteric artery, intestinal veins, vesicular glands, spleen —
     on every frame (pancreas is excluded because parts of it can appear near the
     duodenum). A structure is claimed when it covers more than 0.5% of the
     frame, as in DSAD.
- **Operating points.** A5′ at its DSAD-calibrated thresholds (primary) and
  ungated; comparators at their default. Whether the DSAD-calibrated guarantee
  (≤5%) still holds under domain shift is reported; we expect it may not,
  because calibration assumed exchangeable data.
- **Statistics.** Descriptive: video-level means with 10,000-draw video-level
  bootstrap 95% CIs for A5′ − A1 and A5′ − A4. No p-values.
- **Label sanity check (before any model output is seen).** CholecSeg8k ships
  colour, watershed and annotation-tool masks; we verify the class-to-colour
  mapping on 20 random frames against the paper's figure and fix the mapping in
  code before scoring.

## EndoVis18 specifics

- **Frames.** Only the 15 development sequences (train and validation of the
  thesis-wide EndoVis18 manifests; 2,235 frames). The 4 official test sequences
  are a protected test set of another thesis paper and are not read.
- **Shared structure.** EndoVis18 *intestine* ↔ union of predicted colon and
  small intestine. Accuracy: per-image IoU on frames with intestine; unit =
  sequence (n = 15).
- **Hallucination.** Intestine predicted (> 0.5% of the frame) on frames whose
  dense mask has no intestine. No "impossible structure" analysis: in porcine
  nephrectomy too many DSAD structures may legitimately be in view.
- **Pre-processing.** None beyond the DSAD pipeline (same resolution).
- Instruments are labelled in EndoVis18 but not in DSAD, where they are
  unlabelled background; instrument pixels are therefore scored as background.

## Cost and access

Inference only: about 3 GPU-hours for all arms, seeds and both datasets. The dataset is hosted
on Kaggle and requires an account and acceptance of its terms, which the author
must do (access conditions are accepted by the author only).
