# P2 design — one model for all structures in robotic rectal surgery

**Status:** FROZEN v1.0, 2026-10-29 (git tag `p2-protocol-v1`), approved by the
author. Amendment 1 (tag `p2-protocol-v1.1`, 2026-10-30) in §10. Frozen before any model was trained on DSAD and before any DSAD test
pixel was read. Changes after this point are amendments: dated, justified, and
listed in §10 without consulting test results.

**Working title:** *One encoder, eleven structures: learning critical anatomy
from partially annotated robotic surgery video with foundation features and
image-level presence labels.*

## 1. Problem and gap

In robot-assisted rectal resection the ureter, the inferior mesenteric artery
(IMA) and the intestinal veins are the structures whose injury matters most.
They are also where models fail. On DSAD (Carstens et al., Sci Data 2023) the
reference models of Kolbinger et al. (Int J Surg 2023) reach test IoU 0.36–0.46
on the ureter, 0.41–0.46 on the IMA, 0.44–0.52 on the veins and 0.28–0.37 on the
pancreas, with **one network per structure**. Their single combined network is
worse on exactly these structures (ureter 0.29, IMA 0.40, veins 0.35,
pancreas 0.23). Tomar et al. (IEEE BigData 2025, arXiv 2607.29509) use
class-specific decoders and cross-domain pretraining (Dice 0.624), without
foundation encoders or image-level labels.

DSAD carries supervision that has not been exploited:

1. **Image-level presence labels** (`weak_labels.csv`) for every frame. These
   say which *other* structures are absent, so they give whole-image negatives
   for structures that the frame's mask does not annotate.
2. **Three independent annotators** per mask, which allows soft targets and
   calibration against annotator disagreement.
3. **1,434 frames annotated in 2–3 structure folders**, which can be fused.

Our probe (job 22093) shows that the published mask equals the majority vote of
the three annotators in 99.6% of pixels, that no frame is shared across
surgeries, and that the presence columns follow alphabetical structure order
(as documented in the dataset article; some rows carry a 12th padding value).
The `vesicular_glands` folder's presence file marks only that padding column, so
in that folder the presence of other structures is treated as unknown.

## 2. Questions and hypotheses

- **RQ1 (primary).** Can one network trained on all partially annotated DSAD
  frames match or beat eleven structure-specific specialists on the critical
  structures?
  *H1:* the final single model improves the critical-structure macro IoU
  (ureter, IMA, intestinal veins, pancreas) over our reproduced specialist
  ensemble.
- **RQ2.** Do image-level presence labels reduce hallucinated structures?
  *H2:* adding whole-image negatives from presence labels lowers the
  hallucination rate (defined in §5) without lowering IoU.
- **RQ3 (secondary).** Do frozen or LoRA-adapted DINOv2 features help most on
  the thin structures and in the low-label regime (5, 10, 21 training
  surgeries)?
- **RQ4 (secondary).** Do soft targets from annotator disagreement improve
  calibration (ECE against the annotator mean) and boundary agreement?

## 3. Data and splits

- **DSAD** (CC BY 4.0, MD5-verified). Split of Kolbinger et al. adopted
  unchanged: train 21 surgeries, validation 3 (3, 21, 26), test 8 (2, 7, 11, 13,
  14, 18, 20, 32). The test surgeries are not read until the protocol is frozen
  and the candidates are fixed.
- **Frame identity** = MD5 of the image bytes, within its surgery. Annotations
  of the same frame from different folders are fused into one record.
- **Per frame and structure**, the label state is one of:
  `mask` (pixel labels available), `absent` (presence label 0), `unknown`
  (present but no mask, or no presence label).
- `multilabel/` (1,430 frames, 7 organs, dense) is used as an extra
  training source only in an ablation, because it adds dense labels and would
  otherwise confound H1.
- **EndoVis18** (second da Vinci dataset, official split) is used only to test
  whether the recipe transfers (secondary, no claims of state of the art).

## 4. Arms

All arms: native frames resized to 640×512 (CNN) or 644×518 (ViT/14), same
augmentation (scale 0.75–1.25, flip, rotation ±15°, colour jitter), AdamW,
cosine schedule with 3 warm-up epochs and BCE + soft Dice loss (A1 instead
follows Kolbinger: binary cross-entropy, learning rate ×0.9 every 10 epochs), 100 epochs, fp16 on V100, three seeds (11, 23, 37),
selection by validation critical-structure macro IoU (specialists: their own
structure's IoU), computed at training resolution. Final scoring upsamples
logits bilinearly to the native 1280×1024 and thresholds at 0.5.

| Id | Encoder | Heads / loss | Role |
| --- | --- | --- | --- |
| A1 | DeepLabV3-R50 (COCO, as Kolbinger) | 11 separate binary models, each on its own folder | reproduction and primary comparator |
| A2 | DeepLabV3-R50 | one model, 11 sigmoid heads, partial-label loss (masks only) | Kolbinger combined, reproduced (architecture and recipe: Amendment 1.3) |
| A3 | DINOv2 ViT-B/14, frozen | as A2, light DPT decoder | foundation features |
| A4 | DINOv2 ViT-L/14, LoRA r=16 | as A2 | adapted foundation features |
| A5 | best of A3/A4 by validation | + presence-label whole-image negatives + image-level presence head | **final candidate** |
| A6 | A5 | + soft targets from 3 annotators | RQ4 |
| Ablations | A5 | without LoRA, without presence head, + `multilabel/`, label budgets 5/10/21 surgeries | secondary |
| C1 | SAM2 Hiera-B+ image encoder (Apache-2.0), LoRA | as A4 | foundation comparator (segmentation-pretrained) |
| C2 | Surgical-domain ViT-B (EndoViT or equivalent, if licence permits) | as A4 | in-domain pretraining comparator |
| C3 | DeepLabV3-R50, A5 losses | as A5 | isolates the loss contribution from the encoder (A2 architecture: Amendment 1.3) |

Arms A3/A4/C1/C2 are compared on validation only; the encoder of A5 is the one
with the highest validation critical macro IoU (3-seed mean; ties → fewer
trainable parameters). C1–C3 are reported on test as secondary comparators.

The partial-label loss is BCE + soft Dice per structure, applied only where the
label state is `mask` (all pixels) or `absent` (all pixels negative, A5 on);
`unknown` contributes nothing. Pixels annotated as another structure are also
negatives (Kolbinger's rule), except for pairs that can overlap in the image
(vessels over mesentery/small intestine); the overlap list is fixed from
training masks before any model is trained.

Overlap pairs (exempt from the other-mask negative rule) are fixed by a
rule, not by hand: a pair is exempt if, in training frames where both carry a
mask, the intersection covers ≥1% of the smaller mask in ≥5% of such frames.

Acceptance check for the pipeline before any new method: A1 must reproduce
Kolbinger's DeepLabv3 structure-specific IoU within ±0.05 per structure on the
**validation** surgeries, comparing against their reported test values only as
a sanity range. A failed check stops the study until explained. (Reference
corrected to Kolbinger's validation values in Amendment 1.1.)

## 5. Endpoints

- **Primary (H1):** critical-structure macro IoU = mean over {ureter, IMA,
  intestinal veins, pancreas} of the surgery-level mean IoU, where IoU is
  computed per test image on the images where that structure has a mask.
- **Primary (H2):** hallucination rate = fraction of test images with
  presence label 0 for a structure in which the model predicts more than 0.5%
  of the image as that structure; macro-averaged over structures, surgery-level.
- **Secondary:** per-structure IoU and Dice (image mean, for direct comparison
  with Kolbinger), 11-structure macro IoU, clDice, HD95 and normalised surface
  Dice (5 px at 640×512) for ureter and vessels, ECE against the annotator mean,
  label-efficiency curve, inference latency on one V100.
- **Dense scene evaluation (secondary):** on the `multilabel/` frames of the
  test surgeries, which have masks for all 7 organs, the single model is scored
  as a full scene parser (argmax over organs + background) and the 11
  specialists are combined by the same rule. Arms trained with `multilabel/`
  frames are excluded from this comparison.
- **Robustness to the split (secondary):** A1 and A5 are rerun in 4-fold
  surgery-level cross-validation over all 32 surgeries (folds fixed by a seeded
  permutation of surgery ids, seed 2027), one seed per fold; pooled
  out-of-fold critical macro IoU with surgery-level bootstrap CI.
- **Annotator ceiling:** inter-annotator IoU (each annotator vs. the other two's
  majority) on the test frames, reported next to model IoU per structure.

## 6. Statistics

- Unit = test surgery (n = 8). Seeds are averaged within surgery first.
- Paired contrasts: H1 = A5 − A1 on the critical macro IoU; H2 = A5 − (A5
  without presence negatives) on hallucination rate.
- Exact sign-flip test over the 2^8 = 256 assignments and a 10,000-draw
  surgery-level percentile bootstrap for the 95% CI; Holm across the two
  primary contrasts.
- Non-inferiority for the 11-structure macro IoU of A5 versus A1 with margin
  0.03.
- Everything else is descriptive, with intervals and no p-values.

## 7. Success and failure

- **Success:** H1 superiority (Holm-adjusted p < 0.05, CI excluding 0) and
  11-structure non-inferiority.
- **Partial:** H2 alone, or H1 only on validation → report as a benchmark of
  partial-label foundation models on DSAD with a negative primary result.
- **Kill or redirect:** if A1 fails the reproduction check, or if no
  foundation arm beats A2 on validation, the paper becomes a reproducibility
  and benchmark study (still publishable in IJCARS) and the method effort moves
  to temporal propagation (P4).

## 8. Compute and timeline

About 33 specialist runs (A1) + 3×(A2–A6) + ablations ≈ 80 runs of 0.5–2 h on
one V100 each, about 100 GPU-hours. The `sc-gpu` queue is shared (19 jobs of
another user were pending on 2026-10-29), so runs are packed as arrays of
single-GPU jobs.

Compute rises to ≈ 180 GPU-hours with C1–C3 and the cross-validation.

| Week | Work |
| --- | --- |
| 1 | data manifest, fused labels, tests; A1 on validation (reproduction check) |
| 2 | A2–A4; freeze protocol and tag |
| 3 | A5–A6 and ablations on validation; fix candidates |
| 4 | single test evaluation; statistics; figures |
| 5–6 | manuscript (target MICCAI 2027, deadline ~March 2027; fallback IJCARS) |

## 9. Threats to validity

- Hyperparameters other than those stated are fixed before training: AdamW
  (β = 0.9, 0.999), weight decay 0.01, batch 8, learning rate 1e-4 (CNN) and
  1e-4 decoder / 5e-5 LoRA (ViT), gradient clipping 1.0, validation every 2
  epochs, best validation checkpoint kept.

- Eight test surgeries: wide intervals; we report them and do not claim
  clinical utility.
- Kolbinger's split was chosen to balance frame counts, not at random; we keep
  it for comparability.
- Presence labels may contain errors; their accuracy is spot-checked against
  frames annotated in several folders before training.
- DINOv2 was pretrained on natural images (LVD-142M); no surgical-specific
  foundation encoder is used in the primary arms, to keep the comparison
  clean. A surgical foundation encoder can be added as a secondary arm if its
  licence permits.

## 10. Amendments

### Amendment 1 — 2026-10-30 (tag `p2-protocol-v1.1`)

Made after A1 was trained and scored on validation, before any other arm was
trained, and with the test surgeries still unread. Approved by the author.

**1.1 Reproduction reference.** §4 compared A1 validation IoU with
Kolbinger's *test* IoU, which mixes two sets of surgeries. Kolbinger et al.
also report validation IoU for both their structure-specific and combined
DeepLabv3 models (Supplementary Tables 1b and 2b, Supplemental Digital
Content 1), which we had not located at freezing time. The check now uses
those values: same split, same metric (mean image IoU over validation images
that carry the structure) and the same selection (best validation epoch). The
tolerance is unchanged (±0.05 per structure). The published values are stored
in `src/davinci/data/dsad_reference.py`.

**1.2 Outcome for A1 and decision.** Jobs 22109 (training) and 22213
(per-surgery rescoring); `results/a1_reproduction_check_v1.json`.

| Structure | A1 (3 seeds) | Seed SD | Kolbinger val | Diff | Surgery 3 / 21 / 26 |
| --- | --- | --- | --- | --- | --- |
| Abdominal wall | 0.850 | 0.004 | 0.85 | −0.000 | 0.82 / 0.88 / 0.86 |
| Colon | 0.774 | 0.004 | 0.74 | +0.034 | 0.76 / 0.79 / 0.76 |
| IMA | 0.459 | 0.013 | 0.49 | −0.031 | 0.45 / 0.58 / 0.35 |
| Intestinal veins | 0.559 | 0.009 | 0.53 | +0.029 | 0.44 / 0.70 / 0.54 |
| Liver | 0.662 | 0.052 | 0.66 | +0.002 | 0.67 / 0.65 / 0.67 |
| Pancreas | 0.581 | 0.004 | 0.58 | +0.001 | 0.55 / 0.61 / 0.54 |
| Small intestine | 0.834 | 0.005 | 0.81 | +0.024 | 0.83 / 0.87 / 0.81 |
| Spleen | 0.682 | 0.013 | 0.63 | **+0.052** | 0.49 / 0.90 / 0.70 |
| Stomach | 0.518 | 0.005 | 0.51 | +0.008 | 0.57 / 0.63 / 0.39 |
| Ureter | 0.281 | 0.022 | 0.34 | **−0.059** | 0.37 / 0.27 / 0.19 |
| Vesicular glands | 0.352 | 0.011 | 0.32 | +0.032 | 0.35 / 0.36 / 0.35 |

Nine of eleven structures are within ±0.05; the mean signed difference is
+0.008. Spleen (+0.052) and ureter (−0.059) are outside by 0.002 and 0.009,
in opposite directions. Both are small compared with the spread between the
three validation surgeries (spleen 0.49–0.90, ureter 0.19–0.37) and with the
published image-level SD (0.23 and 0.25). Known differences from Kolbinger's
recipe: we add horizontal flips and saturation/hue jitter (they report scaling,
rotation, brightness and contrast), and our checkpoint selection uses the
surgery-level mean. **Decision:** under the strict rule the check fails on two
structures; we consider this explained by the reasons above and accept A1 as
the reproduction. A1 is not retrained or tuned. Because A1's ureter is below
the published value, the paper reports, next to H1, every structure of A5
against Kolbinger's published specialist test values (descriptive), so that
a weak comparator cannot inflate H1.

**1.3 A2 architecture and recipe.** Kolbinger's combined model is a shared
COCO ResNet50 backbone with **one DeepLabv3 decoder per structure**, trained
with the specialist recipe. A2 now implements exactly that
(`deeplabv3_r50_coco_multihead`, BCE, ×0.9 every 10 epochs, Kolbinger's
other-mask negatives, no presence negatives) and gets its own reproduction
check against Supplementary Table 2b with the same rule. C3 uses the A2
architecture with the A5 losses and schedule (BCE + soft Dice, cosine,
presence negatives), so that the chain A2 → C3 → A5 separates the loss
contribution from the encoder contribution.

**1.4 Data facts that correct §1 and fix the overlap list.** The 1,434
frames that appear in more than one folder are almost all `multilabel/`
duplicates of a structure-folder frame; only 35 frames (24 in train) carry
masks of two structures, and in none of them do the masks overlap. The frozen
overlap rule therefore yields **no exempt pair** (`results/overlap_pairs_v1.json`),
and Kolbinger's other-mask negatives act on almost no pixels: in the combined
model each decoder effectively learns only from its own ~700 frames. In
contrast, presence labels mark each structure absent in 3,300–7,100 training
frames and present-without-mask in 100–3,100 frames
(`dsad_frame_manifest_v3.json`, which adds the image-level presence to v2;
records are otherwise identical). Presence audit (§9): in train/val, 9 of
9,039 presence rows contradict a non-empty mask of the same frame (0.1%;
`results/presence_audit_v1.json`).

**1.5 Presence head (A5) specified.** A linear layer (dropout 0.1) on globally
pooled encoder features predicts one presence logit per structure, trained with
BCE (weight 0.5 relative to the segmentation loss) on every frame whose
presence label is known, including present-without-mask frames as positives.
At inference a structure's mask is emptied when its presence probability is
below 0.5. Selection and all endpoints use the gated output; the ungated
output is reported for the "without presence head" ablation.

**1.6 C2 encoder and test contamination.** Both public surgical foundation
encoders we considered include all 32 DSAD surgeries, test surgeries
included, in their self-supervised pretraining data: EndoViT (Endo700k,
Batić et al., IJCARS 2024, Table 1: DSAD, 13,195 images) and SurgeNetXL
(Jaspers et al., arXiv 2501.09436, Table 1: DSAD, 14,623 frames). C2 is
therefore SurgeNetXL-pretrained DINOv2 (de Jong et al., MIDL 2026; same
ViT-B/14 and ViT-L/14 architectures as A3/A4, weights CC BY-NC-SA, not
redistributed) and is reported as a **test-contaminated comparator**: it is
excluded from the A5 encoder selection and from every claim, and its results
are labelled as such. C1 is SAM2.1 Hiera-B+ (Apache-2.0, SA-1B/SA-V), whose
pretraining data do not include DSAD.

**1.7 Engineering (no effect on results).** Inputs are read from a lossless
cache of the resized frames and masks (bit-identical to on-the-fly resizing,
checked on 200 frames per size). Training writes resumable checkpoints and is
requeued when it reaches the walltime; resumption restores model, optimiser,
scheduler, scaler and RNG states.

**1.8 Training budget and arm specifications.** Measured one-epoch cost on a
V100 (jobs 22214, 22215, 22220): single-head DeepLabv3 377 s, A2 875 s,
DINOv2-B LoRA 417 s, DINOv2-L LoRA 939 s, SAM2 Hiera-B+ LoRA 361 s. One
hundred epochs for every arm would need about 900 GPU-hours on four shared
GPUs. Before any of these arms is trained, the budget is fixed as follows:

- A1: 100 epochs (done, as Kolbinger).
- A2: 100 epochs, Kolbinger recipe (reproduction).
- All other single-model arms (A3–A6, C1–C3, ablations): **50 epochs**,
  cosine schedule with 3 warm-up epochs, BCE + soft Dice. The A2 → C3 contrast
  is therefore a change of the whole training recipe (loss, schedule, epochs,
  presence supervision) on a fixed architecture, and C3 → A5 a change of
  encoder only, at an identical recipe.
- Effective batch 8 everywhere; ViT-L arms use micro-batches of 4 with two
  accumulation steps.
- A3, A4, C1 and C2 use masks only (no presence negatives, no presence head),
  like A2. A5 adds presence negatives and the presence head (1.5) to the
  selected encoder; the H2 contrast arm is A5 without presence negatives and
  without the presence head.
- Encoder candidates for A5: A3, A4, C1 (C2 excluded, 1.6); 3-seed mean of
  validation critical macro IoU, ties (difference < 0.005) → fewer trainable
  parameters (as recorded in each run's `result.json`).
- Ablations and the label-budget curve run with one seed (11); they are
  descriptive. Cross-validation runs one seed per fold, as in §5.
- Convergence: if, for an arm, at least two of three seeds select one of the
  last five epochs, this is reported as a limitation; budgets are not
  extended after seeing results.

### Amendment 2 — 2026-10-30 (tag `p2-protocol-v1.2`)

Made while A3, A4 and C1 were training, before any single-model result was
seen and with the test surgeries unread. Approved by the author.

**2.1 Related work found after freezing, and revised claims.**
- Jenke et al. (IJCARS 2024, "One model to use them all") train one DeepLabv3
  on DSAD with mutual exclusion (positive masks of other classes as negatives,
  unknown background masked) and beat an ensemble of binary models (+4.4%
  Dice). They use **six large organs only** (no vessels, no ureter, no
  vesicular glands, no spleen) and **do not use presence labels**, which they
  name as future work.
- Kolbinger et al. (CVPRW 2024) add class-negative images sampled from the
  presence labels (1:1) to **binary specialists** and report fewer false
  positives and higher precision and specificity.
- Conformal false-positive control exists for pixel masks (Mossina &
  Friedrich, arXiv 2511.15406, polyps) and instance FDR (Dai et al., arXiv
  2504.04482); we found no structure-level presence guarantee for surgical
  anatomy.

Consequently the paper does **not** claim that single models or presence
negatives are new. The claimed contributions are: (i) a single model for all
eleven DSAD structures, including the critical thin ones that Jenke et al.
excluded, with foundation encoders under partial labels; (ii) image-level
presence supervision (negatives, weak positives and a presence head) in a
single multi-structure model; (iii) the hallucination analysis of
specialists; (iv) the pretraining-contamination audit (1.6); (v)
distribution-free control of structure hallucination (2.3). H1 and H2 and
their tests are unchanged. A2 with other-mask negatives is the
Kolbinger/Jenke mutual-exclusion baseline.

**2.2 New comparator A1N (secondary).** Specialists as A1 (same recipe, 100
epochs, three seeds) plus class-negative images as in Kolbinger et al. CVPRW
2024: for each structure, a fixed random set (seed = run seed) of training
frames whose presence label is 0, as many as that structure's mask frames,
every pixel a negative. A1N is scored like A1. The contrast A5 − A1N on the
critical macro IoU and on the hallucination rate is reported with surgery-level
bootstrap CIs, without p-values; it separates the single-model effect from the
effect of presence negatives. A1N runs after the arms already queued.

**2.3 RQ5 — risk-controlled presence (secondary).** For structure s and image
x, a presence score q_s(x) is the presence-head probability (arms with a head)
or the predicted area fraction (other arms). The structure is claimed present,
and its mask kept, when q_s(x) ≥ λ_s (and, for area scores, the area exceeds
0.5%). Risk = hallucination rate on images whose presence label for s is 0.
λ_s is calibrated on the **validation** images by Learn-then-Test
fixed-sequence testing from the strictest threshold down, with the
Hoeffding–Bentkus p-value, α = 0.05, δ = 0.10; grids: presence probability 0
to 1 in steps of 0.005 plus 1.0001 (abstain); area fraction 200 log-spaced
values from 0.005 to 1 plus 1.0001. If no threshold is certified the model
always abstains for s. Implemented in `src/davinci/metrics/risk_control.py`
(unit-tested for coverage on synthetic data).

Test reporting, per structure and arm (A5 primary; A1, A1N and the selected
encoder without presence supervision descriptive): pooled and surgery-level
hallucination rate at λ_s against α; miss rate on mask images; IoU with
abstentions counted as 0; and the same quantities at the default operating
point (A5: presence probability 0.5; others: area 0.5%). Each structure is
calibrated separately (no family-wise correction). The guarantee assumes
calibration and test frames are exchangeable, which frames from different
surgeries need not be; we therefore report the number of test surgeries in
which the rate exceeds α and treat the result as an empirical test of the
guarantee, not a proof.

**2.4 Engineering.** `scripts/p2_score.py` stores raw (ungated) per-image
values and presence scores; gating, hallucination and calibrated thresholds
are computed by the analysis script from those values.

### Amendment 3 — 2026-11-02 (tag `p2-protocol-v1.3`)

Made after A3, A4, C1 and A5 were scored on validation, with the test surgeries
unread. Approved by the author.

**3.1 Loss defect.** Under the v1.2 loss, soft Dice was averaged over every
(image, structure) pair with any loss weight, including structures whose
presence label is 0 (empty targets). For an empty target the soft Dice loss is
1 − ε/(Σp + ε) with ε = 1, i.e. ≈0.9 for 10 predicted pixels and ≈0.99 for 100,
so it rewards predicting nothing; with presence negatives, 6–9 of the 11
channels of a typical frame are empty, so these terms dominate the loss. On
validation A5 removed hallucination almost completely (e.g. abdominal wall
42% → 1%, liver 21% → 0% relative to A4) but under-segmented (critical macro
IoU 0.385 vs 0.506 for A4); the ungated output was equally low (0.397), so the
presence gate is not the cause. Arms without presence negatives (A1–A4, C1,
C2) have almost no empty-target channels and are unaffected.

**3.2 Correction.** Soft Dice is computed only over channels whose label state
is `mask`; absent-structure pixels enter the BCE term only (`--dice-on mask`).
This is the only change. **A5′** (A5 with the corrected loss, three seeds,
otherwise identical) becomes the primary candidate for H1 and H2 and is the
arm called "A5" in the endpoints from now on. C3 and A6 use the corrected loss.

**3.3 Discipline.** This is the single permitted correction of the A5 recipe.
A5′ remains the primary candidate whatever its validation score, including if
it stays below A4; A4 − A1 is then reported as a secondary result. The v1.2 A5
runs are kept and reported as an ablation ("presence negatives with
empty-target Dice").

### Amendment 4 — 2026-11-04 (tag `p2-protocol-v1.4`)

Made after A5′ was scored on validation, with the test surgeries unread.
Approved by the author.

**4.1 Observation.** On validation (3 seeds), A5′ with the fixed presence gate
at 0.5 reached a critical macro IoU of 0.474 and a hallucination rate of 2.4%;
the same networks without the gate reached 0.524 and 3.6%. The fixed gate
removes true positives of thin structures (IMA 0.522 → 0.428, ureter 0.340 →
0.291) for a small reduction of hallucination. The value 0.5 was an arbitrary
default fixed in amendment 1.5.

**4.2 Change of the primary operating point.** The primary operating point of
A5′ for H1 and H2 becomes the **risk-controlled presence threshold** of RQ5
(amendment 2.3): per seed and structure, the smallest threshold λ_s whose
validation hallucination rate is certified ≤ α = 0.05 with δ = 0.10
(Learn-then-Test, Hoeffding–Bentkus), computed from native-resolution
validation predictions by `scripts/p2_calibrate.py`. λ_s = 0 (no gating) is
allowed. The thresholds are written to `results/a5p_calibration_v1.json` and
fixed before any test access; their SHA-256 is added to
`CANDIDATES_FROZEN.json`. The fixed 0.5 gate and the ungated output are
reported as secondary operating points. Comparators keep their default
operating point (area > 0.5%) for H1/H2; their risk-controlled versions are
reported under RQ5.

**4.3 Why this is not a free choice among variants.** The new operating point
is defined by a rule that was already pre-specified (RQ5) and does not depend
on IoU; it is chosen because a fixed threshold has no justification, not
because it scored best (the ungated output scored higher on validation and is
not chosen). Validation is used both for checkpoint selection and for
calibration, which can make the validation guarantee optimistic; the test set
remains the only basis for claims.

**4.4 Confirmation.** The 4-fold surgery-level cross-validation of §5 is run
for A1 and A5′ with the same recipe; in each fold three training surgeries
(seeded choice, seed 2027) serve as validation for checkpoint selection and
calibration. It runs after the primary test evaluation and reports the H1
contrast at the risk-controlled operating point and at the fixed 0.5 gate.

**4.5 Disclosure.** The manuscript states in its Methods that the primary
operating point was changed from the fixed 0.5 gate to the risk-controlled
threshold after validation and before test access, and reports both.

### Amendment 5 — 2026-11-04 (tag `p2-protocol-v1.5`)

Implementation details of arms that were pre-specified but not yet defined in
code, fixed before any of them is trained (test unread). All use the A5′ recipe
(DINOv2-L, LoRA r = 16, presence negatives, presence head, Dice on mask
channels, 50 epochs) unless stated, and are selected and scored like A5′.

- **A6** (three seeds): each mask target is the mean of its three annotator
  masks (values 0, 1/3, 2/3, 1), in BCE and soft Dice; validation and test
  targets stay binary (published masks).
- **Ablations** (seed 11): without LoRA (frozen DINOv2-L), without presence
  head (presence negatives kept), with `multilabel/` (dense frames: each of the
  seven organs with a non-empty mask is supervised, the others are absent), and
  label budgets of 5 and 10 training surgeries (nested subsets: first 5 and 10
  of a permutation of the 21 training surgeries with seed 2027; 21 = A5′).
- The A5 v1.2 runs (empty-target Dice) are reported as an additional ablation.
- Ablation arms and A6 use the risk-controlled operating point calibrated on
  validation in the same way as A5′ (amendment 4), and the fixed 0.5 gate as
  secondary.

### Amendment 6 — 2026-11-05 (tag `p2-protocol-v1.6`): A2 reproduction record

Recorded before candidates are frozen and before test access; no arm changes.
A2 against Kolbinger's combined DeepLabv3 validation IoU (Supplementary Table
2b; `results/a2_reproduction_check_v1.json`): 6 of 11 structures within
±0.05, mean signed difference **+0.033**; outside: abdominal wall +0.052, colon
+0.071, intestinal veins +0.075, liver +0.062 (all higher) and stomach −0.068.
Under the strict rule the check fails. Explanation: the deviation is
systematic in the favourable direction (10 of 11 structures higher), consistent
with our stronger augmentation (flips, saturation/hue jitter) and per-structure
BCE; stomach varies most between validation surgeries (0.21–0.55). A stronger
A2 makes every A5′ − A2 contrast conservative, so A2 is accepted as the
combined-model baseline and reported next to Kolbinger's published values.

### Amendment 7 — 2026-11-12 (tag `p2-protocol-v1.7`): external zero-shot evaluation

Adopted after the single test evaluation and before any external image was
passed through a model; approved by the author. Text of the plan (from
`EXTERNAL_EVAL_PLAN.md`), followed by the label check.

#### Why

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

#### Label check (done before scoring)

CholecSeg8k was downloaded by the author from Kaggle (archive SHA-256
aa0c6f3d…95e551, 8,080 frames). Watershed masks have equal channels and the grey
values 5, 11, 12, 13, 21, 22, 23, 24, 25, 31, 32, 33, 50, plus 255 (thin lines at
the image border, 1.2M pixels in 400 sampled masks) and 0 (3 pixels). Overlays
on three random frames confirmed 21 = liver, 11 = abdominal wall,
13 = gastrointestinal tract, 22 = gallbladder and 12 = fat; 255 and 0 are
treated as background. The mapping is fixed in `scripts/p2_external_eval.py`.

### Amendment 8 — 2026-11-13 (tag `p2-protocol-v1.8`): cross-validation selection rule

Made before any held-out cross-validation prediction was computed. The seeded
choice of three validation surgeries per fold (amendment 4.4) left fold 1
without ureter masks and fold 3 without intestinal-vein masks in validation
(held-out surgeries contain all structures). Specialists of those structures,
and A5′ (whose selection metric includes all four critical structures), have
no defined validation metric in those folds. Rule: **a model whose validation
selection metric is undefined keeps its final checkpoint**; everything else is
unchanged, and calibration uses the fold's validation surgeries as before.

### Analysis note 9 — 2026-11-16: annotator ceiling and dense scene evaluation (post test)

Both §5 secondary analyses were run after the test evaluation on the frozen
candidates (`scripts/p2_annotator_ceiling.py`, `scripts/p2_dense_scene.py`, job
25775/25776). Implementation choices, fixed before the first run unless marked:

- **Annotator ceiling.** "Majority of the other two" = strict majority = their
  intersection. Sensitivity: union of the other two, mean pairwise IoU, and
  IoU with the published mask (optimistic, the published mask includes the
  annotator). Frame → surgery → mean over surgeries, as for model IoU. All
  3,328 test frame masks have three annotators. Descriptive only.
- **Dense scene.** Argmax over the seven organ channels, background when the
  winning logit ≤ 0; channels dropped by each arm's frozen operating point; GT
  pixels claimed by two organs ignored (6,590 px in total). Organ IoU = Σ
  intersection / Σ union over the surgery's dense frames; scene hallucination
  = (frame, organ) pairs with empty GT and > 0.5% predicted area. n = 5
  surgeries (2, 7, 11, 18, 20), so the smallest two-sided exact sign-flip p is
  0.0625: no contrast can be significant at 0.05.
- **Post hoc (after seeing v1):** the 365 dense test frames are all frames of
  the stomach series; spleen GT is empty in every frame and small intestine is
  present in 4 frames of one surgery, so any false-positive spleen pixel gives
  organ IoU 0 in every surgery. We therefore add `miou_gt_present` (mean over
  organs with non-empty GT in that surgery) next to the pre-specified
  `miou_organs` (organs with non-zero union); both are reported
  (`results/p2_dense_scene_v2.json`; v1 is identical without the variant).
