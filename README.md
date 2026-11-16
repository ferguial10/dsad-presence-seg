# One model, eleven structures

Code, protocol and summary results for:

> Á. Fernández Guirao. *One model, eleven structures: presence supervision lets a
> foundation model outperform structure-specific specialists and control
> hallucination in robot-assisted rectal surgery.* Manuscript under review.

Anatomy segmentation on the Dresden Surgical Anatomy Dataset (DSAD) is usually
trained one structure at a time. Each specialist is scored only on images that
contain its structure, so it is never measured on whether it draws that structure
where it is absent. This repository trains single models with foundation
encoders, using a partial-label loss and image-level presence supervision.
Per-structure presence thresholds are calibrated with Learn-then-Test to certify
a hallucination rate of at most 5%. Everything is evaluated under a protocol
that was frozen before training.

Main test results (8 held-out surgeries, 3 seeds per arm):

| | Critical-structure IoU | Hallucination |
|---|---|---|
| A1: 11 specialists (Kolbinger et al. 2023) | 0.378 | 31.4% |
| A5′: DINOv2-L + LoRA + presence, risk-controlled | **0.440** | **≤ 5% for every structure** |

The full set of endpoints, ablations, cross-validation, external zero-shot
evaluation, annotator ceiling and dense scene evaluation are in
`papers/P2_anatomy_segmentation/results/` and in the paper.

## Protocol and provenance

- `papers/P2_anatomy_segmentation/DESIGN.md` holds the pre-registered protocol
  (tag `p2-protocol-v1`). Every amendment below it is dated and was made before
  test access unless it says otherwise. Analyses run after the test evaluation
  are labelled as such.
- `papers/P2_anatomy_segmentation/CANDIDATES_FROZEN.json` stores the SHA-256 of
  every candidate checkpoint and calibration file, fixed before the single test
  evaluation. `scripts/p2_score.py --split test` refuses any other file.
- `papers/P2_anatomy_segmentation/runs_specs/` holds the exact command-line
  arguments of every training run.

## Installation

```bash
conda env create -f envs/davinci.yaml   # Python 3.11, PyTorch 2.6 + CUDA 12.4
conda activate davinci
pip install -e .
pytest -q
```

The experiments ran on NVIDIA V100 32 GB with fp16 autocast. Encoders are
downloaded from the Hugging Face Hub on first use: `facebook/dinov2-*` (Apache
2.0) and `facebook/sam2.1-hiera-*` (Apache 2.0). The C2 comparator needs the
SurgeNetXL weights (CC BY-NC-SA), which are not redistributed here.

## Data

- **DSAD** (CC BY 4.0): Carstens et al., *Sci Data* 2023,
  [doi:10.6084/m9.figshare.21702600](https://doi.org/10.6084/m9.figshare.21702600).
  Unpack it to `data/DSAD/`. The split is that of Kolbinger et al., *Int J Surg*
  2023 (`src/davinci/data/dsad.py`).
- **CholecSeg8k** and **EndoVis 2018 RSS** are used only for the external
  zero-shot evaluation (see `EXTERNAL_EVAL_PLAN.md`).

```bash
python scripts/build_dsad_manifest.py --root data/DSAD --output manifest.json
# or reuse the exact manifest of the paper (Zenodo archive) and point it to your copy:
python scripts/relocate_manifest.py papers/P2_anatomy_segmentation/results/dsad_frame_manifest_v3.json \
    --root data/DSAD --output papers/P2_anatomy_segmentation/results/dsad_frame_manifest_v3_local.json
python scripts/build_dsad_cache.py --manifest <manifest> --cache-root cache --size 512x640
python scripts/build_dsad_cache.py --manifest <manifest> --cache-root cache --size 518x644
```

## Checkpoints and full results (Zenodo)

The Zenodo record (DOI below) contains:

- the frozen checkpoints of A5′ (3 seeds) and of the 33 A1 specialists (11
  structures × 3 seeds);
- the per-image score files (`*_native_v1.json`, `*_external_v1.json`, `cv/`)
  and the frame manifests, which are too large for git.

Unpack the checkpoints into `runs/p2/`, then check them against the frozen hashes:

```bash
python scripts/verify_checkpoints.py runs/p2
```

## Reproducing the paper

All commands run from the repository root with `PYTHONPATH=src:scripts`. The
SLURM wrappers in `slurm/` show the exact sequence. Their partition and account
lines are specific to our cluster.

```bash
# 1. Train (one line of a specs file per run; ~6-12 GPU-hours per DINOv2-L run)
sbatch --array=0-2 slurm/p2_train.sbatch papers/P2_anatomy_segmentation/runs_specs/a5p_specs.txt
# 2. Score on validation and calibrate the presence thresholds (Learn-then-Test, alpha 0.05, delta 0.1)
sbatch --array=0-N slurm/p2_val_calib.sbatch papers/P2_anatomy_segmentation/runs_specs/val_calib_arms.txt
# 3. Score the frozen candidates on test (refuses unfrozen checkpoints)
sbatch --array=0-N slurm/p2_test_score.sbatch papers/P2_anatomy_segmentation/runs_specs/test_arms.txt
# 4. Endpoints and statistics
python scripts/p2_analyze.py --arm A5p <a5p_test_native_v1.json> calibrated:<a5p_calibration_v1.json> \
    --arm A1 <a1_test_native_v1.json> default --arm A4 <a4_test_native_v1.json> default --output analysis.json
# 5. Secondary analyses: cross-validation, external, annotator ceiling, dense scenes
#    slurm/p2_cv_score.sbatch, slurm/p2_external.sbatch, slurm/p2_secondary.sbatch
# 6. Tables and figures of the paper
python papers/P2_anatomy_segmentation/manuscript/make_tables.py
python papers/P2_anatomy_segmentation/manuscript/make_figures.py   # needs the Zenodo score files
```

The summary JSONs in `papers/P2_anatomy_segmentation/results/` are the ones the
paper reports: `p2_test_analysis_v1.json`, `p2_external_analysis_v1.json`,
`cv/p2_cv_analysis_v1.json`, `p2_annotator_ceiling_v1.json` and
`p2_dense_scene_v2.json`, together with the calibrations.

## Layout

```
src/davinci/          data (DSAD manifest, fused partial labels), models, metrics, statistics
scripts/              training, scoring, calibration, analysis
slurm/                job wrappers used for the paper
tests/                unit tests (pytest)
papers/P2_anatomy_segmentation/
  DESIGN.md           protocol and amendments
  runs_specs/         exact arguments of every run
  results/            summary results (per-image files on Zenodo)
  manuscript/         table and figure generators
```

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The checkpoints on Zenodo
are released under CC BY 4.0 and were trained on DSAD (CC BY 4.0) from DINOv2
(Apache 2.0). Not for clinical use. To cite this work, use `CITATION.cff` or
the Zenodo DOI.
