#!/bin/bash
# Merge the four folds and analyse A5p (risk-controlled and fixed 0.5 gate) against A1.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export PYTHONPATH=src:scripts
CV=papers/P2_anatomy_segmentation/results/cv
for L in a1 a5p; do
  python scripts/p2_merge_cv.py $CV/fold{0,1,2,3}_${L}_heldout_v1.json --output $CV/cv_${L}_heldout_merged_v1.json
  python scripts/p2_merge_cv.py $CV/fold{0,1,2,3}_${L}_calibration_v1.json --output $CV/cv_${L}_calibration_merged_v1.json
done
python scripts/p2_analyze.py --arm A1 $CV/cv_a1_heldout_merged_v1.json default \
  --arm A5p $CV/cv_a5p_heldout_merged_v1.json calibrated:$CV/cv_a5p_calibration_merged_v1.json \
  --arm A5p_gate $CV/cv_a5p_heldout_merged_v1.json gate0.5 \
  --arm A5p_ungated $CV/cv_a5p_heldout_merged_v1.json ungated \
  --arm A1_cal $CV/cv_a1_heldout_merged_v1.json calibrated:$CV/cv_a1_calibration_merged_v1.json \
  --output $CV/p2_cv_analysis_v1.json
