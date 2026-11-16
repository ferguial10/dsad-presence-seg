#!/bin/bash
#SBATCH --job-name=p2-secondary
#SBATCH --partition=sc-gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Secondary analyses of DESIGN.md §5: annotator ceiling and dense scene evaluation (test, frozen candidates).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export PYTHONPATH=src:scripts
R=papers/P2_anatomy_segmentation/results
[ -e $R/p2_annotator_ceiling_v1.json ] || python scripts/p2_annotator_ceiling.py --manifest $R/dsad_frame_manifest_v3.json \
  --analysis $R/p2_test_analysis_v1.json --output $R/p2_annotator_ceiling_v1.json
python scripts/p2_dense_scene.py --manifest $R/dsad_frame_manifest_v3.json --output $R/p2_dense_scene_v2.json \
  --arm A5p $R/a5p_test_native_v1.json calibrated:$R/a5p_calibration_v1.json \
  --arm A5p_ungated $R/a5p_test_native_v1.json ungated \
  --arm A1 $R/a1_test_native_v1.json default \
  --arm A1_cal $R/a1_test_native_v1.json calibrated:$R/a1_calibration_v1.json \
  --arm A1N $R/a1n_test_native_v1.json default \
  --arm A2 $R/a2_test_native_v1.json default \
  --arm A4 $R/a4_test_native_v1.json default \
  --arm A6 $R/a6_test_native_v1.json calibrated:$R/a6_calibration_v1.json \
  --arm C1 $R/c1_test_native_v1.json default \
  --arm C2 $R/c2_test_native_v1.json default \
  --arm C3 $R/c3_test_native_v1.json gate0.5
