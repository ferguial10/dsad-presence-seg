#!/bin/bash
#SBATCH --job-name=p2-cvscore
#SBATCH --partition=sc-gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Array 0-7: task = fold * 2 + arm (0 = A1, 1 = A5p). Calibrate on the fold's
# validation surgeries, then score its held-out surgeries.
set -euo pipefail
K=$((SLURM_ARRAY_TASK_ID / 2)); ARM=$([ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ] && echo A1 || echo A5p)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export PYTHONPATH=src:scripts
CV=papers/P2_anatomy_segmentation/results/cv; M=$CV/fold${K}_manifest.json; L=$(echo $ARM | tr A-Z a-z)
if [ $ARM = A1 ]; then RUNS=$(ls -d runs/p2/CV${K}_A1_*_seed11); else RUNS=runs/p2/CV${K}_A5p_seed11; fi
python scripts/p2_score.py $RUNS --arm $ARM --split val --manifest $M --output $CV/fold${K}_${L}_val_v1.json
python scripts/p2_calibrate.py $CV/fold${K}_${L}_val_v1.json --output $CV/fold${K}_${L}_calibration_v1.json
python scripts/p2_score.py $RUNS --arm $ARM --split heldout --manifest $M --output $CV/fold${K}_${L}_heldout_v1.json
