#!/bin/bash
#SBATCH --job-name=p2-valcal
#SBATCH --partition=sc-gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Usage: sbatch --array=0-N p2_val_calib.sbatch <arms.txt>; line = <arm> <run globs...>
set -euo pipefail
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1"); ARM=${LINE%% *}; GLOBS=${LINE#* }
source ~/miniconda3/etc/profile.d/conda.sh; conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export PYTHONPATH=src
R=papers/P2_anatomy_segmentation/results; L=$(echo $ARM | tr A-Z a-z)
python scripts/p2_score.py $GLOBS --arm $ARM --split val --manifest $R/dsad_frame_manifest_v3.json --output $R/${L}_val_native_v1.json
python scripts/p2_calibrate.py $R/${L}_val_native_v1.json --output $R/${L}_calibration_v1.json
