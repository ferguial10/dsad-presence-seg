#!/bin/bash
#SBATCH --job-name=p2-ext
#SBATCH --partition=sc-gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
# Line = <arm> <dataset> <root> <calibration or -> <run globs...>
set -euo pipefail
read -r ARM DS ROOT CAL GLOBS <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export PYTHONPATH=src:scripts
R=papers/P2_anatomy_segmentation/results; L=$(echo $ARM | tr A-Z a-z)
EXTRA=""; [ "$CAL" != "-" ] && EXTRA="--calibration $CAL"
python scripts/p2_external_eval.py $GLOBS --arm $ARM --dataset $DS --root $ROOT $EXTRA --output $R/${L}_${DS}_external_v1.json
