#!/bin/bash
#SBATCH --job-name=p2-dsad
#SBATCH --partition=sc-gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
# Usage: sbatch --array=0-N p2_train.sbatch <specs.txt>
# Each line of specs.txt: <run_name> <extra train_dsad.py args...>
set -euo pipefail
SPECS=$1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPECS")
NAME=${LINE%% *}; ARGS=${LINE#* }
source ~/miniconda3/etc/profile.d/conda.sh
conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH=$PWD/src
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID commit=$(git rev-parse --short HEAD) run=$NAME"
# Runs longer than the walltime save last.pt and exit 75; requeue to resume.
set +e
python scripts/train_dsad.py \
  --manifest papers/P2_anatomy_segmentation/results/dsad_frame_manifest_v2.json \
  --cache-root ${P2_CACHE_ROOT:-cache} \
  --time-budget-min 450 --output runs/p2/$NAME $ARGS
STATUS=$?
set -e
if [ $STATUS -eq 75 ]; then
  scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
exit $STATUS
