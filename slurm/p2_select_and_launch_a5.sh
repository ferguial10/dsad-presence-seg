#!/bin/bash
# After A3/A4/C1: select the A5 encoder, submit A5, then release A2, C3, C2 and A1N behind it (nice orders them).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate davinci
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export PYTHONPATH=src
python scripts/p2_select_encoder.py --output papers/P2_anatomy_segmentation/results/encoder_selection_v1.json --specs papers/P2_anatomy_segmentation/runs_specs/a5_specs.txt --common "--manifest papers/P2_anatomy_segmentation/results/dsad_frame_manifest_v3.json --overlap-pairs papers/P2_anatomy_segmentation/results/overlap_pairs_v1.json --structures all --loss bce_dice --scheduler cosine --epochs 50"
sbatch --array=0-2 slurm/p2_train.sbatch papers/P2_anatomy_segmentation/runs_specs/a5_specs.txt
scontrol update jobid=22238 nice=10
scontrol update jobid=22239 nice=60
scontrol release 22238 22239 22251 22265
