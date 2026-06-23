#!/bin/bash
#SBATCH --job-name=dpgad_tax51_prep
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

echo "===== DP-DGAD TAX51 target preparation started: $(date) ====="
echo "Node: $(hostname)"

"$PYTHON" prepare_dpgad_data.py \
  --dataset tax51 \
  --raw-dir ../Data/Dynamic_Graph_data/TAX51 \
  --event-mode temporal_events \
  --anomaly-per 0.1 \
  --anomaly-base total \
  --output-name tax51 \
  --max-edges 24 \
  --save-csv

echo "===== DP-DGAD TAX51 target preparation finished: $(date) ====="
