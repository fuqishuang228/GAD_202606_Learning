#!/bin/bash
#SBATCH --job-name=dpgad_mooc_prep
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

echo "===== DP-DGAD MOOC source preparation started: $(date) ====="
echo "Node: $(hostname)"

"$PYTHON" prepare_dpgad_data.py \
  --dataset mooc \
  --output-name MOOC \
  --max-edges 24 \
  --save-csv

echo "===== DP-DGAD MOOC source preparation finished: $(date) ====="
