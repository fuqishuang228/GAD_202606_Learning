#!/bin/bash
#SBATCH --job-name=dpgad_train_source
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

echo "===== DP-DGAD source training started: $(date) ====="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"$PYTHON" train_source.py \
  --dir_data ./data \
  --source_datasets Wikipedia MOOC \
  --target_datasets btc_alpha \
  --batch_size 512 \
  --n_epochs 50 \
  --num_data_workers 0

echo "===== DP-DGAD source training finished: $(date) ====="
