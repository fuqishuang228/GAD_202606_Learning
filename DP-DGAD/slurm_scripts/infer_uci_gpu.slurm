#!/bin/bash
#SBATCH --job-name=dpgad_infer_uci
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

echo "===== DP-DGAD UCI inference started: $(date) ====="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"$PYTHON" infer_target.py \
  --dir_data ./data \
  --target_datasets uci \
  --pretrained_model_path ./save_model/Wikipedia_MOOC_best_model.pt \
  --results_dir ./results \
  --representation_dir ./representation \
  --save_dir ./save_model \
  --batch_size 128 \
  --n_epochs 1 \
  --num_data_workers 0

echo "===== DP-DGAD UCI inference finished: $(date) ====="
