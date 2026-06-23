#!/bin/bash
#SBATCH --job-name=dpgad_infer_targets_cpu
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%A_%a.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

DATASETS=(
  as_topology
  btc_alpha
  email_dnc
  btc_otc
  tax51
)

TARGET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "===== DP-DGAD CPU inference started: $(date) ====="
echo "Target: ${TARGET}"
echo "Node: $(hostname)"

"$PYTHON" infer_target.py \
  --dir_data ./data \
  --target_datasets "${TARGET}" \
  --pretrained_model_path ./save_model/Wikipedia_MOOC_best_model.pt \
  --results_dir "./results/all_targets_cpu/${TARGET}" \
  --representation_dir "./representation/all_targets_cpu/${TARGET}" \
  --save_dir ./save_model \
  --batch_size 128 \
  --n_epochs 1 \
  --num_data_workers 0

echo "===== DP-DGAD CPU inference finished: $(date) ====="
