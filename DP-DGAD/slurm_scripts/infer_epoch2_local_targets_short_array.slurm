#!/bin/bash
#SBATCH --job-name=dpgad_e2_infer
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --array=0-6%1
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%A_%a.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python
MODEL=./save_model/source_epoch2_20260618_194146/Wikipedia_MOOC_best_model.pt

DATASETS=(
  btc_alpha
  btc_otc
  email_dnc
  as_topology
  tax51
  uci
  uci_temporal
)

TARGET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "===== DP-DGAD epoch=2 CPU inference started: $(date) ====="
echo "Target: ${TARGET}"
echo "Model: ${MODEL}"
echo "Node: $(hostname)"

"$PYTHON" infer_target.py \
  --dir_data ./data \
  --target_datasets "${TARGET}" \
  --pretrained_model_path "${MODEL}" \
  --results_dir "./results/epoch2_infer/${TARGET}" \
  --representation_dir "./representation/epoch2_infer/${TARGET}" \
  --save_dir ./save_model/source_epoch2_20260618_194146 \
  --batch_size 128 \
  --n_epochs 1 \
  --num_data_workers 0

echo "===== DP-DGAD epoch=2 CPU inference finished: $(date) ====="
