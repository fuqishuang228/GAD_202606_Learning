#!/bin/bash
#SBATCH --job-name=dpgad_infer_targets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%A_%a.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

DATASETS=(
  as_topology
  btc_alpha
  email_dnc
  btc_otc
  tax51
)

TARGET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "===== DP-DGAD inference started: $(date) ====="
echo "Target: ${TARGET}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"$PYTHON" infer_target.py \
  --dir_data ./data \
  --target_datasets "${TARGET}" \
  --pretrained_model_path ./save_model/Wikipedia_MOOC_best_model.pt \
  --results_dir "./results/all_targets/${TARGET}" \
  --representation_dir "./representation/all_targets/${TARGET}" \
  --save_dir ./save_model \
  --batch_size 128 \
  --n_epochs 1 \
  --num_data_workers 0

echo "===== DP-DGAD inference finished: $(date) ====="
