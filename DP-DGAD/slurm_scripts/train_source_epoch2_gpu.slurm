#!/bin/bash
#SBATCH --job-name=dpgad_train_e2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python
SAVE_DIR=./save_model/source_epoch2_$(date +%Y%m%d_%H%M%S)

echo "===== DP-DGAD source epoch=2 training started: $(date) ====="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Save dir: ${SAVE_DIR}"
nvidia-smi || true

"$PYTHON" train_source.py \
  --dir_data ./data \
  --source_datasets Wikipedia MOOC \
  --target_datasets btc_alpha \
  --batch_size 512 \
  --n_epochs 2 \
  --num_data_workers 0 \
  --save_dir "${SAVE_DIR}"

echo "===== DP-DGAD source epoch=2 training finished: $(date) ====="
