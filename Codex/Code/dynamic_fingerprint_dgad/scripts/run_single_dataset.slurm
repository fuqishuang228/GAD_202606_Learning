#!/bin/bash
#SBATCH --job-name=dfp_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad/slurm_logs/%x-%A_%a.err

set -euo pipefail

ROOT=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code
PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python
DATASETS=(uci btc_otc)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "===== single-dataset run started: $(date) ====="
echo "Node: $(hostname)"
echo "Dataset: ${DATASET}"

"$PYTHON" -m dynamic_fingerprint_dgad.training.run_single_dataset \
  --dataset "${DATASET}" \
  --num-snapshots 50 \
  --train-ratio 0.7 \
  --history-window 5 \
  --cheb-order 4 \
  --epochs 10 \
  --hidden-dim 64 \
  --chunk-size 512 \
  --out-dir "dynamic_fingerprint_dgad/results/single_${DATASET}"

echo "===== single-dataset run finished: $(date) ====="

