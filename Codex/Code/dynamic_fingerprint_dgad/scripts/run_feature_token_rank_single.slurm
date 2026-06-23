#!/bin/bash
#SBATCH --job-name=dfp_ftr_single
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

echo "===== feature-token rank single run started: $(date) ====="
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
  --num-layers 2 \
  --num-heads 4 \
  --chunk-size 512 \
  --model-type feature_token \
  --loss-type bce_rank \
  --rank-lambda 1.0 \
  --rank-margin 1.0 \
  --out-dir "dynamic_fingerprint_dgad/results/feature_token_rank_single_${DATASET}"

echo "===== feature-token rank single run finished: $(date) ====="

