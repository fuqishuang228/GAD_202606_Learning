#!/bin/bash
#SBATCH --job-name=gdg_logits
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%3
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/slurm_logs/%x-%A_%a.err

set -euo pipefail

ROOT=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG
PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

DATASETS=(uci btc_otc btc_alpha email_dnc as_topology)
SEEDS=(0 1 2)

DATASET_ID=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_ID=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
DATASET=${DATASETS[$DATASET_ID]}
SEED=${SEEDS[$SEED_ID]}

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT_DIR="results/paper_baselines/GeneralDyG_within_bce_logits/${DATASET}/seed_${SEED}"

echo "===== GeneralDyG within-dataset BCEWithLogits started: $(date) ====="
echo "Node: $(hostname)"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATASET}"
echo "Seed: ${SEED}"
echo "Output: ${OUT_DIR}"

"$PYTHON" run_within_dataset.py \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --train-ratio 0.7 \
  --epochs 50 \
  --patience 8 \
  --batch-size 64 \
  --input-dim 128 \
  --hidden-dim 258 \
  --n-heads 4 \
  --dropout 0.4 \
  --n-layer 6 \
  --lr 0.0001 \
  --loss-fn bce_with_logits \
  --out-dir "$OUT_DIR"

echo "===== GeneralDyG within-dataset BCEWithLogits finished: $(date) ====="
