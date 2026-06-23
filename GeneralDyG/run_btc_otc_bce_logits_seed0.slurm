#!/bin/bash
#SBATCH --job-name=gdg_otc_lg0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG
PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

OUT_DIR="results/paper_baselines/GeneralDyG_within_bce_logits/btc_otc/seed_0"

echo "===== GeneralDyG BTC-OTC BCEWithLogits seed 0 started: $(date) ====="
echo "Node: $(hostname)"
echo "Output: ${OUT_DIR}"

"$PYTHON" run_within_dataset.py \
  --dataset btc_otc \
  --seed 0 \
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

echo "===== GeneralDyG BTC-OTC BCEWithLogits seed 0 finished: $(date) ====="
