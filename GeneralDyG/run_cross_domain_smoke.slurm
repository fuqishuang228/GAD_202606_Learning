#!/bin/bash
#SBATCH --job-name=gdg_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

echo "===== GeneralDyG smoke started: $(date) ====="
echo "Node: $(hostname)"

"$PYTHON" run_cross_domain.py \
  --source MOOC Wikipedia \
  --target uci btc_otc \
  --epochs 2 \
  --patience 2 \
  --batch-size 16 \
  --input-dim 64 \
  --hidden-dim 128 \
  --n-layer 2 \
  --max-train-samples 1000 \
  --max-target-samples 1000 \
  --seed 0 \
  --out-dir results/paper_baselines/GeneralDyG/smoke

echo "===== GeneralDyG smoke finished: $(date) ====="
