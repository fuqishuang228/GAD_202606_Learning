#!/bin/bash
#SBATCH --job-name=gdg_3seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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

echo "===== GeneralDyG 3-seed cross-domain started: $(date) ====="
echo "Node: $(hostname)"

for SEED in 0 1 2; do
  echo "===== seed=${SEED} ====="
  "$PYTHON" run_cross_domain.py \
    --source MOOC Wikipedia \
    --target uci btc_otc btc_alpha email_dnc as_topology \
    --epochs 50 \
    --patience 8 \
    --batch-size 64 \
    --input-dim 128 \
    --hidden-dim 258 \
    --n-heads 4 \
    --dropout 0.4 \
    --n-layer 6 \
    --lr 0.0001 \
    --val-ratio 0.2 \
    --seed "$SEED" \
    --out-dir "results/paper_baselines/GeneralDyG/seed_${SEED}"
done

echo "===== GeneralDyG 3-seed cross-domain finished: $(date) ====="
