#!/bin/bash
#SBATCH --job-name=dfp_highonly_s0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad/slurm_logs/%x-%A_%a.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad/slurm_logs/%x-%A_%a.err

set -euo pipefail

ROOT=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code
PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

TARGETS=(
  uci_temporal_all_injection
  btc_otc_all_injection
  btc_alpha_all_injection
  email_dnc_all_injection
  as_topology_all_injection
)
SHORT_NAMES=(
  uci_temporal_all
  btc_otc_all
  btc_alpha_all
  email_dnc_all
  as_topology_all
)

TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
SHORT_NAME=${SHORT_NAMES[$SLURM_ARRAY_TASK_ID]}

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "===== high_order_only edge_transformer seed=0 BCE target=${TARGET} started: $(date) ====="
echo "Node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

"$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
  --source MOOC Wikipedia \
  --target "$TARGET" \
  --num-snapshots 50 \
  --history-window 5 \
  --cheb-order 4 \
  --epochs 50 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --chunk-size 512 \
  --model-type edge_transformer \
  --feature-view high_order_only \
  --loss-type bce \
  --sampler balanced \
  --feature-ablation full \
  --val-ratio 0.2 \
  --early-stop-metric auroc \
  --patience 8 \
  --seed 0 \
  --out-dir "dynamic_fingerprint_dgad/results/high_order_only_bestval50_bce/seed_0_all_targets/${SHORT_NAME}_seed_0"

echo "===== high_order_only edge_transformer seed=0 BCE target=${TARGET} finished: $(date) ====="
