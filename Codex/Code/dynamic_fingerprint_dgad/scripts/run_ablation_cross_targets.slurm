#!/bin/bash
#SBATCH --job-name=dfp_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code
PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

COMMON_ARGS=(
  --source MOOC Wikipedia
  --target uci btc_otc btc_alpha email_dnc as_topology
  --num-snapshots 50
  --history-window 5
  --cheb-order 4
  --epochs 5
  --hidden-dim 64
  --num-layers 2
  --num-heads 4
  --chunk-size 512
  --sampler balanced
)

echo "===== ablation cross-target run started: $(date) ====="
echo "Node: $(hostname)"

echo "===== ablation: no_rank_loss ====="
"$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
  "${COMMON_ARGS[@]}" \
  --model-type feature_token \
  --loss-type bce \
  --feature-ablation full \
  --out-dir dynamic_fingerprint_dgad/results/ablation_no_rank_loss

echo "===== ablation: no_temporal_delta ====="
"$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
  "${COMMON_ARGS[@]}" \
  --model-type feature_token \
  --loss-type bce_rank \
  --rank-lambda 1.0 \
  --rank-margin 1.0 \
  --feature-ablation no_temporal \
  --out-dir dynamic_fingerprint_dgad/results/ablation_no_temporal_delta

echo "===== ablation: no_feature_token_encoder ====="
"$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
  "${COMMON_ARGS[@]}" \
  --model-type edge_transformer \
  --loss-type bce_rank \
  --rank-lambda 1.0 \
  --rank-margin 1.0 \
  --feature-ablation full \
  --out-dir dynamic_fingerprint_dgad/results/ablation_no_feature_token_encoder

echo "===== ablation cross-target run finished: $(date) ====="
