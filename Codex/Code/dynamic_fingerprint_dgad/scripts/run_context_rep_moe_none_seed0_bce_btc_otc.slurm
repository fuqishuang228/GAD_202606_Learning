#!/bin/bash
#SBATCH --job-name=dfp_ctxrep_none_btc_s0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
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

echo "===== Context Representation MoE none raw10 seed=0 target=btc_otc_all_injection started: $(date) ====="
echo "Node: $(hostname)"

"$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
  --source MOOC Wikipedia \
  --target btc_otc_all_injection \
  --num-snapshots 50 \
  --history-window 5 \
  --cheb-order 4 \
  --epochs 50 \
  --hidden-dim 64 \
  --num-layers 1 \
  --num-heads 4 \
  --chunk-size 512 \
  --model-type context_rep_moe \
  --context-mode none \
  --loss-type bce \
  --sampler balanced \
  --balanced-neg-ratio 1 \
  --feature-ablation full \
  --feature-view full \
  --val-ratio 0.2 \
  --early-stop-metric auroc \
  --patience 8 \
  --seed 0 \
  --out-dir dynamic_fingerprint_dgad/results/context_rep_moe_none_raw10_bestval50_bce/btc_otc_seed_0

echo "===== Context Representation MoE none raw10 seed=0 target=btc_otc_all_injection finished: $(date) ====="
