#!/bin/bash
#SBATCH --job-name=dfp_sampler_ratio_btc_s0
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
OUT_ROOT=dynamic_fingerprint_dgad/results/sampler_ratio_proto_router_btc_otc_seed0

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run_balanced_ratio() {
  local ratio="$1"
  local out_dir="${OUT_ROOT}/ratio_${ratio}"
  echo "===== balanced neg_ratio=${ratio} started: $(date) ====="
  "$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
    --source MOOC Wikipedia \
    --target btc_otc_all_injection \
    --num-snapshots 50 \
    --history-window 5 \
    --cheb-order 4 \
    --epochs 50 \
    --hidden-dim 64 \
    --num-layers 2 \
    --num-heads 4 \
    --chunk-size 512 \
    --model-type mechanism_proto_router_moe \
    --router-temperature 1.0 \
    --prototype-momentum 0.9 \
    --no-mechanism-context \
    --loss-type bce \
    --sampler balanced \
    --balanced-neg-ratio "$ratio" \
    --feature-ablation full \
    --val-ratio 0.2 \
    --early-stop-metric auroc \
    --patience 8 \
    --seed 0 \
    --out-dir "$out_dir"
  echo "===== balanced neg_ratio=${ratio} finished: $(date) ====="
}

run_snapshot() {
  local out_dir="${OUT_ROOT}/snapshot_pos_weight"
  echo "===== snapshot sampler + pos_weight started: $(date) ====="
  "$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
    --source MOOC Wikipedia \
    --target btc_otc_all_injection \
    --num-snapshots 50 \
    --history-window 5 \
    --cheb-order 4 \
    --epochs 50 \
    --hidden-dim 64 \
    --num-layers 2 \
    --num-heads 4 \
    --chunk-size 512 \
    --model-type mechanism_proto_router_moe \
    --router-temperature 1.0 \
    --prototype-momentum 0.9 \
    --no-mechanism-context \
    --loss-type bce \
    --sampler snapshot \
    --feature-ablation full \
    --val-ratio 0.2 \
    --early-stop-metric auroc \
    --patience 8 \
    --seed 0 \
    --out-dir "$out_dir"
  echo "===== snapshot sampler + pos_weight finished: $(date) ====="
}

echo "===== sampler ratio diagnosis btc_otc_all seed=0 started: $(date) ====="
echo "Node: $(hostname)"

run_balanced_ratio 1
run_balanced_ratio 4
run_balanced_ratio 8
run_snapshot

echo "===== sampler ratio diagnosis btc_otc_all seed=0 finished: $(date) ====="
