#!/bin/bash
set -euo pipefail

ROOT=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code
PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PYTHON" -m dynamic_fingerprint_dgad.training.run_experiment \
  --source MOOC Wikipedia \
  --target uci btc_otc \
  --num-snapshots 10 \
  --history-window 3 \
  --cheb-order 3 \
  --epochs 1 \
  --hidden-dim 16 \
  --num-layers 1 \
  --num-heads 2 \
  --chunk-size 64 \
  --max-rows 1000 \
  --out-dir dynamic_fingerprint_dgad/results/smoke
