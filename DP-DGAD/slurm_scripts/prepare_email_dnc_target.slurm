#!/bin/bash
#SBATCH --job-name=dpgad_email_dnc_prep
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.out
#SBATCH --error=/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PYTHON=/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python

echo "===== DP-DGAD Email-DNC target preparation started: $(date) ====="
echo "Node: $(hostname)"

"$PYTHON" prepare_dpgad_data.py \
  --dataset email_dnc \
  --raw-dir ../Data/Dynamic_Graph_data/Email-DNC \
  --event-mode temporal_events \
  --anomaly-per 0.1 \
  --anomaly-base total \
  --output-name email_dnc \
  --max-edges 24 \
  --save-csv

echo "===== DP-DGAD Email-DNC target preparation finished: $(date) ====="
