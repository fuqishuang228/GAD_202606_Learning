# Codex Handoff: Dynamic Fingerprint DGAD

Date: 2026-06-23

## User Goal

The long-term goal is to develop a general dynamic graph anomaly detection algorithm for AAAI submission around late July.

Current idea:

- Extract lightweight dynamic graph fingerprints.
- Feed these raw multi-dimensional features into an encoder.
- Later consider cross-attention or MoE over feature representations.
- Avoid too many hand-crafted features, because too many features look like feature engineering.
- Prefer fast, interpretable metrics.

Important user preference:

- Explain simply and concretely.
- Avoid overly long theory dumps.
- Focus on one feasible plan at a time.

## Current Main Method

Project path:

`/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad`

Data path:

`/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data`

Python:

`/home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/python`

Main model currently used:

- `FeatureTokenDetector`
- 10-dimensional dynamic fingerprint:
  - static structural: `nCN`, `CP1`, `CP2`, `CP3`, `CP4`
  - temporal delta: `d_nCN`, `d_CP1`, `d_CP2`, `d_CP3`, `d_CP4`
- Source datasets: `MOOC + Wikipedia`
- Target datasets: `uci`, `btc_otc`, `btc_alpha`, `email_dnc`, `as_topology`
- Training:
  - epochs 50
  - validation ratio 0.2
  - patience 8
  - early stop on validation AUROC
  - best validation model is loaded before testing
  - sampler is `balanced`
  - loss is ordinary `BCEWithLogitsLoss` because balanced sampler disables `pos_weight`

Balanced sampler details:

- Each batch uses half anomaly edges and half normal edges.
- With `chunk_size=512`, it samples about:
  - 256 anomaly edges
  - 256 normal edges
- If a class has fewer samples than needed, it samples with replacement.
- Because normal edges are much more numerous, this effectively oversamples anomalies and undersamples normals.
- Evaluation is full target set evaluation, no sampling.

Relevant code:

- `training/run_experiment.py`
- `models/detector.py`
- `fingerprints/extractor.py`

## Our Current Main Results

Result directory:

`results/seed_stability_bestval50_bce`

Setting:

`MOOC + Wikipedia -> target`

Feature-token detector, 3 seeds.

AUROC mean/std:

- `uci`: 0.7090 ± 0.0038
- `btc_otc`: 0.7027 ± 0.0012
- `btc_alpha`: 0.7169 ± 0.0013
- `email_dnc`: 0.8843 ± 0.0207
- `as_topology`: 0.7663 ± 0.0021
- macro: 0.7558 ± 0.0041

AUPRC macro:

- 0.1947 ± 0.0051

Important observation:

- AUROC is okay, but AUPRC is low, especially on `uci`.
- Interpretation: the model has some global ranking ability, but top anomaly retrieval is weak. It does not push true anomalies cleanly to the highest score region.

## Other Model Variants Already Tried

Cross-attention:

- Result directory: `results/cross_attention_bestval50_bce`
- macro AUROC: 0.7536 ± 0.0096
- macro AUPRC: 0.1939 ± 0.0079
- Similar to feature-token, not clearly better.

Semantic MoE:

- Result directory: `results/semantic_moe_bestval50_bce`
- macro AUROC: 0.7257 ± 0.0451
- macro AUPRC: 0.1878 ± 0.0172
- Unstable and worse.

Feature-token MoE + center:

- Result directory: `results/feature_token_moe_center_bestval50_bce`
- macro AUROC: 0.7366 ± 0.0208
- macro AUPRC: 0.1898 ± 0.0096
- Worse than main model.

Prototype-MoE with PromptMoE-style load balance:

- Result directory: `results/proto_moe_balance_bestval50_bce`
- macro AUROC: 0.7542 ± 0.0062
- macro AUPRC: 0.1959 ± 0.0027
- Similar to main model.
- Useful finding: load-balanced MoE avoids collapse, but does not clearly improve performance.

MoE details:

- Implemented `FeatureTokenProtoMoE` in `models/detector.py`.
- Supports:
  - `router_top_k`
  - router temperature
  - PromptMoE-style load balance loss:
    `alpha * E * sum_j(mean_batch p_j)^2`
  - warmup epochs

## Discussion About AUPRC

User asked why our AUPRC is low.

Short explanation:

- AUROC asks whether a random anomaly gets higher score than a random normal edge.
- AUPRC asks whether the highest-scored edges are truly anomalous.
- Our model can roughly rank anomalies above normals, but top-ranked edges contain many normal edges.
- This means top anomaly retrieval / precision is weak.

Likely causes:

- Balanced 1:1 sampler makes anomalies look too common during training.
- Normal edge diversity is undersampled.
- Cross-domain shift is large: `MOOC + Wikipedia -> uci`.
- BCE does not directly optimize top-ranked anomaly retrieval.

Potential next experiment:

- Try gentler sampling ratios instead of 1:1:
  - anomaly:normal = 1:4
  - anomaly:normal = 1:8
- Or use full/less-balanced training with `pos_weight = num_negative / num_positive`.

Current code behavior:

- If `sampler != balanced`, code uses:
  `pos_weight = num_negative / num_positive`
- If `sampler == balanced`, code uses:
  `pos_weight = None`

## GeneralDyG Baseline

GeneralDyG path:

`/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG`

Paper:

`/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/2024_A Generalizable Anomaly Detection Method in Dynamic Graphs.pdf`

Important finding:

- GeneralDyG original paper/code setting is not source-domain train -> target-domain test.
- It is within-dataset train/test.
- For Bitcoin Alpha/OTC it follows TADDY-style anomaly ratios and uses prepared labeled data.
- The code does not inject anomalies during training.
- `generate_datasets.py` reads already-labeled CSV with a `label` column and converts it to pkl.

GeneralDyG seed:

- `option.py` has default seed `95540`.
- It appears arbitrary/fixed.
- For our baseline, seeds 0, 1, 2 are used for consistency.

## GeneralDyG Adaptation

I did not modify original GeneralDyG model files:

- `model/`
- `datasets.py`
- `generate_datasets.py`
- `train.py`
- `option.py`
- `utils.py`

I added wrapper scripts:

- `run_within_dataset.py`
- `run_within_dataset_5x3.slurm`
- `run_within_dataset_5x3_bce_logits.slurm`
- `run_cross_domain.py`
- `run_cross_domain_3seeds.slurm`
- `run_cross_domain_smoke.slurm`

Current official GeneralDyG baseline used:

- `run_within_dataset.py`
- `run_within_dataset_5x3.slurm`

Behavior:

- Reads our prepared pkl files from `DP-DGAD/data`.
- Splits each dataset temporally:
  - first 70% train
  - last 30% test
- Uses GeneralDyG original model components.
- Trains 50 epochs with patience 8.
- Currently selects best model by test AUROC, similar to GeneralDyG original code style, but this is test leakage.

Important caveat:

- The wrapper uses best test AUROC for early stopping/model selection.
- This is not fully rigorous.
- A cleaner version should use train/val/test:
  - train: train model
  - val: early stop and select best model
  - test: final evaluation only once

## GeneralDyG BCE vs BCEWithLogits

GeneralDyG model output:

- `TransformerBinaryClassifier` applies `sigmoid` inside forward.
- Therefore it returns probability, not raw logits.

Original `train.py` uses:

`binary_cross_entropy_with_logits`

This is mathematically mismatched if input is already sigmoid probability.

My first wrapper used:

`binary_cross_entropy(pred.clamp(...), y)`

This is more consistent with probability output.

User asked to also try source-like `binary_cross_entropy_with_logits`.

I updated `run_within_dataset.py` to support:

- `--loss-fn prob_bce`
- `--loss-fn bce_with_logits`

New BCEWithLogits job submitted:

- job id: `57438631`
- script:
  `run_within_dataset_5x3_bce_logits.slurm`
- output:
  `results/paper_baselines/GeneralDyG_within_bce_logits/{dataset}/seed_{seed}/metrics.json`

Current status at time of handoff:

- job `57438631` was submitted and pending.
- Need monitor with:

```bash
squeue -j 57438631 -o '%.18i %.14j %.8T %.10M %.9l %.20b %.30R'
```

## GeneralDyG Probability-BCE Results

Result directory:

`/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/results/paper_baselines/GeneralDyG_within`

All 15 tasks completed.

Results:

```text
uci
seed_0  AUROC=0.6938  AUPRC=0.2715
seed_1  AUROC=0.6976  AUPRC=0.2966
seed_2  AUROC=0.7495  AUPRC=0.3571

btc_otc
seed_0  AUROC=0.6855  AUPRC=0.3530
seed_1  AUROC=0.7461  AUPRC=0.4011
seed_2  AUROC=0.7778  AUPRC=0.4725

btc_alpha
seed_0  AUROC=0.7272  AUPRC=0.4504
seed_1  AUROC=0.6075  AUPRC=0.3005
seed_2  AUROC=0.5910  AUPRC=0.3373

email_dnc
seed_0  AUROC=0.8995  AUPRC=0.7148
seed_1  AUROC=0.9233  AUPRC=0.7813
seed_2  AUROC=0.9225  AUPRC=0.7615

as_topology
seed_0  AUROC=0.8303  AUPRC=0.6123
seed_1  AUROC=0.7874  AUPRC=0.5528
seed_2  AUROC=0.7355  AUPRC=0.4629
```

Note:

- These results are within-dataset.
- Our main model results are cross-dataset.
- Therefore they are not directly fair to compare.

## AtlasULP Code Finding

AtlasULP path:

`/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Atlas/AtlasULP_code`

User asked whether AtlasULP selects model using test or train.

Findings:

- Main script `main.py` has no best model selection.
- It trains fixed epochs, default 1000.
- It does not early stop.
- It evaluates test only after training.
- So `main.py` uses the final epoch model.

`complete.py` has a different traditional link-prediction training loop:

- It uses validation metric for early stopping.
- It does not use test to select best.
- But it also does not appear to reload the best checkpoint before inference.

## Recommended Next Steps

1. Monitor GeneralDyG BCEWithLogits job:

```bash
squeue -j 57438631 -o '%.18i %.14j %.8T %.10M %.9l %.20b %.30R'
```

2. Summarize BCEWithLogits results after completion and compare with probability-BCE results.

3. For fair comparison, decide one of these settings:

- within-dataset for both our method and GeneralDyG
- or source-domain training for both our method and GeneralDyG

4. Improve our method's AUPRC:

- try sampler ratios:
  - 1:4 anomaly:normal
  - 1:8 anomaly:normal
- try non-balanced sampler with weighted BCE
- consider AUPRC/top-ranking-oriented loss
- consider more normal diversity in each epoch

5. Make our experimental protocol stricter:

- train/val/test
- val selects best model
- test evaluated once

## Useful Commands

Check GeneralDyG BCEWithLogits job:

```bash
squeue -j 57438631 -o '%.18i %.14j %.8T %.10M %.9l %.20b %.30R'
```

Check GeneralDyG logs:

```bash
find '/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/slurm_logs' \
  -maxdepth 1 -name 'gdg_logits-57438631_*.out' -print | sort | \
  xargs -r -I{} sh -c 'echo "===== $(basename "$1") ====="; tail -8 "$1"' sh {}
```

Summarize GeneralDyG probability-BCE results:

```bash
python - <<'PY'
import json
from pathlib import Path
root=Path('/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/GeneralDyG/results/paper_baselines/GeneralDyG_within')
for p in sorted(root.glob('*/*/metrics.json')):
    if 'smoke' in str(p):
        continue
    d=json.loads(p.read_text())
    print(f'{p.parts[-3]:12s} {p.parts[-2]:7s} best={d["training"]["best_epoch"]:3d} AUROC={d["test"]["auroc"]:.4f} AUPRC={d["test"]["auprc"]:.4f}')
PY
```

Summarize our main model results:

```bash
cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/Codex/Code/dynamic_fingerprint_dgad
find results/seed_stability_bestval50_bce -path '*/metrics.json' -print
```

