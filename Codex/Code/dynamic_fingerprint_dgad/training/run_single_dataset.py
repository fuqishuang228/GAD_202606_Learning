from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dynamic_fingerprint_dgad.data.csv_dynamic_dataset import CSVDynamicDataset
from dynamic_fingerprint_dgad.fingerprints.extractor import DynamicFingerprintExtractor, FingerprintConfig
from dynamic_fingerprint_dgad.models.detector import (
    CrossAttentionDetector,
    DynamicFingerprintDetector,
    FeatureTokenDetector,
)
from dynamic_fingerprint_dgad.training.metrics import compute_metrics
from dynamic_fingerprint_dgad.training.run_experiment import normalize_features, score_model, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_dataset_features(dataset: CSVDynamicDataset, fp_cfg: FingerprintConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
    extractor = DynamicFingerprintExtractor(dataset.num_nodes, fp_cfg)
    xs, ys = [], []
    for snap in dataset.snapshots:
        print(
            f"extract dataset={dataset.name} snapshot={snap.index + 1}/{len(dataset.snapshots)} "
            f"edges={len(snap.edges)}",
            flush=True,
        )
        xs.append(extractor.extract(snap.edges))
        ys.append(snap.labels.astype(np.float32))
        extractor.update(snap.edges)
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data"))
    parser.add_argument("--dataset", type=str, default="uci")
    parser.add_argument("--out-dir", type=Path, default=Path("results/single_uci"))
    parser.add_argument("--num-snapshots", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--cheb-order", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--model-type",
        choices=["edge_transformer", "feature_token", "feature_token_moe", "cross_attention"],
        default="edge_transformer",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--loss-type", choices=["bce", "bce_rank"], default="bce")
    parser.add_argument("--rank-lambda", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--include-edge-surprise", action="store_true")
    parser.add_argument("--include-node-activity", action="store_true")
    parser.add_argument("--relative-delta", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"device={device}", flush=True)

    dataset = CSVDynamicDataset(args.data_dir / f"{args.dataset}.csv", args.num_snapshots, args.max_rows)
    print("dataset:", dataset.summary(), flush=True)

    fp_cfg = FingerprintConfig(
        cheb_order=args.cheb_order,
        history_window=args.history_window,
        include_edge_surprise=args.include_edge_surprise,
        include_node_activity=args.include_node_activity,
        relative_delta=args.relative_delta,
    )
    xs, ys = extract_dataset_features(dataset, fp_cfg)

    split_idx = int(len(xs) * args.train_ratio)
    split_idx = max(1, min(split_idx, len(xs) - 1))
    train_xs, train_ys = xs[:split_idx], ys[:split_idx]
    test_xs, test_ys = xs[split_idx:], ys[split_idx:]
    print(
        f"temporal split: train_snapshots={len(train_xs)} test_snapshots={len(test_xs)} "
        f"train_edges={sum(len(y) for y in train_ys)} test_edges={sum(len(y) for y in test_ys)} "
        f"train_anom={int(np.concatenate(train_ys).sum())} test_anom={int(np.concatenate(test_ys).sum())}",
        flush=True,
    )

    train_xs, normalized_groups, stats = normalize_features(train_xs, [test_xs])
    test_xs = normalized_groups[0]

    if args.model_type == "edge_transformer":
        model = DynamicFingerprintDetector(
            in_dim=fp_cfg.feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    elif args.model_type in {"feature_token", "feature_token_moe"}:
        model = FeatureTokenDetector(
            cheb_order=fp_cfg.cheb_order,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            use_moe=args.model_type == "feature_token_moe",
            include_edge_surprise=fp_cfg.include_edge_surprise,
            include_node_activity=fp_cfg.include_node_activity,
        )
    elif args.model_type == "cross_attention":
        model = CrossAttentionDetector(
            cheb_order=fp_cfg.cheb_order,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            include_edge_surprise=fp_cfg.include_edge_surprise,
            include_node_activity=fp_cfg.include_node_activity,
        )
    else:
        raise ValueError(f"unknown model_type: {args.model_type}")
    model = model.to(device)
    print(f"model_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
    train_model(
        model,
        train_xs,
        train_ys,
        device,
        args.epochs,
        args.lr,
        args.chunk_size,
        args.loss_type,
        args.rank_lambda,
        args.rank_margin,
    )

    scores = score_model(model, test_xs, device, args.chunk_size)
    labels = np.concatenate(test_ys)
    metrics = compute_metrics(labels, scores)
    result = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "dataset": dataset.summary(),
        "feature_stats": stats,
        "split": {
            "train_snapshots": len(train_xs),
            "test_snapshots": len(test_xs),
            "train_edges": int(sum(len(y) for y in train_ys)),
            "test_edges": int(len(labels)),
            "train_anomalies": int(np.concatenate(train_ys).sum()),
            "test_anomalies": int(labels.sum()),
        },
        "test": metrics | {
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        },
    }
    print(f"test={result['test']}", flush=True)

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"saved={args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
