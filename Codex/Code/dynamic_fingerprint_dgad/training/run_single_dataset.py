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
    ContextRepresentationMoE,
    CrossAttentionDetector,
    DynamicFingerprintDetector,
    FeatureTokenDetector,
    MechanismContextSoftMoE,
    MechanismPrototypeRouterMoE,
    MechanismSoftMoE,
)
from dynamic_fingerprint_dgad.training.metrics import compute_metrics
from dynamic_fingerprint_dgad.training.run_experiment import (
    mechanism_expert_statistics,
    normalize_features,
    prototype_evidence_statistics,
    router_statistics,
    score_model,
    train_model,
    transform_feature_view,
    build_target_context_stats,
    context_representation_statistics,
)


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
        choices=[
            "edge_transformer",
            "feature_token",
            "feature_token_moe",
            "context_rep_moe",
            "mechanism_context_soft_moe",
            "mechanism_proto_router_moe",
            "mechanism_soft_moe",
            "cross_attention",
        ],
        default="edge_transformer",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--loss-type", choices=["bce", "bce_rank"], default="bce")
    parser.add_argument("--rank-lambda", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument("--prototype-evidence-loss-lambda", type=float, default=0.0)
    parser.add_argument("--sampler", choices=["snapshot", "balanced"], default="snapshot")
    parser.add_argument("--balanced-neg-ratio", type=float, default=1.0)
    parser.add_argument("--feature-view", choices=["full", "high_order_only", "high_order_multiview", "high_order_residual"], default="full")
    parser.add_argument("--residual-alpha-fast", type=float, default=0.7)
    parser.add_argument("--residual-alpha-slow", type=float, default=0.95)
    parser.add_argument("--mechanism-router-mode", choices=["learned", "uniform", "global"], default="learned")
    parser.add_argument("--mechanism-high-order-view", choices=["raw", "multiview"], default="raw")
    parser.add_argument("--no-mechanism-context", action="store_true")
    parser.add_argument("--context-mode", choices=["none", "target_mean"], default="none")
    parser.add_argument("--include-edge-surprise", action="store_true")
    parser.add_argument("--include-node-activity", action="store_true")
    parser.add_argument("--relative-delta", action="store_true")
    parser.add_argument("--snapshot-relative-features", action="store_true")
    parser.add_argument("--relative-feature-mode", choices=["none", "all", "local"], default="none")
    parser.add_argument("--fingerprint-variant", choices=["raw10", "atlas_local_k2", "atlas_local_k2_fast", "atlas_local_k1"], default="raw10")
    parser.add_argument("--mechanism-feature-mode", choices=["raw10", "atlas_local_k2", "atlas_local_k2_fast", "atlas_local_k1"], default="raw10")
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
        snapshot_relative_features=args.snapshot_relative_features,
        relative_feature_mode=args.relative_feature_mode,
        fingerprint_variant=args.fingerprint_variant,
    )
    xs, ys = extract_dataset_features(dataset, fp_cfg)
    xs = transform_feature_view(
        xs,
        args.feature_view,
        [snap.edges for snap in dataset.snapshots],
        dataset.num_nodes,
        args.residual_alpha_fast,
        args.residual_alpha_slow,
    )
    feature_dim = int(xs[0].shape[1]) if xs else fp_cfg.feature_dim
    print(f"feature_view={args.feature_view} feature_dim={feature_dim}", flush=True)

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
    if args.model_type == "context_rep_moe":
        if feature_dim != 10:
            raise ValueError(f"context_rep_moe expects raw10 full features, got feature_dim={feature_dim}")
        if args.context_mode == "target_mean":
            print("single-dataset context-mode=target_mean uses unlabeled test split features as context", flush=True)

    if args.model_type == "edge_transformer":
        model = DynamicFingerprintDetector(
            in_dim=feature_dim,
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
    elif args.model_type == "context_rep_moe":
        model = ContextRepresentationMoE(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            context_mode=args.context_mode,
        )
    elif args.model_type == "mechanism_soft_moe":
        model = MechanismSoftMoE(
            cheb_order=fp_cfg.cheb_order,
            hidden_dim=args.hidden_dim,
            include_edge_surprise=fp_cfg.include_edge_surprise,
            include_node_activity=fp_cfg.include_node_activity,
            router_mode=args.mechanism_router_mode,
        )
    elif args.model_type == "mechanism_context_soft_moe":
        model = MechanismContextSoftMoE(
            cheb_order=fp_cfg.cheb_order,
            hidden_dim=args.hidden_dim,
            include_edge_surprise=fp_cfg.include_edge_surprise,
            include_node_activity=fp_cfg.include_node_activity,
            router_mode=args.mechanism_router_mode,
        )
    elif args.model_type == "mechanism_proto_router_moe":
        model = MechanismPrototypeRouterMoE(
            cheb_order=fp_cfg.cheb_order,
            hidden_dim=args.hidden_dim,
            router_temperature=args.router_temperature,
            prototype_momentum=args.prototype_momentum,
            use_context=not args.no_mechanism_context,
            include_edge_surprise=fp_cfg.include_edge_surprise,
            include_node_activity=fp_cfg.include_node_activity,
            snapshot_relative_features=fp_cfg.snapshot_relative_features,
            relative_feature_mode=fp_cfg.relative_feature_mode,
            high_order_view=args.mechanism_high_order_view,
            mechanism_feature_mode=args.mechanism_feature_mode,
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
    if args.model_type == "context_rep_moe" and args.context_mode == "target_mean":
        target_mean, target_std = build_target_context_stats(test_xs)
        model.set_target_context(
            torch.from_numpy(target_mean).to(device),
            torch.from_numpy(target_std).to(device),
        )
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
        args.sampler,
        prototype_momentum=args.prototype_momentum,
        prototype_evidence_loss_lambda=args.prototype_evidence_loss_lambda,
        balanced_neg_ratio=args.balanced_neg_ratio,
    )

    scores = score_model(model, test_xs, device, args.chunk_size)
    labels = np.concatenate(test_ys)
    metrics = compute_metrics(labels, scores)
    router_stats = router_statistics(model, test_xs, device, args.chunk_size)
    expert_stats = mechanism_expert_statistics(model, test_xs, labels, device, args.chunk_size)
    evidence_stats = prototype_evidence_statistics(model, test_xs, device, args.chunk_size, labels)
    context_stats = context_representation_statistics(model)
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
    if router_stats is not None:
        result["test"] |= router_stats
    if expert_stats is not None:
        result["test"] |= expert_stats
    if evidence_stats is not None:
        result["test"] |= evidence_stats
    if context_stats is not None:
        result["test"] |= context_stats
    print(f"test={result['test']}", flush=True)

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"saved={args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
