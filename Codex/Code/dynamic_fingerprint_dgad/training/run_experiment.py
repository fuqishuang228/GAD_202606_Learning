from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dynamic_fingerprint_dgad.data.csv_dynamic_dataset import CSVDynamicDataset
from dynamic_fingerprint_dgad.fingerprints.extractor import DynamicFingerprintExtractor, FingerprintConfig
from dynamic_fingerprint_dgad.models.detector import (
    CrossAttentionDetector,
    DynamicFingerprintDetector,
    FeatureTokenDetector,
    FeatureTokenProtoMoE,
)
from dynamic_fingerprint_dgad.training.metrics import compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_datasets(data_dir: Path, names: list[str], num_snapshots: int, max_rows: int | None) -> list[CSVDynamicDataset]:
    datasets = []
    for name in names:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        datasets.append(CSVDynamicDataset(path, num_snapshots=num_snapshots, max_rows=max_rows))
    return datasets


def extract_dataset_features(dataset: CSVDynamicDataset, fp_cfg: FingerprintConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
    extractor = DynamicFingerprintExtractor(dataset.num_nodes, fp_cfg)
    xs, ys = [], []
    for snap in dataset.snapshots:
        print(
            f"extract dataset={dataset.name} snapshot={snap.index + 1}/{len(dataset.snapshots)} "
            f"edges={len(snap.edges)}",
            flush=True,
        )
        x = extractor.extract(snap.edges)
        xs.append(x)
        ys.append(snap.labels.astype(np.float32))
        extractor.update(snap.edges)
    return xs, ys


def normalize_features(train_xs: list[np.ndarray], groups: list[list[np.ndarray]]) -> tuple[list[np.ndarray], list[list[np.ndarray]], dict]:
    train_all = np.concatenate(train_xs, axis=0)
    mean = train_all.mean(axis=0, keepdims=True)
    std = train_all.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_norm = [((x - mean) / std).astype(np.float32) for x in train_xs]
    groups_norm = [[((x - mean) / std).astype(np.float32) for x in xs] for xs in groups]
    stats = {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}
    return train_norm, groups_norm, stats


def apply_feature_ablation(
    xs: list[np.ndarray],
    ablation: str,
    cheb_order: int,
    include_edge_surprise: bool,
    include_node_activity: bool,
) -> list[np.ndarray]:
    if ablation == "full":
        return xs
    out = [x.copy() for x in xs]
    static_end = 1 + cheb_order
    cursor = static_end
    temporal_static_cols = []
    if include_node_activity:
        temporal_static_cols.append(cursor)
        cursor += 1
    if include_edge_surprise:
        temporal_static_cols.append(cursor)
        cursor += 1
    delta_start = cursor
    delta_end = delta_start + 1 + cheb_order
    temporal_delta_cols = []
    if include_edge_surprise:
        temporal_delta_cols.append(delta_end)
    if ablation == "no_temporal":
        for x in out:
            if temporal_static_cols:
                x[:, temporal_static_cols] = 0.0
            x[:, delta_start:delta_end] = 0.0
            if temporal_delta_cols:
                x[:, temporal_delta_cols] = 0.0
    elif ablation == "no_structural":
        for x in out:
            x[:, :static_end] = 0.0
            x[:, delta_start:delta_end] = 0.0
    else:
        raise ValueError(f"unknown feature ablation: {ablation}")
    return out


def apply_group_feature_ablation(
    groups: list[list[np.ndarray]],
    ablation: str,
    cheb_order: int,
    include_edge_surprise: bool,
    include_node_activity: bool,
) -> list[list[np.ndarray]]:
    return [
        apply_feature_ablation(xs, ablation, cheb_order, include_edge_surprise, include_node_activity)
        for xs in groups
    ]


def split_train_val(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    val_ratio: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if val_ratio <= 0.0:
        return xs, ys, [], []
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys).astype(np.float32)
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for label_value in [0, 1]:
        idx = np.flatnonzero(y_all == label_value)
        rng.shuffle(idx)
        val_count = int(round(len(idx) * val_ratio))
        if label_value == 1 and len(idx) > 1:
            val_count = max(1, min(val_count, len(idx) - 1))
        val_idx.append(idx[:val_count])
        train_idx.append(idx[val_count:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    train_xs = [x_all[train_idx].astype(np.float32)]
    train_ys = [y_all[train_idx].astype(np.float32)]
    val_xs = [x_all[val_idx].astype(np.float32)] if len(val_idx) else []
    val_ys = [y_all[val_idx].astype(np.float32)] if len(val_idx) else []
    return train_xs, train_ys, val_xs, val_ys


def train_model(
    model: nn.Module,
    train_xs: list[np.ndarray],
    train_ys: list[np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    chunk_size: int,
    loss_type: str = "bce",
    rank_lambda: float = 1.0,
    rank_margin: float = 1.0,
    sampler: str = "snapshot",
    val_xs: list[np.ndarray] | None = None,
    val_ys: list[np.ndarray] | None = None,
    early_stop_metric: str = "auroc",
    patience: int = 0,
    center_loss_lambda: float = 0.0,
    center_margin: float = 0.5,
    warmup_epochs: int = 0,
) -> dict:
    labels_all = np.concatenate(train_ys)
    pos = float(labels_all.sum())
    neg = float(len(labels_all) - pos)
    pos_weight = None if sampler == "balanced" else torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    val_xs = val_xs or []
    val_ys = val_ys or []
    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    def forward_with_optional_embedding(batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        aux_loss = torch.tensor(0.0, device=batch_x.device)
        if center_loss_lambda > 0.0:
            try:
                out = model(batch_x, return_embedding=True, return_aux=True)
                if len(out) == 3:
                    logits, embedding, aux_loss = out
                else:
                    logits, embedding = out
                return logits, embedding, aux_loss
            except TypeError:
                pass
            try:
                logits, embedding = model(batch_x, return_embedding=True)
                return logits, embedding, aux_loss
            except TypeError:
                pass
        try:
            out = model(batch_x, return_aux=True)
            logits, aux_loss = out
            return logits, None, aux_loss
        except TypeError:
            return model(batch_x), None, aux_loss

    def normal_center_loss(embedding: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        if embedding is None or center_loss_lambda <= 0.0:
            return torch.tensor(0.0, device=labels.device)
        normal_mask = labels <= 0.5
        anomaly_mask = labels > 0.5
        if normal_mask.sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        z = F.normalize(embedding, dim=-1)
        center = F.normalize(z[normal_mask].mean(dim=0, keepdim=True), dim=-1)
        distance = 1.0 - torch.sum(z * center, dim=-1)
        normal_loss = distance[normal_mask].mean()
        if anomaly_mask.sum() == 0:
            return normal_loss
        anomaly_loss = torch.relu(center_margin - distance[anomaly_mask]).mean()
        return normal_loss + anomaly_loss

    model.train()
    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0:
            lr_scale = min(1.0, epoch / float(warmup_epochs))
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_scale
        total_loss = 0.0
        chunks = 0

        if sampler == "balanced":
            x_all = np.concatenate(train_xs, axis=0)
            y_all = labels_all.astype(np.float32)
            pos_idx_all = np.flatnonzero(y_all > 0.5)
            neg_idx_all = np.flatnonzero(y_all <= 0.5)
            steps = max(1, int(np.ceil(len(y_all) / chunk_size)))
            pos_batch = max(1, chunk_size // 2)
            neg_batch = max(1, chunk_size - pos_batch)
            for _ in range(steps):
                pos_sel = np.random.choice(pos_idx_all, size=pos_batch, replace=len(pos_idx_all) < pos_batch)
                neg_sel = np.random.choice(neg_idx_all, size=neg_batch, replace=len(neg_idx_all) < neg_batch)
                sel = np.concatenate([pos_sel, neg_sel])
                np.random.shuffle(sel)
                x = torch.from_numpy(x_all[sel]).to(device)
                y = torch.from_numpy(y_all[sel]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, embedding, aux_loss = forward_with_optional_embedding(x)
                loss = loss_fn(logits, y)
                loss = loss + center_loss_lambda * normal_center_loss(embedding, y)
                loss = loss + aux_loss
                if loss_type == "bce_rank":
                    pos_scores = logits[y > 0.5]
                    neg_scores = logits[y <= 0.5]
                    pair_count = min(pos_scores.numel(), neg_scores.numel())
                    pos_idx = torch.randint(pos_scores.numel(), (pair_count,), device=device)
                    neg_idx = torch.randint(neg_scores.numel(), (pair_count,), device=device)
                    rank_loss = torch.relu(rank_margin - pos_scores[pos_idx] + neg_scores[neg_idx]).mean()
                    loss = loss + rank_lambda * rank_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                chunks += 1
        else:
            order = list(range(len(train_xs)))
            random.shuffle(order)
            for idx in order:
                x_np, y_np = train_xs[idx], train_ys[idx]
                perm = np.random.permutation(len(x_np))
                for start in range(0, len(perm), chunk_size):
                    sel = perm[start:start + chunk_size]
                    x = torch.from_numpy(x_np[sel]).to(device)
                    y = torch.from_numpy(y_np[sel]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits, embedding, aux_loss = forward_with_optional_embedding(x)
                    loss = loss_fn(logits, y)
                    loss = loss + center_loss_lambda * normal_center_loss(embedding, y)
                    loss = loss + aux_loss
                    if loss_type == "bce_rank":
                        pos_scores = logits[y > 0.5]
                        neg_scores = logits[y <= 0.5]
                        if pos_scores.numel() > 0 and neg_scores.numel() > 0:
                            pair_count = min(pos_scores.numel(), neg_scores.numel())
                            pos_idx = torch.randint(pos_scores.numel(), (pair_count,), device=device)
                            neg_idx = torch.randint(neg_scores.numel(), (pair_count,), device=device)
                            rank_loss = torch.relu(rank_margin - pos_scores[pos_idx] + neg_scores[neg_idx]).mean()
                            loss = loss + rank_lambda * rank_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    chunks += 1
        epoch_loss = total_loss / max(chunks, 1)
        record = {"epoch": epoch, "loss": epoch_loss}
        if val_xs:
            val_scores = score_model(model, val_xs, device, chunk_size)
            val_labels = np.concatenate(val_ys)
            val_metrics = compute_metrics(val_labels, val_scores)
            record |= {f"val_{k}": v for k, v in val_metrics.items()}
            cur_metric = val_metrics.get(early_stop_metric, float("nan"))
            if np.isfinite(cur_metric) and cur_metric > best_metric:
                best_metric = cur_metric
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(
                f"epoch={epoch:03d} loss={epoch_loss:.6f} "
                f"val_auroc={val_metrics['auroc']:.6f} val_auprc={val_metrics['auprc']:.6f} "
                f"best_epoch={best_epoch:03d}",
                flush=True,
            )
            history.append(record)
            if patience > 0 and stale_epochs >= patience:
                print(f"early_stop epoch={epoch:03d} stale_epochs={stale_epochs}", flush=True)
                break
        else:
            if epoch == epochs:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            print(f"epoch={epoch:03d} loss={epoch_loss:.6f}", flush=True)
            history.append(record)
    model.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_metric": None if best_metric == float("-inf") else float(best_metric),
        "history": history,
    }


@torch.no_grad()
def score_model(model: nn.Module, xs: list[np.ndarray], device: torch.device, chunk_size: int) -> np.ndarray:
    model.eval()
    scores = []
    for x_np in xs:
        cur = []
        for start in range(0, len(x_np), chunk_size):
            x = torch.from_numpy(x_np[start:start + chunk_size]).to(device)
            cur.append(torch.sigmoid(model(x)).detach().cpu().numpy())
        scores.append(np.concatenate(cur))
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data"))
    parser.add_argument("--source", nargs="+", default=["MOOC", "Wikipedia"])
    parser.add_argument("--target", nargs="+", default=["uci", "btc_otc"])
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-snapshots", type=int, default=50)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--cheb-order", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--model-type",
        choices=["edge_transformer", "feature_token", "feature_token_moe", "feature_token_proto_moe", "cross_attention"],
        default="edge_transformer",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--loss-type", choices=["bce", "bce_rank"], default="bce")
    parser.add_argument("--rank-lambda", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--center-loss-lambda", type=float, default=0.0)
    parser.add_argument("--center-margin", type=float, default=0.5)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-top-k", type=int, default=0)
    parser.add_argument("--router-entropy-lambda", type=float, default=0.0)
    parser.add_argument("--load-balance-lambda", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--sampler", choices=["snapshot", "balanced"], default="snapshot")
    parser.add_argument("--feature-ablation", choices=["full", "no_temporal", "no_structural"], default="full")
    parser.add_argument("--include-edge-surprise", action="store_true")
    parser.add_argument("--include-node-activity", action="store_true")
    parser.add_argument("--relative-delta", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", choices=["auroc", "auprc"], default="auroc")
    parser.add_argument("--patience", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"device={device}", flush=True)

    fp_cfg = FingerprintConfig(
        cheb_order=args.cheb_order,
        history_window=args.history_window,
        include_edge_surprise=args.include_edge_surprise,
        include_node_activity=args.include_node_activity,
        relative_delta=args.relative_delta,
    )
    source_sets = load_datasets(args.data_dir, args.source, args.num_snapshots, args.max_rows)
    target_sets = load_datasets(args.data_dir, args.target, args.num_snapshots, args.max_rows)
    print("sources:", [d.summary() for d in source_sets], flush=True)
    print("targets:", [d.summary() for d in target_sets], flush=True)

    source_features = []
    source_labels = []
    for ds in source_sets:
        xs, ys = extract_dataset_features(ds, fp_cfg)
        source_features.extend(xs)
        source_labels.extend(ys)

    target_features_by_dataset = []
    target_labels_by_dataset = []
    for ds in target_sets:
        xs, ys = extract_dataset_features(ds, fp_cfg)
        target_features_by_dataset.append(xs)
        target_labels_by_dataset.append(ys)

    train_features, train_labels, val_features, val_labels = split_train_val(
        source_features,
        source_labels,
        args.val_ratio,
        args.seed,
    )
    groups_to_normalize = target_features_by_dataset + ([val_features] if val_features else [])
    train_features, normalized_groups, stats = normalize_features(train_features, groups_to_normalize)
    target_features_by_dataset = normalized_groups[: len(target_features_by_dataset)]
    val_features = normalized_groups[len(target_features_by_dataset)] if val_features else []

    train_anomalies = int(np.concatenate(train_labels).sum())
    val_anomalies = int(np.concatenate(val_labels).sum()) if val_labels else 0
    print(
        f"source split: train_edges={sum(len(y) for y in train_labels)} train_anom={train_anomalies} "
        f"val_edges={sum(len(y) for y in val_labels)} val_anom={val_anomalies}",
        flush=True,
    )

    train_features = apply_feature_ablation(
        train_features,
        args.feature_ablation,
        fp_cfg.cheb_order,
        fp_cfg.include_edge_surprise,
        fp_cfg.include_node_activity,
    )
    val_features = apply_feature_ablation(
        val_features,
        args.feature_ablation,
        fp_cfg.cheb_order,
        fp_cfg.include_edge_surprise,
        fp_cfg.include_node_activity,
    ) if val_features else []
    target_features_by_dataset = apply_group_feature_ablation(
        target_features_by_dataset,
        args.feature_ablation,
        fp_cfg.cheb_order,
        fp_cfg.include_edge_surprise,
        fp_cfg.include_node_activity,
    )
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
    elif args.model_type == "feature_token_proto_moe":
        model = FeatureTokenProtoMoE(
            cheb_order=fp_cfg.cheb_order,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_experts=args.num_experts,
            router_temperature=args.router_temperature,
            router_top_k=args.router_top_k,
            router_entropy_lambda=args.router_entropy_lambda,
            load_balance_lambda=args.load_balance_lambda,
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
    training_info = train_model(
        model,
        train_features,
        train_labels,
        device,
        args.epochs,
        args.lr,
        args.chunk_size,
        args.loss_type,
        args.rank_lambda,
        args.rank_margin,
        args.sampler,
        val_features,
        val_labels,
        args.early_stop_metric,
        args.patience,
        args.center_loss_lambda,
        args.center_margin,
        args.warmup_epochs,
    )

    results = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "feature_stats": stats,
        "source": [d.summary() for d in source_sets],
        "source_split": {
            "train_edges": int(sum(len(y) for y in train_labels)),
            "train_anomalies": train_anomalies,
            "val_edges": int(sum(len(y) for y in val_labels)),
            "val_anomalies": val_anomalies,
        },
        "training": training_info,
        "target": {},
    }
    for ds, xs, ys in zip(target_sets, target_features_by_dataset, target_labels_by_dataset):
        scores = score_model(model, xs, device, args.chunk_size)
        labels = np.concatenate(ys)
        metrics = compute_metrics(labels, scores)
        results["target"][ds.name] = metrics | {
            "rows": int(len(labels)),
            "anomalies": int(labels.sum()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        print(f"target={ds.name} {results['target'][ds.name]}", flush=True)

    result_path = args.out_dir / "metrics.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"saved={result_path}", flush=True)


if __name__ == "__main__":
    main()
