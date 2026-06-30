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
    ContextRepresentationMoE,
    CrossAttentionDetector,
    DynamicFingerprintDetector,
    FeatureTokenDetector,
    FeatureTokenProtoMoE,
    MechanismContextSoftMoE,
    MechanismPrototypeRouterMoE,
    MechanismSoftMoE,
)
from dynamic_fingerprint_dgad.training.metrics import compute_metrics


ATLAS_LOCAL_K2_FEATURE_NAMES = [
    "HN_1_1", "HN_1_2", "HN_2_1", "HN_2_2",
    "RA_1_1", "RA_1_2", "RA_2_1", "RA_2_2",
    "LC_1_1", "LC_1_2", "LC_2_1", "LC_2_2",
    "CP1", "CP2", "CP3", "CP4",
    "d_HN_1_1", "d_HN_1_2", "d_HN_2_1", "d_HN_2_2",
    "d_RA_1_1", "d_RA_1_2", "d_RA_2_1", "d_RA_2_2",
    "d_LC_1_1", "d_LC_1_2", "d_LC_2_1", "d_LC_2_2",
    "d_CP1", "d_CP2", "d_CP3", "d_CP4",
]

ATLAS_LOCAL_K2_FAST_FEATURE_NAMES = [
    "HN_1_1", "HN_1_2", "HN_2_1", "HN_2_2",
    "RA_1_1", "RA_1_2", "RA_2_1", "RA_2_2",
    "LC_1_1",
    "CP1", "CP2", "CP3", "CP4",
    "d_HN_1_1", "d_HN_1_2", "d_HN_2_1", "d_HN_2_2",
    "d_RA_1_1", "d_RA_1_2", "d_RA_2_1", "d_RA_2_2",
    "d_LC_1_1",
    "d_CP1", "d_CP2", "d_CP3", "d_CP4",
]

ATLAS_LOCAL_K1_FEATURE_NAMES = [
    "HN_1_1", "RA_1_1", "LC_1_1",
    "CP1", "CP2", "CP3", "CP4",
    "d_HN_1_1", "d_RA_1_1", "d_LC_1_1",
    "d_CP1", "d_CP2", "d_CP3", "d_CP4",
]


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


def feature_diagnostics(xs: list[np.ndarray]) -> dict:
    all_x = np.concatenate(xs, axis=0)
    return {
        "feature_mean": all_x.mean(axis=0).reshape(-1).tolist(),
        "feature_std": all_x.std(axis=0).reshape(-1).tolist(),
        "feature_min": all_x.min(axis=0).reshape(-1).tolist(),
        "feature_max": all_x.max(axis=0).reshape(-1).tolist(),
    }


def feature_names(feature_dim: int, fingerprint_variant: str, feature_view: str) -> list[str]:
    if fingerprint_variant == "atlas_local_k2":
        return ATLAS_LOCAL_K2_FEATURE_NAMES
    if fingerprint_variant == "atlas_local_k2_fast":
        return ATLAS_LOCAL_K2_FAST_FEATURE_NAMES
    if fingerprint_variant == "atlas_local_k1":
        return ATLAS_LOCAL_K1_FEATURE_NAMES
    if feature_view == "high_order_only":
        return ["CP3", "CP4", "d_CP3", "d_CP4"]
    if feature_view == "high_order_residual":
        return ["CP3", "CP4", "d_CP3", "d_CP4", "r_fast", "r_slow", "r_fast_slow", "velocity", "acceleration"]
    if feature_view == "high_order_multiview":
        return [
            "CP3", "CP4", "d_CP3", "d_CP4", "high_mean", "d_high_mean",
            "high_gap", "d_high_gap", "high_low_contrast", "d_high_low_contrast",
        ]
    return [f"feature_{idx}" for idx in range(feature_dim)]


def transform_feature_view(
    xs: list[np.ndarray],
    feature_view: str,
    edge_xs: list[np.ndarray] | None = None,
    num_nodes: int | None = None,
    residual_alpha_fast: float = 0.7,
    residual_alpha_slow: float = 0.95,
) -> list[np.ndarray]:
    if feature_view == "full":
        return xs
    if feature_view == "high_order_residual":
        if edge_xs is None or num_nodes is None:
            raise ValueError("high_order_residual requires edge_xs and num_nodes for causal endpoint state")
        if not (0.0 <= residual_alpha_fast < 1.0):
            raise ValueError(f"residual_alpha_fast must be in [0, 1), got {residual_alpha_fast}")
        if not (0.0 <= residual_alpha_slow < 1.0):
            raise ValueError(f"residual_alpha_slow must be in [0, 1), got {residual_alpha_slow}")
        return transform_high_order_residual_view(
            xs,
            edge_xs,
            num_nodes,
            alpha_fast=residual_alpha_fast,
            alpha_slow=residual_alpha_slow,
        )
    out = []
    for x in xs:
        if x.shape[1] < 10:
            raise ValueError(f"{feature_view} expects at least 10 raw fingerprint dims, got {x.shape[1]}")
        cp1 = x[:, 1:2]
        cp2 = x[:, 2:3]
        cp3 = x[:, 3:4]
        cp4 = x[:, 4:5]
        dcp1 = x[:, 6:7]
        dcp2 = x[:, 7:8]
        dcp3 = x[:, 8:9]
        dcp4 = x[:, 9:10]
        if feature_view == "high_order_only":
            view = np.concatenate([cp3, cp4, dcp3, dcp4], axis=1)
        elif feature_view == "high_order_multiview":
            high_mean = 0.5 * (cp3 + cp4)
            d_high_mean = 0.5 * (dcp3 + dcp4)
            high_gap = cp4 - cp3
            d_high_gap = dcp4 - dcp3
            high_low_contrast = high_mean - 0.5 * (cp1 + cp2)
            d_high_low_contrast = d_high_mean - 0.5 * (dcp1 + dcp2)
            view = np.concatenate(
                [
                    cp3,
                    cp4,
                    dcp3,
                    dcp4,
                    high_mean,
                    d_high_mean,
                    high_gap,
                    d_high_gap,
                    high_low_contrast,
                    d_high_low_contrast,
                ],
                axis=1,
            )
        else:
            raise ValueError(f"unknown feature_view: {feature_view}")
        out.append(view.astype(np.float32))
    return out


def transform_high_order_residual_view(
    xs: list[np.ndarray],
    edge_xs: list[np.ndarray],
    num_nodes: int,
    alpha_fast: float,
    alpha_slow: float,
) -> list[np.ndarray]:
    """Causal endpoint-level surprise over high-order CP3/CP4 features.

    Unseen nodes start from zero state. For each edge, features are computed from
    endpoint history first, then both endpoint states are updated with current h.
    """
    if len(xs) != len(edge_xs):
        raise ValueError(f"xs and edge_xs length mismatch: {len(xs)} vs {len(edge_xs)}")
    fast = np.zeros((num_nodes, 2), dtype=np.float32)
    slow = np.zeros((num_nodes, 2), dtype=np.float32)
    prev_h = np.zeros((num_nodes, 2), dtype=np.float32)
    prev_delta = np.zeros((num_nodes, 2), dtype=np.float32)
    out = []
    for x, edges in zip(xs, edge_xs):
        if x.shape[1] < 10:
            raise ValueError(f"high_order_residual expects at least 10 raw fingerprint dims, got {x.shape[1]}")
        h_all = x[:, [3, 4]].astype(np.float32)
        d_high = x[:, [8, 9]].astype(np.float32)
        residual = np.zeros((len(x), 5), dtype=np.float32)
        for idx, ((u, v), h) in enumerate(zip(edges.astype(np.int64), h_all)):
            u = int(u)
            v = int(v)
            fast_uv = 0.5 * (fast[u] + fast[v])
            slow_uv = 0.5 * (slow[u] + slow[v])
            prev_uv = 0.5 * (prev_h[u] + prev_h[v])
            prev_delta_uv = 0.5 * (prev_delta[u] + prev_delta[v])
            current_delta = h - prev_uv
            residual[idx, 0] = np.linalg.norm(h - fast_uv)
            residual[idx, 1] = np.linalg.norm(h - slow_uv)
            residual[idx, 2] = np.linalg.norm(fast_uv - slow_uv)
            residual[idx, 3] = np.linalg.norm(current_delta)
            residual[idx, 4] = np.linalg.norm(current_delta - prev_delta_uv)
            for node in (u, v):
                fast[node] = alpha_fast * fast[node] + (1.0 - alpha_fast) * h
                slow[node] = alpha_slow * slow[node] + (1.0 - alpha_slow) * h
                prev_h[node] = h
                prev_delta[node] = current_delta
        out.append(np.concatenate([h_all, d_high, residual], axis=1).astype(np.float32))
    return out


def transform_group_feature_view(
    groups: list[list[np.ndarray]],
    feature_view: str,
    edge_groups: list[list[np.ndarray]] | None = None,
    num_nodes: list[int] | None = None,
    residual_alpha_fast: float = 0.7,
    residual_alpha_slow: float = 0.95,
) -> list[list[np.ndarray]]:
    if feature_view == "high_order_residual":
        if edge_groups is None or num_nodes is None:
            raise ValueError("high_order_residual group transform requires edge_groups and num_nodes")
        return [
            transform_feature_view(xs, feature_view, edges, n, residual_alpha_fast, residual_alpha_slow)
            for xs, edges, n in zip(groups, edge_groups, num_nodes)
        ]
    return [transform_feature_view(xs, feature_view) for xs in groups]


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
    prototype_momentum: float = 0.9,
    prototype_evidence_loss_lambda: float = 0.0,
    balanced_neg_ratio: float = 1.0,
) -> dict:
    if balanced_neg_ratio <= 0.0:
        raise ValueError(f"balanced_neg_ratio must be positive, got {balanced_neg_ratio}")
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

    def forward_with_optional_embedding(batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        aux_loss = torch.tensor(0.0, device=batch_x.device)
        evidence = None
        if prototype_evidence_loss_lambda > 0.0:
            try:
                out = model(batch_x, return_aux=True, return_evidence=True)
                if len(out) == 3:
                    logits, aux_loss, evidence = out
                else:
                    logits, evidence = out
                return logits, None, aux_loss, evidence
            except TypeError:
                pass
        if center_loss_lambda > 0.0:
            try:
                out = model(batch_x, return_embedding=True, return_aux=True)
                if len(out) == 3:
                    logits, embedding, aux_loss = out
                else:
                    logits, embedding = out
                return logits, embedding, aux_loss, evidence
            except TypeError:
                pass
            try:
                logits, embedding = model(batch_x, return_embedding=True)
                return logits, embedding, aux_loss, evidence
            except TypeError:
                pass
        try:
            out = model(batch_x, return_aux=True)
            logits, aux_loss = out
            return logits, None, aux_loss, evidence
        except TypeError:
            return model(batch_x), None, aux_loss, evidence

    def prototype_evidence_loss(evidence: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        if evidence is None or prototype_evidence_loss_lambda <= 0.0:
            return torch.tensor(0.0, device=labels.device)
        targets = labels.unsqueeze(1).expand_as(evidence)
        return F.binary_cross_entropy_with_logits(evidence, targets)

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
    if hasattr(model, "prototype_momentum"):
        model.prototype_momentum = prototype_momentum
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
            pos_batch = max(1, int(round(chunk_size / (1.0 + balanced_neg_ratio))))
            pos_batch = min(pos_batch, chunk_size - 1) if chunk_size > 1 else 1
            neg_batch = max(1, chunk_size - pos_batch)
            for _ in range(steps):
                pos_sel = np.random.choice(pos_idx_all, size=pos_batch, replace=len(pos_idx_all) < pos_batch)
                neg_sel = np.random.choice(neg_idx_all, size=neg_batch, replace=len(neg_idx_all) < neg_batch)
                sel = np.concatenate([pos_sel, neg_sel])
                np.random.shuffle(sel)
                x = torch.from_numpy(x_all[sel]).to(device)
                y = torch.from_numpy(y_all[sel]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, embedding, aux_loss, evidence = forward_with_optional_embedding(x)
                loss = loss_fn(logits, y)
                loss = loss + center_loss_lambda * normal_center_loss(embedding, y)
                loss = loss + prototype_evidence_loss_lambda * prototype_evidence_loss(evidence, y)
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
                if hasattr(model, "update_prototypes"):
                    model.update_prototypes(x, y)
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
                    logits, embedding, aux_loss, evidence = forward_with_optional_embedding(x)
                    loss = loss_fn(logits, y)
                    loss = loss + center_loss_lambda * normal_center_loss(embedding, y)
                    loss = loss + prototype_evidence_loss_lambda * prototype_evidence_loss(evidence, y)
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
                    if hasattr(model, "update_prototypes"):
                        model.update_prototypes(x, y)
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
        "sampler": sampler,
        "balanced_neg_ratio": float(balanced_neg_ratio),
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


@torch.no_grad()
def router_statistics(model: nn.Module, xs: list[np.ndarray], device: torch.device, chunk_size: int) -> dict | None:
    model.eval()
    weights = []
    for x_np in xs:
        for start in range(0, len(x_np), chunk_size):
            x = torch.from_numpy(x_np[start:start + chunk_size]).to(device)
            try:
                out = model(x, return_router=True)
            except TypeError:
                return None
            if not isinstance(out, tuple) or len(out) < 2:
                return None
            weights.append(out[-1].detach().cpu())
    if not weights:
        return None
    w = torch.cat(weights, dim=0)
    entropy = -(w * torch.log(w.clamp_min(1e-8))).sum(dim=1)
    return {
        "router_mean_used": w.mean(dim=0).tolist(),
        "router_std_used": w.std(dim=0, unbiased=False).tolist(),
        "router_entropy_used_mean": float(entropy.mean().item()),
    }


@torch.no_grad()
def mechanism_expert_statistics(
    model: nn.Module,
    xs: list[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> dict | None:
    if not hasattr(model, "expert_logits"):
        return None
    model.eval()
    expert_logits = []
    for x_np in xs:
        for start in range(0, len(x_np), chunk_size):
            x = torch.from_numpy(x_np[start:start + chunk_size]).to(device)
            expert_logits.append(model.expert_logits(x).detach().cpu())
    if not expert_logits:
        return None
    logits = torch.cat(expert_logits, dim=0)
    scores = torch.sigmoid(logits).numpy()
    all_names = ["local_structure", "low_order_spectral", "high_order_spectral", "temporal_delta"]
    if hasattr(model, "expert_names"):
        names = list(model.expert_names)
    else:
        names = all_names[: scores.shape[1]]
    per_expert = {
        name: compute_metrics(labels, scores[:, idx])
        for idx, name in enumerate(names[: scores.shape[1]])
    }
    return {
        "expert_logit_mean": logits.mean(dim=0).tolist(),
        "expert_logit_std": logits.std(dim=0, unbiased=False).tolist(),
        "per_expert_metrics": per_expert,
    }


@torch.no_grad()
def prototype_evidence_statistics(
    model: nn.Module,
    xs: list[np.ndarray],
    device: torch.device,
    chunk_size: int,
    labels: np.ndarray | None = None,
) -> dict | None:
    if not hasattr(model, "prototype_evidence"):
        return None
    model.eval()
    evidences = []
    for x_np in xs:
        for start in range(0, len(x_np), chunk_size):
            x = torch.from_numpy(x_np[start:start + chunk_size]).to(device)
            evidences.append(model.prototype_evidence(x).detach().cpu())
    if not evidences:
        return None
    evidence = torch.cat(evidences, dim=0)
    stats = {
        "prototype_evidence_mean": evidence.mean(dim=0).tolist(),
        "prototype_evidence_std": evidence.std(dim=0, unbiased=False).tolist(),
    }
    if labels is not None:
        labels_tensor = torch.from_numpy(labels.astype(np.float32))
        normal_mask = labels_tensor <= 0.5
        anomaly_mask = labels_tensor > 0.5
        if normal_mask.any():
            stats["prototype_evidence_normal_mean"] = evidence[normal_mask].mean(dim=0).tolist()
        if anomaly_mask.any():
            stats["prototype_evidence_anomaly_mean"] = evidence[anomaly_mask].mean(dim=0).tolist()
    if hasattr(model, "normal_prototypes") and hasattr(model, "anomaly_prototypes"):
        stats["prototype_norms_normal"] = model.normal_prototypes.detach().cpu().norm(dim=-1).tolist()
        stats["prototype_norms_anomaly"] = model.anomaly_prototypes.detach().cpu().norm(dim=-1).tolist()
    return stats


def build_target_context_stats(target_xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    target_all = np.concatenate(target_xs, axis=0)
    mean = target_all.mean(axis=0).astype(np.float32)
    std = target_all.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def context_representation_statistics(model: nn.Module) -> dict | None:
    if not hasattr(model, "context_diagnostics"):
        return None
    return model.context_diagnostics()


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
        choices=[
            "edge_transformer",
            "feature_token",
            "feature_token_moe",
            "feature_token_proto_moe",
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
    parser.add_argument("--center-loss-lambda", type=float, default=0.0)
    parser.add_argument("--center-margin", type=float, default=0.5)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-top-k", type=int, default=0)
    parser.add_argument("--router-entropy-lambda", type=float, default=0.0)
    parser.add_argument("--load-balance-lambda", type=float, default=0.0)
    parser.add_argument("--mechanism-router-mode", choices=["learned", "uniform", "global"], default="learned")
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument("--prototype-evidence-loss-lambda", type=float, default=0.0)
    parser.add_argument("--mechanism-high-order-view", choices=["raw", "multiview"], default="raw")
    parser.add_argument("--no-mechanism-context", action="store_true")
    parser.add_argument("--context-mode", choices=["none", "target_mean"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--sampler", choices=["snapshot", "balanced"], default="snapshot")
    parser.add_argument("--balanced-neg-ratio", type=float, default=1.0)
    parser.add_argument("--feature-ablation", choices=["full", "no_temporal", "no_structural"], default="full")
    parser.add_argument("--feature-view", choices=["full", "high_order_only", "high_order_multiview", "high_order_residual"], default="full")
    parser.add_argument("--residual-alpha-fast", type=float, default=0.7)
    parser.add_argument("--residual-alpha-slow", type=float, default=0.95)
    parser.add_argument("--include-edge-surprise", action="store_true")
    parser.add_argument("--include-node-activity", action="store_true")
    parser.add_argument("--relative-delta", action="store_true")
    parser.add_argument("--snapshot-relative-features", action="store_true")
    parser.add_argument("--relative-feature-mode", choices=["none", "all", "local"], default="none")
    parser.add_argument("--fingerprint-variant", choices=["raw10", "atlas_local_k2", "atlas_local_k2_fast", "atlas_local_k1"], default="raw10")
    parser.add_argument("--mechanism-feature-mode", choices=["raw10", "atlas_local_k2", "atlas_local_k2_fast", "atlas_local_k1"], default="raw10")
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
        snapshot_relative_features=args.snapshot_relative_features,
        relative_feature_mode=args.relative_feature_mode,
        fingerprint_variant=args.fingerprint_variant,
    )
    source_sets = load_datasets(args.data_dir, args.source, args.num_snapshots, args.max_rows)
    target_sets = load_datasets(args.data_dir, args.target, args.num_snapshots, args.max_rows)
    print("sources:", [d.summary() for d in source_sets], flush=True)
    print("targets:", [d.summary() for d in target_sets], flush=True)

    source_features = []
    source_labels = []
    for ds in source_sets:
        xs, ys = extract_dataset_features(ds, fp_cfg)
        xs = transform_feature_view(
            xs,
            args.feature_view,
            [snap.edges for snap in ds.snapshots],
            ds.num_nodes,
            args.residual_alpha_fast,
            args.residual_alpha_slow,
        )
        source_features.extend(xs)
        source_labels.extend(ys)

    target_features_by_dataset = []
    target_labels_by_dataset = []
    for ds in target_sets:
        xs, ys = extract_dataset_features(ds, fp_cfg)
        xs = transform_feature_view(
            xs,
            args.feature_view,
            [snap.edges for snap in ds.snapshots],
            ds.num_nodes,
            args.residual_alpha_fast,
            args.residual_alpha_slow,
        )
        target_features_by_dataset.append(xs)
        target_labels_by_dataset.append(ys)

    feature_dim = int(source_features[0].shape[1]) if source_features else fp_cfg.feature_dim
    print(f"feature_view={args.feature_view} feature_dim={feature_dim}", flush=True)

    train_features, train_labels, val_features, val_labels = split_train_val(
        source_features,
        source_labels,
        args.val_ratio,
        args.seed,
    )
    feature_diag = {
        "feature_names": feature_names(feature_dim, args.fingerprint_variant, args.feature_view),
        "source_train": feature_diagnostics(train_features),
    }
    if val_features:
        feature_diag["source_val"] = feature_diagnostics(val_features)
    for ds, xs in zip(target_sets, target_features_by_dataset):
        feature_diag[f"target_{ds.name}"] = feature_diagnostics(xs)

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
    if args.feature_view == "full":
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
    elif args.feature_ablation != "full":
        raise ValueError("--feature-ablation other than full is only supported with --feature-view full")
    if args.model_type == "context_rep_moe":
        if feature_dim != 10:
            raise ValueError(f"context_rep_moe expects raw10 full features, got feature_dim={feature_dim}")
        if fp_cfg.include_edge_surprise or fp_cfg.include_node_activity:
            raise ValueError("context_rep_moe does not support edge_surprise or node_activity")
        if args.context_mode == "target_mean" and len(target_features_by_dataset) != 1:
            raise ValueError("context-mode=target_mean currently expects exactly one target dataset")
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
        target_mean, target_std = build_target_context_stats(target_features_by_dataset[0])
        model.set_target_context(
            torch.from_numpy(target_mean).to(device),
            torch.from_numpy(target_std).to(device),
        )
        print(
            "target_context constructed from unlabeled normalized target features "
            f"mean_dim={target_mean.shape[0]} std_dim={target_std.shape[0]}",
            flush=True,
        )
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
        args.prototype_momentum,
        args.prototype_evidence_loss_lambda,
        args.balanced_neg_ratio,
    )

    results = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "feature_stats": stats,
        "feature_diagnostics": feature_diag,
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
    if val_features:
        val_scores = score_model(model, val_features, device, args.chunk_size)
        val_label_array = np.concatenate(val_labels)
        validation = compute_metrics(val_label_array, val_scores) | {
            "rows": int(len(val_label_array)),
            "anomalies": int(val_label_array.sum()),
            "score_mean": float(val_scores.mean()),
            "score_std": float(val_scores.std()),
        }
        router_stats = router_statistics(model, val_features, device, args.chunk_size)
        expert_stats = mechanism_expert_statistics(model, val_features, val_label_array, device, args.chunk_size)
        evidence_stats = prototype_evidence_statistics(model, val_features, device, args.chunk_size, val_label_array)
        context_stats = context_representation_statistics(model)
        if router_stats is not None:
            validation |= router_stats
        if expert_stats is not None:
            validation |= expert_stats
        if evidence_stats is not None:
            validation |= evidence_stats
        if context_stats is not None:
            validation |= context_stats
        results["validation"] = validation
        print(f"validation={validation}", flush=True)
    for ds, xs, ys in zip(target_sets, target_features_by_dataset, target_labels_by_dataset):
        scores = score_model(model, xs, device, args.chunk_size)
        labels = np.concatenate(ys)
        metrics = compute_metrics(labels, scores)
        router_stats = router_statistics(model, xs, device, args.chunk_size)
        expert_stats = mechanism_expert_statistics(model, xs, labels, device, args.chunk_size)
        evidence_stats = prototype_evidence_statistics(model, xs, device, args.chunk_size, labels)
        context_stats = context_representation_statistics(model)
        results["target"][ds.name] = metrics | {
            "rows": int(len(labels)),
            "anomalies": int(labels.sum()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        if router_stats is not None:
            results["target"][ds.name] |= router_stats
        if expert_stats is not None:
            results["target"][ds.name] |= expert_stats
        if evidence_stats is not None:
            results["target"][ds.name] |= evidence_stats
        if context_stats is not None:
            results["target"][ds.name] |= context_stats
        print(f"target={ds.name} {results['target'][ds.name]}", flush=True)

    result_path = args.out_dir / "metrics.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"saved={result_path}", flush=True)


if __name__ == "__main__":
    main()
