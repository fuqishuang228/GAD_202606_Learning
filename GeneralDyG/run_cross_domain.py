from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model.CensNet import CensNet
from model.Combine import CombinedModel
from model.Transformer import TransformerBinaryClassifier


FIELDS = ["nodefeatures", "edgefeatures", "labels", "Tmats", "adjs", "eadjs"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def take_indices(data: dict, indices: np.ndarray) -> dict:
    out = {}
    for key in FIELDS:
        values = data[key]
        if isinstance(values, np.ndarray):
            out[key] = values[indices]
        else:
            out[key] = [values[int(i)] for i in indices]
    return out


def offset_feature_ids(data: dict, node_offset: int, edge_offset: int) -> tuple[dict, int, int]:
    node_features = []
    max_node = node_offset
    for arr in data["nodefeatures"]:
        cur = np.asarray(arr, dtype=np.int64) + node_offset
        node_features.append(cur)
        if len(cur):
            max_node = max(max_node, int(cur.max()) + 1)

    edge_features = []
    max_edge = edge_offset
    for arr in data["edgefeatures"]:
        cur = np.asarray(arr, dtype=np.int64) + edge_offset
        edge_features.append(cur)
        if len(cur):
            max_edge = max(max_edge, int(cur.max()) + 1)

    out = {key: data[key] for key in FIELDS}
    out["nodefeatures"] = np.asarray(node_features, dtype=object)
    out["edgefeatures"] = np.asarray(edge_features, dtype=object)
    return out, max_node, max_edge


def concat_data(parts: list[dict]) -> dict:
    out = {}
    for key in FIELDS:
        values = [part[key] for part in parts]
        if isinstance(values[0], np.ndarray):
            out[key] = np.concatenate(values)
        else:
            merged = []
            for value in values:
                merged.extend(value)
            out[key] = merged
    return out


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    labels = labels.astype(int)
    for label_value in [0, 1]:
        idx = np.flatnonzero(labels == label_value)
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
    return train_idx, val_idx


def maybe_subsample_indices(labels: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    all_idx = np.arange(len(labels))
    if max_samples is None or len(all_idx) <= max_samples:
        return all_idx
    train_idx, val_idx = stratified_split(labels, max_samples / len(all_idx), seed)
    idx = np.concatenate([train_idx[:0], val_idx])
    if len(idx) > max_samples:
        idx = idx[:max_samples]
    return idx


class CrossDomainDygDataset(Dataset):
    def __init__(self, data: dict, input_dim: int, seed: int):
        self.node_features = data["nodefeatures"]
        self.edge_features = data["edgefeatures"]
        self.labels = np.asarray(data["labels"], dtype=np.float32)
        self.Tmats = data["Tmats"]
        self.adjs = data["adjs"]
        self.eadjs = data["eadjs"]
        self.max_edge_len = max(len(arr) for arr in self.edge_features)

        max_node_id = max(int(np.asarray(arr, dtype=np.int64).max()) for arr in self.node_features if len(arr))
        max_edge_id = max(int(np.asarray(arr, dtype=np.int64).max()) for arr in self.edge_features if len(arr))
        rng = np.random.default_rng(seed)
        self.Nfeatures = rng.uniform(0.0, 1.0, size=(max_node_id + 1, input_dim)).astype(np.float32)
        self.Efeatures = rng.uniform(0.0, 1.0, size=(max_edge_id + 1, input_dim)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, item: int) -> dict:
        node_idx = np.asarray(self.node_features[item], dtype=np.int64)
        edge_idx = np.asarray(self.edge_features[item], dtype=np.int64)
        edge_values = self.Efeatures[edge_idx]
        edge_pad = np.zeros((self.max_edge_len, self.Efeatures.shape[1]), dtype=np.float32)
        edge_pad[: len(edge_idx)] = edge_values
        mask_edge = np.ones((self.max_edge_len,), dtype=np.float32)
        mask_edge[: len(edge_idx)] = 0.0
        return {
            "input_nodes_feature": torch.from_numpy(self.Nfeatures[node_idx]),
            "input_edges_feature": torch.from_numpy(edge_values),
            "input_edges_pad": torch.from_numpy(edge_pad),
            "labels": torch.tensor(self.labels[item], dtype=torch.float32),
            "Tmats": self.Tmats[item],
            "adjs": self.adjs[item],
            "eadjs": self.eadjs[item],
            "mask_edge": torch.from_numpy(mask_edge),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "input_nodes_feature": [b["input_nodes_feature"] for b in batch],
        "input_edges_feature": [b["input_edges_feature"] for b in batch],
        "input_edges_pad": torch.stack([b["input_edges_pad"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "Tmats": [b["Tmats"] for b in batch],
        "adjs": [b["adjs"] for b in batch],
        "eadjs": [b["eadjs"] for b in batch],
        "mask_edge": torch.stack([b["mask_edge"] for b in batch], dim=0),
    }


def build_source_data(data_dir: Path, names: list[str], max_samples: int | None, seed: int) -> dict:
    parts = []
    node_offset = 0
    edge_offset = 0
    for name in names:
        data = load_pkl(data_dir / f"{name}.pkl")
        idx = maybe_subsample_indices(np.asarray(data["labels"]), max_samples, seed) if max_samples else None
        if idx is not None:
            data = take_indices(data, idx)
        data, node_offset, edge_offset = offset_feature_ids(data, node_offset, edge_offset)
        parts.append(data)
    return concat_data(parts)


def build_target_data(data_dir: Path, name: str, max_samples: int | None, seed: int) -> dict:
    data = load_pkl(data_dir / f"{name}.pkl")
    if max_samples:
        idx = maybe_subsample_indices(np.asarray(data["labels"]), max_samples, seed)
        data = take_indices(data, idx)
    data, _, _ = offset_feature_ids(data, 0, 0)
    return data


def make_model(config: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    gnn = CensNet(config.input_dim, config.drop_out)
    transformer = TransformerBinaryClassifier(config, device, hidden_size=config.hidden_dim)
    return CombinedModel(gnn, transformer).to(device)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "input_nodes_feature": [x.to(device) for x in batch["input_nodes_feature"]],
        "input_edges_feature": [x.to(device) for x in batch["input_edges_feature"]],
        "input_edges_pad": batch["input_edges_pad"].to(device),
        "labels": batch["labels"].to(device),
        "Tmats": [x.to(device) for x in batch["Tmats"]],
        "adjs": [x.to(device) for x in batch["adjs"]],
        "eadjs": [x.to(device) for x in batch["eadjs"]],
        "mask_edge": batch["mask_edge"].to(device),
    }


def forward_model(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    return model(
        batch["input_nodes_feature"],
        batch["input_edges_feature"],
        batch["input_edges_pad"],
        batch["eadjs"],
        batch["adjs"],
        batch["Tmats"],
        batch["mask_edge"],
    )


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels.astype(int))) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, dict]:
    model.eval()
    losses, scores, labels = [], [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        pred = forward_model(model, batch).view(-1)
        y = batch["labels"].float().view(-1)
        loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
        losses.append(float(loss.detach().cpu()))
        scores.append(pred.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    scores_np = np.concatenate(scores)
    labels_np = np.concatenate(labels)
    return float(np.mean(losses)), scores_np, labels_np, compute_metrics(labels_np, scores_np)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data"))
    parser.add_argument("--source", nargs="+", default=["MOOC", "Wikipedia"])
    parser.add_argument("--target", nargs="+", default=["uci", "btc_otc", "btc_alpha", "email_dnc", "as_topology"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_baselines/GeneralDyG/seed_0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=258)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GeneralDyG code uses CUDA tensors internally; please run on a GPU node.")
    print(f"device={device}", flush=True)

    config = SimpleNamespace(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        drop_out=args.dropout,
        n_layer=args.n_layer,
    )

    source_data = build_source_data(args.data_dir, args.source, args.max_train_samples, args.seed)
    train_idx, val_idx = stratified_split(np.asarray(source_data["labels"]), args.val_ratio, args.seed)
    train_data = take_indices(source_data, train_idx)
    val_data = take_indices(source_data, val_idx)
    print(
        f"source split train_edges={len(train_idx)} train_anom={int(np.asarray(train_data['labels']).sum())} "
        f"val_edges={len(val_idx)} val_anom={int(np.asarray(val_data['labels']).sum())}",
        flush=True,
    )

    train_loader = DataLoader(
        CrossDomainDygDataset(train_data, args.input_dim, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        CrossDomainDygDataset(val_data, args.input_dim, args.seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = make_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state = copy.deepcopy(model.state_dict())
    best_auroc = float("-inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            pred = forward_model(model, batch).view(-1)
            y = batch["labels"].float().view(-1)
            loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_loss, _, _, val_metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
        }
        history.append(record)
        if np.isfinite(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} loss={record['loss']:.6f} "
            f"val_auroc={val_metrics['auroc']:.6f} val_auprc={val_metrics['auprc']:.6f} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early_stop epoch={epoch:03d} stale_epochs={stale}", flush=True)
            break

    model.load_state_dict(best_state)
    results = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "training": {"best_epoch": best_epoch, "best_metric": best_auroc, "history": history},
        "target": {},
    }
    for target_name in args.target:
        target_data = build_target_data(args.data_dir, target_name, args.max_target_samples, args.seed)
        target_loader = DataLoader(
            CrossDomainDygDataset(target_data, args.input_dim, args.seed + 100),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        _, scores, labels, metrics = evaluate(model, target_loader, device)
        results["target"][target_name] = metrics | {
            "rows": int(len(labels)),
            "anomalies": int(labels.sum()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        print(f"target={target_name} {results['target'][target_name]}", flush=True)

    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"saved={args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
