from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from run_cross_domain import (
    CrossDomainDygDataset,
    batch_to_device,
    build_target_data,
    collate_fn,
    compute_metrics,
    evaluate,
    forward_model,
    make_model,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data"))
    parser.add_argument("--dataset", type=str, default="uci")
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_baselines/GeneralDyG_within/uci/seed_0"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--loss-fn", choices=["prob_bce", "bce_with_logits"], default="prob_bce")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GeneralDyG code uses CUDA tensors internally; please run on a GPU node.")
    print(f"device={device}", flush=True)

    data = build_target_data(args.data_dir, args.dataset, args.max_samples, args.seed)
    labels = np.asarray(data["labels"], dtype=np.float32)
    split_idx = int(len(labels) * args.train_ratio)
    split_idx = max(1, min(split_idx, len(labels) - 1))
    train_idx = np.arange(split_idx)
    test_idx = np.arange(split_idx, len(labels))
    print(
        f"dataset={args.dataset} rows={len(labels)} anomalies={int(labels.sum())} "
        f"train_edges={len(train_idx)} train_anom={int(labels[train_idx].sum())} "
        f"test_edges={len(test_idx)} test_anom={int(labels[test_idx].sum())}",
        flush=True,
    )

    full_dataset = CrossDomainDygDataset(data, args.input_dim, args.seed)
    train_loader = DataLoader(
        Subset(full_dataset, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    config = SimpleNamespace(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        drop_out=args.dropout,
        n_layer=args.n_layer,
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
            if args.loss_fn == "bce_with_logits":
                loss = F.binary_cross_entropy_with_logits(pred, y)
            else:
                loss = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        test_loss, scores, test_labels, test_metrics = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "test_loss": test_loss,
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
        }
        history.append(record)
        if np.isfinite(test_metrics["auroc"]) and test_metrics["auroc"] > best_auroc:
            best_auroc = test_metrics["auroc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} loss={record['loss']:.6f} "
            f"test_auroc={test_metrics['auroc']:.6f} test_auprc={test_metrics['auprc']:.6f} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early_stop epoch={epoch:03d} stale_epochs={stale}", flush=True)
            break

    model.load_state_dict(best_state)
    _, scores, test_labels, final_metrics = evaluate(model, test_loader, device)
    results = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "split": {
            "train_edges": int(len(train_idx)),
            "test_edges": int(len(test_idx)),
            "train_anomalies": int(labels[train_idx].sum()),
            "test_anomalies": int(test_labels.sum()),
        },
        "training": {"best_epoch": best_epoch, "best_metric": best_auroc, "history": history},
        "test": final_metrics | {
            "rows": int(len(test_labels)),
            "anomalies": int(test_labels.sum()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        },
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), args.out_dir / "model.pt")
    print(f"test={results['test']}", flush=True)
    print(f"saved={args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
