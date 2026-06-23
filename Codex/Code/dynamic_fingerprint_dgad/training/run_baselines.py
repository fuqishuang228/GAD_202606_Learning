from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dynamic_fingerprint_dgad.data.csv_dynamic_dataset import CSVDynamicDataset
from dynamic_fingerprint_dgad.fingerprints.extractor import DynamicFingerprintExtractor, FingerprintConfig
from dynamic_fingerprint_dgad.training.metrics import compute_metrics


FEATURE_NAMES = ["nCN", "CP1", "CP2", "CP3", "CP4", "d_nCN", "d_CP1", "d_CP2", "d_CP3", "d_CP4"]


def extract_dataset(dataset: CSVDynamicDataset, fp_cfg: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
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
    return np.concatenate(xs), np.concatenate(ys)


def heuristic_scores(x: np.ndarray, cheb_order: int) -> dict[str, np.ndarray]:
    names = FEATURE_NAMES[: 2 + 2 * cheb_order]
    idx = {name: i for i, name in enumerate(names)}
    scores = {}
    for key in ["nCN", "CP2", "CP3", "CP4", "d_CP3", "d_CP4"]:
        if key in idx:
            scores[f"-{key}"] = -x[:, idx[key]]
            scores[f"+{key}"] = x[:, idx[key]]
    if "CP3" in idx and "CP4" in idx:
        scores["-(CP3+CP4)"] = -(x[:, idx["CP3"]] + x[:, idx["CP4"]])
    if "CP2" in idx and "CP3" in idx and "CP4" in idx:
        scores["-(CP2+CP3+CP4)"] = -(x[:, idx["CP2"]] + x[:, idx["CP3"]] + x[:, idx["CP4"]])
    return scores


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    if len(np.unique(train_y.astype(int))) < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=0,
        ),
    )
    model.fit(train_x, train_y.astype(int))
    return model


def predict_logistic(model, x: np.ndarray, fallback_rate: float) -> np.ndarray:
    if model is None:
        return np.full(len(x), fallback_rate, dtype=np.float32)
    return model.predict_proba(x)[:, 1]


def eval_scores(labels: np.ndarray, scores_by_name: dict[str, np.ndarray]) -> dict:
    return {
        name: compute_metrics(labels, scores)
        | {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }
        for name, scores in scores_by_name.items()
    }


def write_method_results(out_dir: Path, split_name: str, result: dict) -> None:
    by_method: dict[str, dict] = {}
    if split_name == "cross_domain":
        for dataset_name, dataset_result in result[split_name].items():
            for method_name, metrics in dataset_result.items():
                by_method.setdefault(method_name, {})[dataset_name] = metrics
    else:
        for dataset_name, dataset_result in result[split_name].items():
            for method_name, metrics in dataset_result["metrics"].items():
                by_method.setdefault(method_name, {})[dataset_name] = {
                    "split": dataset_result["split"],
                    "metrics": metrics,
                }

    for method_name, method_result in by_method.items():
        safe_name = method_name.replace("+", "plus").replace("-", "minus").replace("(", "").replace(")", "")
        method_dir = out_dir / split_name / safe_name
        method_dir.mkdir(parents=True, exist_ok=True)
        with open(method_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "method": method_name,
                    "split": split_name,
                    "target": method_result,
                },
                f,
                indent=2,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data"))
    parser.add_argument("--source", nargs="+", default=["MOOC", "Wikipedia"])
    parser.add_argument("--target", nargs="+", default=["uci", "btc_otc"])
    parser.add_argument("--single", nargs="*", default=["uci", "btc_otc"])
    parser.add_argument("--out-dir", type=Path, default=Path("dynamic_fingerprint_dgad/results/baselines"))
    parser.add_argument("--num-snapshots", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--history-window", type=int, default=5)
    parser.add_argument("--cheb-order", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fp_cfg = FingerprintConfig(cheb_order=args.cheb_order, history_window=args.history_window)

    result = {
        "config": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir)},
        "cross_domain": {},
        "single_dataset": {},
    }

    print("=== cross-domain feature extraction ===", flush=True)
    source_xs, source_ys = [], []
    for name in args.source:
        ds = CSVDynamicDataset(args.data_dir / f"{name}.csv", args.num_snapshots, args.max_rows)
        x, y = extract_dataset(ds, fp_cfg)
        source_xs.append(x)
        source_ys.append(y)
    train_x = np.concatenate(source_xs)
    train_y = np.concatenate(source_ys)
    logreg = fit_logistic(train_x, train_y)

    for name in args.target:
        ds = CSVDynamicDataset(args.data_dir / f"{name}.csv", args.num_snapshots, args.max_rows)
        x, y = extract_dataset(ds, fp_cfg)
        scores = heuristic_scores(x, args.cheb_order)
        scores["logreg_cross"] = predict_logistic(logreg, x, float(train_y.mean()))
        result["cross_domain"][name] = eval_scores(y, scores)
        print(f"cross target={name}", result["cross_domain"][name], flush=True)

    print("=== single-dataset temporal split ===", flush=True)
    for name in args.single:
        ds = CSVDynamicDataset(args.data_dir / f"{name}.csv", args.num_snapshots, args.max_rows)
        x, y = extract_dataset(ds, fp_cfg)
        split = int(args.train_ratio * args.num_snapshots)
        edge_counts = [len(s.edges) for s in ds.snapshots]
        split_edge = int(np.sum(edge_counts[:split]))
        train_x, train_y = x[:split_edge], y[:split_edge]
        test_x, test_y = x[split_edge:], y[split_edge:]
        model = fit_logistic(train_x, train_y)
        scores = heuristic_scores(test_x, args.cheb_order)
        scores["logreg_single"] = predict_logistic(model, test_x, float(train_y.mean()))
        result["single_dataset"][name] = {
            "split": {
                "train_edges": int(len(train_y)),
                "test_edges": int(len(test_y)),
                "train_anomalies": int(train_y.sum()),
                "test_anomalies": int(test_y.sum()),
            },
            "metrics": eval_scores(test_y, scores),
        }
        print(f"single dataset={name}", result["single_dataset"][name], flush=True)

    out_path = args.out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    write_method_results(args.out_dir, "cross_domain", result)
    write_method_results(args.out_dir, "single_dataset", result)
    print(f"saved={out_path}", flush=True)


if __name__ == "__main__":
    main()
