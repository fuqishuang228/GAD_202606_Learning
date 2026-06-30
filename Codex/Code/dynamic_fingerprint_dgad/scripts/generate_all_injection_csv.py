from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_FILES = {
    "uci": "TADDY/data/raw/uci",
    "btc_otc": "TADDY/data/raw/soc-sign-bitcoinotc.csv",
    "btc_alpha": "TADDY/data/raw/soc-sign-bitcoinalpha.csv",
    "email_dnc": "Data/Dynamic_Graph_data/Email-DNC/email-dnc.edges",
    "as_topology": "Data/Dynamic_Graph_data/AS-Topology/tech-as-topology.edges",
}

DATASET_EVENT_MODES = {
    "uci": "temporal_events",
    "btc_otc": "temporal_events",
    "btc_alpha": "temporal_events",
    "email_dnc": "temporal_events",
    "as_topology": "temporal_events",
}


def load_raw_edges(dataset: str, root: Path) -> np.ndarray:
    path = root / DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(path)

    if dataset in {"uci", "as_topology"}:
        raw = np.loadtxt(path, dtype=float, comments="%", delimiter=" ")
        time_col = 3 if raw.shape[1] > 3 else 2
        raw = raw[np.argsort(raw[:, time_col])]
        edges = raw[:, :2].astype(np.int64)
    elif dataset == "email_dnc":
        raw = np.genfromtxt(path, dtype=float, delimiter=",", encoding="utf-8-sig")
        raw = raw[np.argsort(raw[:, 2])]
        edges = raw[:, :2].astype(np.int64)
    else:
        raw = np.loadtxt(path, dtype=float, delimiter=",")
        raw = raw[np.argsort(raw[:, 3])]
        edges = raw[:, :2].astype(np.int64)

    edges = edges[edges[:, 0] != edges[:, 1]]
    if DATASET_EVENT_MODES[dataset] == "unique_undirected":
        edges = np.sort(edges, axis=1)
        _, first_idx = np.unique(edges, return_index=True, axis=0)
        edges = edges[np.sort(first_idx)]

    _, remapped = np.unique(edges, return_inverse=True)
    return remapped.reshape(-1, 2).astype(np.int64)


def sample_random_nonedges(
    num_nodes: int,
    existing_edges: set[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: set[tuple[int, int]] = set()
    max_possible = num_nodes * (num_nodes - 1) // 2 - len(existing_edges)
    if count > max_possible:
        raise ValueError(f"Requested {count} anomalies, but only {max_possible} non-edges exist.")

    attempts = 0
    max_attempts = max(10000, count * 200)
    while len(sampled) < count:
        if attempts > max_attempts:
            candidates = [
                (u, v)
                for u in range(num_nodes)
                for v in range(u + 1, num_nodes)
                if (u, v) not in existing_edges and (u, v) not in sampled
            ]
            need = count - len(sampled)
            chosen = rng.choice(len(candidates), size=need, replace=False)
            sampled.update(candidates[i] for i in chosen)
            break

        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        attempts += 1
        if u == v:
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in existing_edges or edge in sampled:
            continue
        sampled.add(edge)

    return np.array(sorted(sampled), dtype=np.int64)


def inject_all(edges: np.ndarray, anomaly_per: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num_edges = len(edges)
    num_nodes = int(edges.max()) + 1
    existing_edges = {tuple(sorted(edge)) for edge in edges.tolist()}
    anomaly_num = int(np.floor(anomaly_per * num_edges))
    anomalies = sample_random_nonedges(num_nodes, existing_edges, anomaly_num, rng)

    mixed_len = num_edges + anomaly_num
    anomaly_pos = set(rng.choice(mixed_len, size=anomaly_num, replace=False).tolist())
    rows = []
    normal_idx = 0
    anomaly_idx = 0
    for event_id in range(mixed_len):
        if event_id in anomaly_pos:
            u, v = anomalies[anomaly_idx]
            label = 1
            anomaly_idx += 1
        else:
            u, v = edges[normal_idx]
            label = 0
            normal_idx += 1
        rows.append((int(u), int(v), label, event_id))
    return pd.DataFrame(rows, columns=["u", "i", "label", "id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV-only all-time anomaly injection datasets.")
    parser.add_argument("--root", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD/data"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_FILES))
    parser.add_argument("--anomaly-per", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        edges = load_raw_edges(dataset, args.root)
        df = inject_all(edges, args.anomaly_per, args.seed)
        suffix = "temporal_all_injection" if dataset == "uci" else "all_injection"
        out_path = args.output_dir / f"{dataset}_{suffix}.csv"
        df.to_csv(out_path, index=False)
        labels = df["label"].to_numpy()
        pos = np.flatnonzero(labels == 1)
        print(
            f"{dataset}: saved={out_path} rows={len(df)} anomalies={int(labels.sum())} "
            f"rate={labels.mean():.6f} first_anomaly_pos={int(pos[0]) if len(pos) else 'NA'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
