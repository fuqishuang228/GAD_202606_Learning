"""Prepare DP-DGAD pickle datasets from temporal edge lists.

This script combines the anomaly injection idea used by TADDY with the
ego-graph preprocessing used by GeneralDyG. It reads a temporal edge list,
injects random non-edge anomalies into the test timestamps, then writes the
fields expected by DP-DGAD's ``datasets.py``:

    nodefeatures, edgefeatures, labels, Tmats, adjs, eadjs, ra

Example:
    python prepare_dpgad_data.py --dataset btc_alpha --anomaly-per 0.1
"""

from __future__ import annotations

import argparse
import copy
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm


UNLABELED_DATASET_FILES = {
    "uci": "uci",
    "digg": "digg",
    "btc_alpha": "soc-sign-bitcoinalpha.csv",
    "btc_otc": "soc-sign-bitcoinotc.csv",
    "email_dnc": "email-dnc.edges",
    "as_topology": "tech-as-topology.edges",
    "tax51": "tax51.txt",
}

LABELED_DATASETS = {"wikipedia", "wiki", "mooc"}
DATASET_CHOICES = sorted(set(UNLABELED_DATASET_FILES) | LABELED_DATASETS)
DATASET_OUTPUT_NAMES = {
    "wikipedia": "Wikipedia",
    "wiki": "Wikipedia",
    "mooc": "MOOC",
}
EVENT_MODES = ("unique_undirected", "temporal_events")


def load_raw_edges(dataset: str, raw_dir: Path, event_mode: str) -> np.ndarray:
    """Load raw temporal edges and normalize node IDs to a zero-based range."""
    if dataset not in UNLABELED_DATASET_FILES:
        raise ValueError(f"Unsupported unlabeled dataset {dataset!r}. Choose from {sorted(UNLABELED_DATASET_FILES)}")
    if event_mode not in EVENT_MODES:
        raise ValueError(f"Unsupported event mode {event_mode!r}. Choose from {list(EVENT_MODES)}")

    path = raw_dir / UNLABELED_DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    if dataset in {"uci", "digg", "as_topology", "tax51"}:
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

    if event_mode == "unique_undirected":
        edges = np.sort(edges, axis=1)
        _, first_idx = np.unique(edges, return_index=True, axis=0)
        edges = edges[np.sort(first_idx)]

    _, remapped = np.unique(edges, return_inverse=True)
    return remapped.reshape(-1, 2).astype(np.int64)


def load_labeled_events(dataset: str, dynamic_data_dir: Path) -> pd.DataFrame:
    """Load source datasets that already contain labels."""
    if dataset in {"wikipedia", "wiki"}:
        path = dynamic_data_dir / "Wiki" / "wikipedia.csv"
        if not path.exists():
            raise FileNotFoundError(f"Wikipedia source file not found: {path}")
        raw = np.loadtxt(path, dtype=float, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
        raw = raw[np.argsort(raw[:, 2])]
        users = raw[:, 0].astype(np.int64)
        items = raw[:, 1].astype(np.int64)
        item_offset = int(users.max()) + 1
        labels = raw[:, 3].astype(np.int64)
        rows = {
            "u": users,
            "i": items + item_offset,
            "label": labels,
            "id": np.arange(len(raw), dtype=np.int64),
        }
        return pd.DataFrame(rows)

    if dataset == "mooc":
        mooc_dir = dynamic_data_dir / "MOOC" / "act-mooc"
        actions_path = mooc_dir / "mooc_actions.tsv"
        labels_path = mooc_dir / "mooc_action_labels.tsv"
        if not actions_path.exists():
            raise FileNotFoundError(f"MOOC actions file not found: {actions_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"MOOC labels file not found: {labels_path}")

        actions = pd.read_csv(actions_path, sep="\t")
        labels = pd.read_csv(labels_path, sep="\t")
        graph_df = actions.merge(labels, on="ACTIONID", how="inner")
        graph_df = graph_df.sort_values("TIMESTAMP").reset_index(drop=True)
        item_offset = int(graph_df["USERID"].max()) + 1
        return pd.DataFrame(
            {
                "u": graph_df["USERID"].astype(np.int64),
                "i": graph_df["TARGETID"].astype(np.int64) + item_offset,
                "label": graph_df["LABEL"].astype(np.int64),
                "id": np.arange(len(graph_df), dtype=np.int64),
            }
        )

    raise ValueError(f"Unsupported labeled dataset {dataset!r}. Choose from {sorted(LABELED_DATASETS)}")


def sample_random_nonedges(
    num_nodes: int,
    existing_edges: set[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample undirected node pairs that do not appear in the original graph."""
    sampled: set[tuple[int, int]] = set()
    max_possible = num_nodes * (num_nodes - 1) // 2 - len(existing_edges)
    if count > max_possible:
        raise ValueError(f"Requested {count} anomalies, but only {max_possible} non-edges exist.")

    attempts = 0
    max_attempts = max(10000, count * 200)
    while len(sampled) < count:
        if attempts > max_attempts:
            # Dense graphs can make rejection sampling slow; enumerate the rest.
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


def inject_anomalies(
    edges: np.ndarray,
    train_per: float,
    anomaly_per: float,
    anomaly_base: str,
    seed: int,
) -> pd.DataFrame:
    """Create a time-ordered edge table with injected anomalies.

    Labels follow the common anomaly-detection convention used by TADDY:
    0 = normal edge, 1 = injected anomaly.
    """
    rng = np.random.default_rng(seed)
    num_edges = len(edges)
    num_nodes = int(edges.max()) + 1
    train_num = int(np.floor(train_per * num_edges))

    train = edges[:train_num]
    test = edges[train_num:]

    if anomaly_base == "total":
        anomaly_num = int(np.floor(anomaly_per * num_edges))
    elif anomaly_base == "test":
        anomaly_num = int(np.floor(anomaly_per * len(test)))
    else:
        raise ValueError("anomaly_base must be 'total' or 'test'")

    existing_edges = {tuple(sorted(edge)) for edge in edges.tolist()}
    anomalies = sample_random_nonedges(num_nodes, existing_edges, anomaly_num, rng)

    test_len = len(test) + anomaly_num
    anomaly_pos = set(rng.choice(test_len, size=anomaly_num, replace=False).tolist())
    rows = []

    event_id = 0
    for u, v in train:
        rows.append((int(u), int(v), 0, event_id))
        event_id += 1

    normal_idx = 0
    anomaly_idx = 0
    for pos in range(test_len):
        if pos in anomaly_pos:
            u, v = anomalies[anomaly_idx]
            label = 1
            anomaly_idx += 1
        else:
            u, v = test[normal_idx]
            label = 0
            normal_idx += 1
        rows.append((int(u), int(v), label, event_id))
        event_id += 1

    return pd.DataFrame(rows, columns=["u", "i", "label", "id"])


class EgoGraphBuilder:
    def __init__(self, graph_df: pd.DataFrame, k_hop: int, max_nodes: int, max_edges: int | None, seed: int):
        self.graph_df = graph_df
        self.k_hop = k_hop
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.rng = random.Random(seed)
        self.graph = nx.Graph()
        for row in graph_df.itertuples(index=False):
            u, v, edge_id = int(row.u), int(row.i), int(row.id)
            if self.graph.has_edge(u, v):
                self.graph[u][v]["weight"].append(edge_id)
            else:
                self.graph.add_edge(u, v, weight=[edge_id])

    def extract_k_hop_nodes(self, src: int, dst: int) -> set[int]:
        nodes = {src}
        visited = {src}
        frontier = {src}
        for _ in range(self.k_hop):
            new_nodes = set()
            for node in frontier:
                remaining = self.max_nodes - len(nodes) - len(new_nodes)
                if remaining <= 0:
                    break
                neighbors = [neighbor for neighbor in self.graph.neighbors(node) if neighbor not in visited]
                if len(neighbors) > remaining:
                    neighbors = self.rng.sample(neighbors, remaining)
                new_nodes.update(neighbors)
            visited.update(new_nodes)
            nodes.update(new_nodes)
            frontier = new_nodes

        nodes.add(dst)
        nodes = list(nodes)
        if len(nodes) > self.max_nodes:
            required_nodes = [node for node in (src, dst) if node in nodes]
            removable_nodes = [node for node in nodes if node not in {src, dst}]
            keep_count = max(0, self.max_nodes - len(required_nodes))
            nodes = required_nodes + self.rng.sample(removable_nodes, keep_count)
        return set(nodes)

    @staticmethod
    def reorder_nodes(subgraph: nx.Graph) -> dict[int, int]:
        node_weights = {}
        for node in subgraph.nodes:
            edges = subgraph.edges(node, data="weight")
            weights = []
            for _, _, weight in edges:
                if isinstance(weight, list):
                    weights.append(min(weight))
                else:
                    weights.append(weight)
            node_weights[node] = min(weights) if weights else float("inf")
        return {old_id: new_id for new_id, (old_id, _) in enumerate(sorted(node_weights.items(), key=lambda x: x[1]))}

    def replace_subgraph(
        self,
        subgraph: nx.Graph,
        node_mapping: dict[int, int],
        focal_u: int,
        focal_v: int,
        focal_id: int,
    ) -> tuple[nx.Graph, np.ndarray, dict[tuple[int, int], int]]:
        new_subgraph = nx.Graph()
        node_features = np.zeros(len(subgraph.nodes()), dtype=np.int64)
        edge_features: dict[tuple[int, int], int] = {}

        for old_id, new_id in node_mapping.items():
            node_features[new_id] = int(old_id)
            new_subgraph.add_node(new_id)

        for u, v, data in subgraph.edges(data=True):
            new_u, new_v = node_mapping[u], node_mapping[v]
            edge_id = self.rng.choice(data["weight"])
            if {u, v} == {focal_u, focal_v}:
                edge_id = focal_id
            new_subgraph.add_edge(new_u, new_v, weight=1)
            edge = (new_u, new_v) if new_u < new_v else (new_v, new_u)
            edge_features[edge] = int(edge_id)

        return new_subgraph, node_features, edge_features

    def prune_edges(self, subgraph: nx.Graph, focal_u: int, focal_v: int) -> nx.Graph:
        if self.max_edges is None or subgraph.number_of_edges() <= self.max_edges:
            return subgraph

        focal_edge = (focal_u, focal_v) if subgraph.has_edge(focal_u, focal_v) else None
        other_edges = [edge for edge in subgraph.edges() if focal_edge is None or set(edge) != set(focal_edge)]
        keep_count = self.max_edges - (1 if focal_edge is not None else 0)
        keep_edges = set(self.rng.sample(other_edges, max(0, keep_count)))
        if focal_edge is not None:
            keep_edges.add(focal_edge)

        pruned = nx.Graph()
        pruned.add_nodes_from(subgraph.nodes(data=True))
        for u, v in keep_edges:
            pruned.add_edge(u, v, **subgraph[u][v])
        return pruned

    @staticmethod
    def sparse_to_dense_tensor(matrix: sp.spmatrix) -> torch.Tensor:
        return torch.tensor(matrix.toarray(), dtype=torch.float32)

    @staticmethod
    def normalize(matrix: sp.spmatrix) -> sp.spmatrix:
        rowsum = np.array(matrix.sum(1)).astype(float)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        return sp.diags(r_inv).dot(matrix)

    @staticmethod
    def create_transition_matrix(vertex_adj: sp.spmatrix) -> sp.csr_matrix:
        vertex_adj = vertex_adj.copy()
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edges = int(len(edge_index[0]))
        edge_names = list(zip(edge_index[0], edge_index[1]))
        row_index = [node for edge in edge_names for node in edge]
        col_index = np.repeat(np.arange(num_edges), 2)
        data = np.ones(num_edges * 2)
        return sp.csr_matrix((data, (row_index, col_index)), shape=(vertex_adj.shape[0], num_edges))

    @staticmethod
    def create_edge_adj(vertex_adj: sp.spmatrix) -> tuple[sp.csr_matrix, list[tuple[int, int]]]:
        vertex_adj = vertex_adj.copy()
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        edge_names = list(zip(edge_index[0], edge_index[1]))
        num_edges = int(len(edge_names))
        row_index = [node for edge in edge_names for node in edge]
        col_index = np.repeat(np.arange(num_edges), 2)
        data = np.ones(num_edges * 2)
        transition = sp.csr_matrix((data, (row_index, col_index)), shape=(vertex_adj.shape[0], num_edges))
        edge_adj = transition.T.dot(transition)
        edge_adj.data = np.ones_like(edge_adj.data, dtype=np.float32)
        edge_adj.setdiag(1.0)
        return edge_adj.tocsr(), edge_names

    def build(self, limit: int | None = None) -> dict[str, object]:
        nodefeatures = []
        edgefeatures = []
        tmats = []
        adjs = []
        eadjs = []
        ra = []

        rows = self.graph_df if limit is None else self.graph_df.iloc[:limit]
        for row in tqdm(rows.itertuples(index=False), total=len(rows), desc="Building ego-graphs"):
            src, dst, edge_id = int(row.u), int(row.i), int(row.id)
            src_nodes = self.extract_k_hop_nodes(src, dst)
            dst_nodes = self.extract_k_hop_nodes(dst, src)
            subgraph_nodes = src_nodes.union(dst_nodes)
            subgraph = self.graph.subgraph(subgraph_nodes).copy()

            if not subgraph.has_edge(src, dst):
                subgraph.add_edge(src, dst, weight=[edge_id])
            else:
                subgraph[src][dst]["weight"] = list(set(subgraph[src][dst]["weight"] + [edge_id]))
            subgraph = self.prune_edges(subgraph, src, dst)

            node_mapping = self.reorder_nodes(subgraph)
            new_subgraph, node_feat, edge_feat_lookup = self.replace_subgraph(
                subgraph, node_mapping, src, dst, edge_id
            )

            adj = nx.adjacency_matrix(new_subgraph)
            transition = self.create_transition_matrix(adj)
            edge_adj, edge_names = self.create_edge_adj(adj)

            edge_feat = np.zeros(len(edge_names), dtype=np.int64)
            for edge_idx, (u, v) in enumerate(edge_names):
                edge = (u, v) if u < v else (v, u)
                edge_feat[edge_idx] = edge_feat_lookup[edge]

            tmats.append(self.sparse_to_dense_tensor(transition))
            adjs.append(self.sparse_to_dense_tensor(self.normalize(adj + sp.eye(adj.shape[0]))))
            eadjs.append(self.sparse_to_dense_tensor(self.normalize(edge_adj)))
            nodefeatures.append(node_feat)
            edgefeatures.append(edge_feat)
            ra.append(np.zeros(len(edge_feat), dtype=np.float32))

        edgefeatures = self.compact_edge_ids(edgefeatures)
        labels = rows["label"].to_numpy(dtype=np.int64)
        return {
            "nodefeatures": np.array(nodefeatures, dtype=object),
            "edgefeatures": np.array(edgefeatures, dtype=object),
            "labels": labels,
            "Tmats": tmats,
            "adjs": adjs,
            "eadjs": eadjs,
            "ra": np.array(ra, dtype=object),
        }

    @staticmethod
    def compact_edge_ids(edgefeatures: list[np.ndarray]) -> list[np.ndarray]:
        """Remap edge IDs to a dense range expected by DP-DGAD's loader."""
        unique_ids = np.unique(np.concatenate(edgefeatures))
        id_map = {int(old): new for new, old in enumerate(unique_ids.tolist())}
        return [np.array([id_map[int(edge_id)] for edge_id in edges], dtype=np.int64) for edges in edgefeatures]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DP-DGAD data from raw temporal edge lists.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="btc_alpha")
    parser.add_argument("--raw-dir", type=Path, default=Path("../TADDY/data/raw"))
    parser.add_argument("--dynamic-data-dir", type=Path, default=Path("../Data/Dynamic_Graph_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--train-per", type=float, default=0.5)
    parser.add_argument("--anomaly-per", type=float, choices=[0.01, 0.05, 0.1], default=0.1)
    parser.add_argument("--anomaly-base", choices=["total", "test"], default="total")
    parser.add_argument(
        "--event-mode",
        choices=EVENT_MODES,
        default="unique_undirected",
        help="How to treat repeated raw interactions for unlabeled datasets.",
    )
    parser.add_argument("--k-hop", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=26)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Build only the first N events for smoke tests.")
    parser.add_argument("--save-csv", action="store_true", help="Also save the injected edge table as CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.lower()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if dataset in LABELED_DATASETS:
        graph_df = load_labeled_events(dataset, args.dynamic_data_dir)
    else:
        edges = load_raw_edges(dataset, args.raw_dir, args.event_mode)
        graph_df = inject_anomalies(edges, args.train_per, args.anomaly_per, args.anomaly_base, args.seed)

    builder = EgoGraphBuilder(
        graph_df,
        k_hop=args.k_hop,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        seed=args.seed,
    )
    data = builder.build(limit=args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = DATASET_OUTPUT_NAMES.get(dataset, f"{dataset}_{int(args.anomaly_per * 100)}percent")
    if args.limit is not None:
        suffix += f"_limit{args.limit}"
    output_name = args.output_name or suffix
    output_path = args.output_dir / f"{output_name}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    if args.save_csv:
        csv_path = args.output_dir / f"{output_name}.csv"
        graph_df.to_csv(csv_path, index=False)
        print(f"Saved injected edge table to {csv_path}")

    print(f"Saved DP-DGAD dataset to {output_path}")
    print(f"Samples: {len(data['labels'])}, anomalies: {int(data['labels'].sum())}")


if __name__ == "__main__":
    main()
