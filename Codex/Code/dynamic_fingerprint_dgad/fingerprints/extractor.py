from collections import deque
from dataclasses import dataclass

import numpy as np

from .structural import build_undirected_adj, chebyshev_response, normalized_common_neighbor


@dataclass
class FingerprintConfig:
    cheb_order: int = 4
    history_window: int = 5
    cheb_batch_size: int = 2048
    include_edge_surprise: bool = False
    include_node_activity: bool = False
    relative_delta: bool = False
    snapshot_relative_features: bool = False
    relative_feature_mode: str = "none"
    fingerprint_variant: str = "raw10"


class DynamicFingerprintExtractor:
    """Extracts structural and temporal edge fingerprints from rolling graph history.

    Default layout: [nCN_t, CP_t, delta_nCN_t, delta_CP_t].
    Optional temporal features are inserted between static structure and structural deltas.
    """

    def __init__(self, num_nodes: int, config: FingerprintConfig):
        self.num_nodes = num_nodes
        self.config = config
        self.history: deque[np.ndarray] = deque(maxlen=max(1, config.history_window))
        if self.relative_feature_mode not in {"none", "all", "local"}:
            raise ValueError(f"unknown relative_feature_mode: {self.relative_feature_mode}")
        if self.config.fingerprint_variant not in {"raw10", "atlas_local_k2", "atlas_local_k2_fast", "atlas_local_k1"}:
            raise ValueError(f"unknown fingerprint_variant: {self.config.fingerprint_variant}")

    @property
    def relative_feature_mode(self) -> str:
        if self.config.snapshot_relative_features:
            return "all"
        return self.config.relative_feature_mode

    @property
    def feature_dim(self) -> int:
        if self.config.fingerprint_variant == "atlas_local_k2":
            return 32
        if self.config.fingerprint_variant == "atlas_local_k2_fast":
            return 26
        if self.config.fingerprint_variant == "atlas_local_k1":
            return 14
        base_dim = 2 + 2 * self.config.cheb_order
        if self.config.include_node_activity:
            base_dim += 1
        if self.config.include_edge_surprise:
            base_dim += 2
        if self.relative_feature_mode == "all":
            base_dim *= 2
        elif self.relative_feature_mode == "local":
            base_dim += 2
        return base_dim

    def _history_edges(self) -> np.ndarray:
        if not self.history:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(list(self.history), axis=0)

    def _previous_history_edges(self) -> np.ndarray:
        if len(self.history) <= 1:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(list(self.history)[:-1], axis=0)

    @staticmethod
    def _edge_keys(edges: np.ndarray) -> list[tuple[int, int]]:
        return [(int(u), int(v)) for u, v in edges]

    def _edge_surprise(self, history: list[np.ndarray], query_edges: np.ndarray) -> np.ndarray:
        if len(query_edges) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        counts = dict.fromkeys(self._edge_keys(query_edges), 0)
        for snapshot_edges in history:
            seen = set(self._edge_keys(snapshot_edges))
            for edge in counts:
                if edge in seen:
                    counts[edge] += 1
        denom = float(max(1, self.config.history_window))
        values = np.array([1.0 - counts[edge] / denom for edge in self._edge_keys(query_edges)], dtype=np.float32)
        return values.reshape(-1, 1)

    def _node_activity_shift(self, history: list[np.ndarray], query_edges: np.ndarray) -> np.ndarray:
        if len(query_edges) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        current_edges = np.concatenate(history, axis=0) if history else np.zeros((0, 2), dtype=np.int64)
        previous_edges = (
            np.concatenate(history[:-1], axis=0)
            if len(history) > 1
            else np.zeros((0, 2), dtype=np.int64)
        )
        current_deg = np.bincount(current_edges.reshape(-1), minlength=self.num_nodes).astype(np.float32)
        previous_deg = np.bincount(previous_edges.reshape(-1), minlength=self.num_nodes).astype(np.float32)
        node_shift = np.abs(np.log1p(current_deg) - np.log1p(previous_deg))
        values = 0.5 * (node_shift[query_edges[:, 0]] + node_shift[query_edges[:, 1]])
        return values.astype(np.float32).reshape(-1, 1)

    def _static_features(self, history_edges: np.ndarray, query_edges: np.ndarray) -> np.ndarray:
        adj = build_undirected_adj(history_edges, self.num_nodes)
        cn = normalized_common_neighbor(adj, query_edges)
        cp = chebyshev_response(
            adj,
            query_edges,
            order=self.config.cheb_order,
            batch_size=self.config.cheb_batch_size,
        )
        return np.concatenate([cn, cp], axis=1).astype(np.float32)

    @staticmethod
    def _frontiers(adj, node: int) -> tuple[set[int], set[int]]:
        one = set(int(x) for x in adj.indices[adj.indptr[node] : adj.indptr[node + 1]])
        two: set[int] = set()
        for nbr in one:
            two.update(int(x) for x in adj.indices[adj.indptr[nbr] : adj.indptr[nbr + 1]])
        two.discard(node)
        two.difference_update(one)
        return one, two

    @staticmethod
    def _edge_count_between(adj, left: set[int], right: set[int]) -> int:
        if not left or not right:
            return 0
        seen: set[tuple[int, int]] = set()
        for node in left:
            neighbors = adj.indices[adj.indptr[node] : adj.indptr[node + 1]]
            for nbr in neighbors:
                nbr = int(nbr)
                if nbr in right and nbr != node:
                    seen.add((min(node, nbr), max(node, nbr)))
        return len(seen)

    def _atlas_local_features(self, history_edges: np.ndarray, query_edges: np.ndarray, fast: bool = False) -> np.ndarray:
        local_dim = 9 if fast else 12
        if len(query_edges) == 0:
            return np.zeros((0, local_dim), dtype=np.float32)
        adj = build_undirected_adj(history_edges, self.num_nodes)
        deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
        frontier_cache: dict[int, tuple[set[int], set[int]]] = {}

        def get_frontiers(node: int) -> tuple[set[int], set[int]]:
            node = int(node)
            if node not in frontier_cache:
                frontier_cache[node] = self._frontiers(adj, node)
            return frontier_cache[node]

        out = np.zeros((len(query_edges), local_dim), dtype=np.float32)
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for idx, (u, v) in enumerate(query_edges.astype(np.int64)):
            u_frontiers = get_frontiers(int(u))
            v_frontiers = get_frontiers(int(v))
            hn_values = []
            ra_values = []
            lc_values = []
            for left_idx, right_idx in pairs:
                left = u_frontiers[left_idx]
                right = v_frontiers[right_idx]
                overlap = left.intersection(right)
                hn_values.append(float(len(overlap)))
                if overlap:
                    ra_values.append(float(sum(1.0 / max(float(deg[w]), 1.0) for w in overlap)))
                else:
                    ra_values.append(0.0)
                if not fast or (left_idx == 0 and right_idx == 0):
                    lc_values.append(float(self._edge_count_between(adj, left, right)))
            out[idx] = np.log1p(np.array(hn_values + ra_values + lc_values, dtype=np.float32))
        return out

    def _atlas_local_k2_features(self, query_edges: np.ndarray) -> np.ndarray:
        fast = self.config.fingerprint_variant == "atlas_local_k2_fast"
        current_local = self._atlas_local_features(self._history_edges(), query_edges, fast=fast)
        previous_local = self._atlas_local_features(self._previous_history_edges(), query_edges, fast=fast)
        current_static = self._static_features(self._history_edges(), query_edges)
        previous_static = self._static_features(self._previous_history_edges(), query_edges)
        cp = current_static[:, 1:5]
        d_local = current_local - previous_local
        d_cp = current_static[:, 1:5] - previous_static[:, 1:5]
        return np.concatenate([current_local, cp, d_local, d_cp], axis=1).astype(np.float32)

    def _atlas_local_k1_features(self, query_edges: np.ndarray) -> np.ndarray:
        current_local = self._atlas_local_features(self._history_edges(), query_edges, fast=True)[:, [0, 4, 8]]
        previous_local = self._atlas_local_features(self._previous_history_edges(), query_edges, fast=True)[:, [0, 4, 8]]
        current_static = self._static_features(self._history_edges(), query_edges)
        previous_static = self._static_features(self._previous_history_edges(), query_edges)
        cp = current_static[:, 1:5]
        d_local = current_local - previous_local
        d_cp = current_static[:, 1:5] - previous_static[:, 1:5]
        return np.concatenate([current_local, cp, d_local, d_cp], axis=1).astype(np.float32)

    @staticmethod
    def _relative_to_snapshot(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values.astype(np.float32)
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return ((values - mean) / std).astype(np.float32)

    def extract(self, query_edges: np.ndarray) -> np.ndarray:
        if self.config.fingerprint_variant in {"atlas_local_k2", "atlas_local_k2_fast"}:
            return self._atlas_local_k2_features(query_edges)
        if self.config.fingerprint_variant == "atlas_local_k1":
            return self._atlas_local_k1_features(query_edges)
        history = list(self.history)
        previous_history = history[:-1]
        current = self._static_features(self._history_edges(), query_edges)
        previous = self._static_features(self._previous_history_edges(), query_edges)
        optional_current = []
        optional_delta = []
        if self.config.include_node_activity:
            optional_current.append(self._node_activity_shift(history, query_edges))
        if self.config.include_edge_surprise:
            surprise = self._edge_surprise(history, query_edges)
            previous_surprise = self._edge_surprise(previous_history, query_edges)
            optional_current.append(surprise)
            optional_delta.append(surprise - previous_surprise)
        delta = current - previous
        if self.config.relative_delta:
            delta = self._relative_to_snapshot(delta)
            optional_delta = [self._relative_to_snapshot(x) for x in optional_delta]
        features = np.concatenate([current, *optional_current, delta, *optional_delta], axis=1).astype(np.float32)
        if self.relative_feature_mode == "all":
            relative = self._relative_to_snapshot(features)
            features = np.concatenate([features, relative], axis=1).astype(np.float32)
        elif self.relative_feature_mode == "local":
            relative = self._relative_to_snapshot(features)
            delta_start = 1 + self.config.cheb_order + int(self.config.include_node_activity) + int(self.config.include_edge_surprise)
            features = np.concatenate([features, relative[:, [0, delta_start]]], axis=1).astype(np.float32)
        return features

    def update(self, observed_edges: np.ndarray) -> None:
        if len(observed_edges):
            self.history.append(observed_edges.astype(np.int64, copy=True))
