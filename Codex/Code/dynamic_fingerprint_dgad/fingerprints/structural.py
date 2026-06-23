import numpy as np
from scipy import sparse


def build_undirected_adj(edges: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    if len(edges) == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float32)
    adj = sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj.tocsr()


def normalized_common_neighbor(adj: sparse.csr_matrix, query_edges: np.ndarray) -> np.ndarray:
    if len(query_edges) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    values = np.empty(len(query_edges), dtype=np.float32)
    for idx, (u, v) in enumerate(query_edges):
        cn = adj[u].multiply(adj[v]).sum()
        denom = np.sqrt(max(deg[u] * deg[v], 1.0))
        values[idx] = float(cn) / denom
    return values.reshape(-1, 1)


def normalized_adjacency(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    deg = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    d_inv = sparse.diags(inv_sqrt, dtype=np.float32)
    return (d_inv @ adj @ d_inv).tocsr()


def chebyshev_response(
    adj: sparse.csr_matrix,
    query_edges: np.ndarray,
    order: int = 4,
    batch_size: int = 2048,
) -> np.ndarray:
    """Return [T_1(A)[u,v], ..., T_K(A)[u,v]] for each query edge."""
    if len(query_edges) == 0:
        return np.zeros((0, order), dtype=np.float32)
    a_norm = normalized_adjacency(adj)
    num_nodes = adj.shape[0]
    out = np.zeros((len(query_edges), order), dtype=np.float32)

    for start in range(0, len(query_edges), batch_size):
        end = min(start + batch_size, len(query_edges))
        batch = query_edges[start:end]
        bsz = len(batch)
        cols = np.arange(bsz)
        x0 = sparse.csc_matrix(
            (np.ones(bsz, dtype=np.float32), (batch[:, 1], cols)),
            shape=(num_nodes, bsz),
            dtype=np.float32,
        )
        if order <= 0:
            continue
        x1 = a_norm @ x0
        out[start:end, 0] = np.asarray(x1[batch[:, 0], cols]).reshape(-1)
        xm2, xm1 = x0, x1
        for k in range(1, order):
            xk = 2.0 * (a_norm @ xm1) - xm2
            out[start:end, k] = np.asarray(xk[batch[:, 0], cols]).reshape(-1)
            xm2, xm1 = xm1, xk
    return out

