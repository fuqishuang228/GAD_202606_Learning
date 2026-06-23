import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import degree


# ==========================================
# 1. Macroscopic Features: Chebyshev Recursive Extraction (Chebyshev Polynomial Features)
# Use T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x) to recursively calculate the macroscopic receptive field
# ==========================================
def extract_path_counts_minibatch(edge_index_graph, num_nodes, target_edges, K_hops, batch_size=2048):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    row, col = edge_index_graph[0].to(device), edge_index_graph[1].to(device)

    # [Modification 1] Chebyshev polynomials must use the symmetrically normalized adjacency matrix D^{-1/2} A D^{-1/2}
    # To ensure eigenvalues are within the [-1, 1] range, preventing exponential numerical explosion from consecutive multiplications
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_val = deg_inv_sqrt[row] * torch.ones(edge_index_graph.size(1), device=device) * deg_inv_sqrt[col]

    A_norm = torch.sparse_coo_tensor(torch.stack([row, col]), norm_val, (num_nodes, num_nodes)).coalesce()

    target_edges = target_edges.to(device)
    M = target_edges.size(1)
    path_features = torch.zeros((M, K_hops), dtype=torch.float32, device=device)

    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        batch_u = target_edges[0, i:end]
        batch_v = target_edges[1, i:end]
        bsz = end - i

        # T_0(x) = X_0 (Initial state)
        X_0 = torch.zeros((num_nodes, bsz), dtype=torch.float32, device=device)
        X_0[batch_v, torch.arange(bsz)] = 1.0

        if K_hops > 0:
            # T_1(x) = A_norm * X_0 (1st-order Chebyshev response)
            X_1 = torch.sparse.mm(A_norm, X_0)
            path_features[i:end, 0] = X_1[batch_u, torch.arange(bsz)]

        # Record the states of the previous two steps in preparation for Chebyshev recursion
        X_k_minus_2 = X_0
        X_k_minus_1 = X_1

        # [Modification 2] Standard Chebyshev recursion: T_k(A) = 2 A T_{k-1}(A) - T_{k-2}(A)
        for k in range(1, K_hops):
            X_k = 2 * torch.sparse.mm(A_norm, X_k_minus_1) - X_k_minus_2
            path_features[i:end, k] = X_k[batch_u, torch.arange(bsz)]

            # Slide the state cursor
            X_k_minus_2 = X_k_minus_1
            X_k_minus_1 = X_k

    return path_features.cpu()


# ==========================================
# 2. Microscopic Features: Ultimate Holographic Structure Extraction (DRNL + RA Degree Penalty + PA Activity Signature)
# ==========================================
def extract_structural_features(edge_index_graph, num_nodes, target_edges, K_hops, batch_size=2048):
    device = target_edges.device
    row, col = edge_index_graph[0].to(device), edge_index_graph[1].to(device)
    val = torch.ones(edge_index_graph.size(1), device=device)
    A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (num_nodes, num_nodes)).coalesce()

    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = 1.0 / (deg + 1e-10)
    deg_inv = deg_inv.unsqueeze(1).to(device)

    M = target_edges.size(1)

    # Feature dimensions: 3 groups of grid features (3*K^2) + 2 groups of volume features (2*K)
    struct_features = torch.zeros((M, 3 * K_hops * K_hops + 2 * K_hops), dtype=torch.float32, device=device)

    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        batch_u = target_edges[0, i:end]
        batch_v = target_edges[1, i:end]
        bsz = end - i

        def get_frontiers(roots):
            fronts = []
            visited = torch.zeros((num_nodes, bsz), dtype=torch.bool, device=device)
            visited[roots, torch.arange(bsz)] = True

            curr_f = torch.zeros((num_nodes, bsz), dtype=torch.float32, device=device)
            curr_f[roots, torch.arange(bsz)] = 1.0
            fronts.append(curr_f > 0)

            for k in range(1, K_hops):
                nxt_f = torch.sparse.mm(A, curr_f)
                is_new = (nxt_f > 0) & (~visited)
                fronts.append(is_new)
                visited |= is_new
                curr_f = is_new.float()
            return fronts

        f_u = get_frontiers(batch_u)
        f_v = get_frontiers(batch_v)

        idx = 0
        for a in range(K_hops):
            for b in range(K_hops):
                intersect = (f_u[a] & f_v[b]).float()

                # Grid feature 1: Absolute number of nodes
                raw_count = intersect.sum(dim=0)
                # Grid feature 2: RA penalized number of nodes (Suppressing super-nodes)
                ra_count = (intersect * deg_inv).sum(dim=0)
                # Grid feature 3: Internal routing (Local density)
                A_fv_b = torch.sparse.mm(A, f_v[b].float())
                edges_between = (f_u[a].float() * A_fv_b).sum(dim=0)

                struct_features[i:end, idx] = raw_count
                struct_features[i:end, K_hops * K_hops + idx] = ra_count
                struct_features[i:end, 2 * K_hops * K_hops + idx] = edges_between
                idx += 1

        offset = 3 * K_hops * K_hops
        for a in range(K_hops):
            sz_u = f_u[a].float().sum(dim=0)
            sz_v = f_v[a].float().sum(dim=0)

            # Activity feature: PA product (A magic trick to solve deadlocks)
            struct_features[i:end, offset + a] = sz_u * sz_v
            # Activity feature: Expansion difference
            struct_features[i:end, offset + K_hops + a] = np.abs(sz_u - sz_v)

    return struct_features.cpu()