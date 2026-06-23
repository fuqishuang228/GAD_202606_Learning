import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import shortest_path


def tiered_hybrid_rerank(p_tk, p_rf, feats_tk):
    p_tk, p_rf = p_tk.squeeze(), p_rf.squeeze()
    is_zero = (torch.max(torch.abs(feats_tk[:, 1:]), dim=1)[0] < 1e-12)
    zero_scores = p_tk[is_zero]
    threshold = zero_scores.max().item() if len(zero_scores) > 0 else 0.5

    mask_high = (~is_zero) & (p_tk > threshold)
    mask_zero = is_zero
    mask_low = (~is_zero) & (p_tk <= threshold)

    final_rank = torch.zeros_like(p_tk, dtype=torch.float32)
    if mask_high.any():
        final_rank[mask_high] = torch.argsort(torch.argsort(-p_tk[mask_high], stable=True)).float()
    offset2 = mask_high.sum().item()
    if mask_zero.any():
        final_rank[mask_zero] = torch.argsort(torch.argsort(-p_rf[mask_zero], stable=True)).float() + offset2
    offset3 = offset2 + mask_zero.sum().item()
    if mask_low.any():
        final_rank[mask_low] = torch.argsort(torch.argsort(-p_tk[mask_low], stable=True)).float() + offset3

    return (final_rank.max() + 1.0 - final_rank) / (final_rank.max() + 1.0)


def print_comprehensive_report(pos_pred, neg_pred, split_edge, num_nodes, dataset_name, mode, hop_threshold=7):
    pos_pred, neg_pred = pos_pred.cpu().flatten(), neg_pred.cpu().flatten()

    train_edges = split_edge['train']['edge'].t().cpu().numpy()
    adj = sp.csr_matrix((np.ones(train_edges.shape[1]), (train_edges[0], train_edges[1])), shape=(num_nodes, num_nodes))
    dist_matrix = shortest_path(adj, directed=False, unweighted=True)
    dist_pos = dist_matrix[
        split_edge['test']['edge'].cpu().numpy()[:, 0], split_edge['test']['edge'].cpu().numpy()[:, 1]]
    dist_neg = dist_matrix[
        split_edge['test']['edge_neg'].cpu().numpy()[:, 0], split_edge['test']['edge_neg'].cpu().numpy()[:, 1]]
    mask_ge_thresh = (dist_pos >= hop_threshold)
    mask_lt_thresh = (dist_pos < hop_threshold)

    all_preds = torch.cat([pos_pred, neg_pred])
    all_labels = torch.cat([torch.ones(len(pos_pred)), torch.zeros(len(neg_pred))])
    pos_idx_tracker = torch.cat([torch.arange(len(pos_pred)), torch.full((len(neg_pred),), -1, dtype=torch.long)])
    neg_idx_tracker = torch.cat([torch.full((len(pos_pred),), -1, dtype=torch.long), torch.arange(len(neg_pred))])

    sorted_preds, sort_idx = torch.sort(all_preds, descending=True)
    s_labels, s_pos_idx, s_neg_idx = all_labels[sort_idx], pos_idx_tracker[sort_idx], neg_idx_tracker[sort_idx]

    print(f"\n[Analyzing] Computing global ranking and topological distribution for {dataset_name}...")
    print("\n" + "=" * 70)
    print(f" [Tiered Re-ranking: Global Ranking Analysis] - Dataset: {dataset_name} (Mode: {mode.upper()})")
    print("=" * 70)

    neg_s_sorted, _ = torch.sort(neg_pred)
    ranks = len(neg_pred) - torch.searchsorted(neg_s_sorted, pos_pred)

    print(f"\n[Test Result - {mode.upper()}] AUROC: {roc_auc_score(all_labels.numpy(), all_preds.numpy()):.4f}")

    hit_50_val = 0.0
    for k in [20, 50, 100]:
        hits_mask = (ranks < k).numpy()
        total_hits = np.sum(hits_mask)
        ge_hits = np.sum(hits_mask & mask_ge_thresh)
        lt_hits = np.sum(hits_mask & mask_lt_thresh)
        print(f" Hits@{k:<3}: {total_hits / len(pos_pred) * 100:5.2f}% | Long-range Recall: {ge_hits:<4} | Short-range Recall: {lt_hits:<4}")

        if k == 50:
            hit_50_val = total_hits / len(pos_pred)

    return hit_50_val