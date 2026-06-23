import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.utils import to_undirected, negative_sampling
import numpy as np

from utils import load_data, set_random_seeds
from models import DualStreamInContextPredictor
from features import extract_path_counts_minibatch, extract_structural_features
from evaluator import print_comprehensive_report


def deterministic_structural_sampling(features, num_samples):
    N = features.size(0)
    device = features.device
    if N <= num_samples:
        return torch.arange(N, device=device)

    cn_counts = features[:, 4]
    local_density = features[:, 22]

    combined_score = cn_counts * 10000.0 + local_density * 10.0 + (torch.arange(N, device=device).float() / N)
    sorted_idx = torch.argsort(combined_score, stable=True)

    step = N / num_samples
    sampled_indices = [sorted_idx[int(i * step + step / 2)].item() for i in range(num_samples)]

    return torch.tensor(sampled_indices, device=device, dtype=torch.long)


def get_separated_features(adj_idx, N, edges, tk_hops, struct_hops, device):
    path_feat_raw = extract_path_counts_minibatch(adj_idx, N, edges, tk_hops, batch_size=2048).to(device)
    struct_feat_raw = extract_structural_features(adj_idx, N, edges, struct_hops).to(device)

    def sym_log1p(v): return torch.sign(v) * torch.log1p(torch.abs(v))

    path_feat_log = path_feat_raw
    struct_feat_log = sym_log1p(struct_feat_raw)

    return path_feat_log, struct_feat_log


def get_scores_in_batches(model, sp_tk, sn_tk, sp_st, sn_st, q_tk, q_st, ctx_p_tk, ctx_n_tk, ctx_p_st, ctx_n_st,
                          bsz=2048):
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, q_tk.size(0), bsz):
            end = min(i + bsz, q_tk.size(0))
            logits = model(sp_tk, sn_tk, q_tk[i:end], sp_st, sn_st, q_st[i:end],
                           ctx_p_tk, ctx_n_tk, ctx_p_st, ctx_n_st)
            scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores, dim=0)


def run_experiment(args, seed):
    set_random_seeds(seed)
    args.dataset_dir = Path(args.dataset_dir)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    tk_dim = args.tk_hops
    struct_dim = 3 * (args.struct_hops ** 2) + 2 * args.struct_hops

    print(f"🚀 Mode: DUAL-STREAM | 🌟 Deterministic Structural Sampling | Macro: {tk_dim}-hop | Structure: {struct_dim}")

    model = DualStreamInContextPredictor(tk_dim=tk_dim, struct_dim=struct_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # ==========================================
    # 🌟 Extract and save features for training sets only, and build a 10x negative sample pool
    # ==========================================
    cache_dir = Path(f"./feature_cache/tk{args.tk_hops}_st{args.struct_hops}")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"--- Preloading and processing training datasets: {args.train_c} ---")
    train_datasets_info = []
    for train_name in args.train_c:
        cache_file = cache_dir / f"train_{train_name}_pool.pt"

        # If cache exists, load it immediately
        if cache_file.exists():
            print(f"  [Cache Hit] ⚡ Loading training feature bank: {train_name}")
            train_datasets_info.append(torch.load(cache_file))
            continue

        print(f"  [No Cache] 🐢 Computing training feature bank: {train_name}...")
        data_tr, split_edge_tr = load_data(train_name, args.dataset_dir, use_valedges_as_input=False, no_split=False,
                                           run=0)
        N_tr = data_tr.num_nodes
        train_edges = split_edge_tr['train']['edge']

        # Lock the split seed to ensure the cached base graph is fixed
        generator = torch.Generator().manual_seed(10)
        perm = torch.randperm(train_edges.size(0), generator=generator)
        split_idx = int(train_edges.size(0) * 0.9)
        mp_edges = train_edges[perm[:split_idx]]
        sup_edges = train_edges[perm[split_idx:]]

        adj_mp = to_undirected(mp_edges.t())

        # Build a negative sample pool 10 times the size of positive samples
        num_neg_pool = sup_edges.size(0) * 10
        edge_neg_pool = negative_sampling(to_undirected(train_edges.t()), N_tr, num_neg_pool).t()

        tk_p, st_p = get_separated_features(adj_mp, N_tr, sup_edges.t(), args.tk_hops, args.struct_hops, device)
        tk_n_pool, st_n_pool = get_separated_features(adj_mp, N_tr, edge_neg_pool.t(), args.tk_hops, args.struct_hops,
                                                      device)

        dataset_info = {
            'name': train_name, 'N': N_tr,
            'tk_p': tk_p, 'st_p': st_p,
            'tk_n_pool': tk_n_pool, 'st_n_pool': st_n_pool
        }
        torch.save(dataset_info, cache_file)
        train_datasets_info.append(dataset_info)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_train_samples = sum(t_info['tk_p'].size(0) for t_info in train_datasets_info)

    print(f"\n{'=' * 60}")
    print(f"📊 [Pre-run Stats] Trainable parameters: {total_params:,} | Total positive training samples: {total_train_samples:,}")
    print(f"{'=' * 60}\n")

    print(f"\n--- Joint Cross-Domain Training Started ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for t_info in train_datasets_info:
            tk_p, st_p = t_info['tk_p'], t_info['st_p']
            tk_n_pool, st_n_pool = t_info['tk_n_pool'], t_info['st_n_pool']

            actual_ctx = min(args.k_shot, tk_p.size(0) // 2)
            if actual_ctx == 0: actual_ctx = 1
            bsz = min(1024, tk_p.size(0) - actual_ctx)

            # Calculate the total number of samples needed (Context + Query)
            total_needed = actual_ctx + bsz

            # 🌟 During training: Shuffle and sample positive samples, randomly draw negative samples from the 10x pool!
            idx_p_rand = torch.randperm(tk_p.size(0))
            idx_n_rand = torch.randperm(tk_n_pool.size(0))

            tk_p_curr = tk_p[idx_p_rand[:total_needed]]
            st_p_curr = st_p[idx_p_rand[:total_needed]]
            tk_n_curr = tk_n_pool[idx_n_rand[:total_needed]]
            st_n_curr = st_n_pool[idx_n_rand[:total_needed]]

            ctx_p_tk, ctx_p_st = tk_p_curr[:actual_ctx], st_p_curr[:actual_ctx]
            ctx_n_tk, ctx_n_st = tk_n_curr[:actual_ctx], st_n_curr[:actual_ctx]

            actual_s = min(32, ctx_p_tk.size(0))

            sp_idx = deterministic_structural_sampling(ctx_p_st, actual_s)
            sp_tk, sp_st = ctx_p_tk[sp_idx], ctx_p_st[sp_idx]

            sn_idx = deterministic_structural_sampling(ctx_n_st, actual_s)
            sn_tk, sn_st = ctx_n_tk[sn_idx], ctx_n_st[sn_idx]

            q_p_tk, q_p_st = tk_p_curr[actual_ctx:], st_p_curr[actual_ctx:]
            q_n_tk, q_n_st = tk_n_curr[actual_ctx:], st_n_curr[actual_ctx:]

            query_tk = torch.cat([q_p_tk, q_n_tk], dim=0)
            query_st = torch.cat([q_p_st, q_n_st], dim=0)
            labels = torch.cat([torch.ones(len(q_p_tk)), torch.zeros(len(q_n_tk))]).to(device).unsqueeze(1)

            logits = model(sp_tk, sn_tk, query_tk, sp_st, sn_st, query_st, ctx_p_tk, ctx_n_tk, ctx_p_st, ctx_n_st)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss

        avg_loss = total_loss / len(train_datasets_info)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")

    print(f"\n--- Joint Training Completed, Starting Testing ---")
    model.eval()

    test_results = {}

    # ==========================================
    # 🌟 Test set evaluation remains exactly the same
    # ==========================================
    with torch.no_grad():
        for test_name in args.test_c:
            print(f"\n>>>> Testing dataset: {test_name} <<<<")
            data_te, split_edge_te = load_data(test_name, args.dataset_dir, use_valedges_as_input=False, no_split=False,
                                               run=0)
            N_te = data_te.num_nodes
            test_train_edges = split_edge_te['train']['edge']

            total_edges = test_train_edges.size(0)
            dynamic_ctx_size = int(total_edges * 0.01)
            actual_ctx_te = min(dynamic_ctx_size, 10000)

            row, col = test_train_edges[:, 0], test_train_edges[:, 1]
            deg = torch.bincount(torch.cat([row, col]), minlength=N_te)
            edge_richness = deg[row] + deg[col]

            sorted_idx_te = torch.argsort(edge_richness)
            step_ctx_te = torch.linspace(0, len(sorted_idx_te) - 1, steps=actual_ctx_te).long()
            fixed_ctx_idx_te = sorted_idx_te[step_ctx_te]

            ctx_p_edges_te = test_train_edges[fixed_ctx_idx_te].t()

            mask_te = torch.ones(len(test_train_edges), dtype=torch.bool)
            mask_te[fixed_ctx_idx_te] = False
            te_mp_edges = test_train_edges[mask_te].t()
            adj_ctx_mp = to_undirected(te_mp_edges)

            if 'edge_neg' in split_edge_te['train']:
                neg_edges = split_edge_te['train']['edge_neg']
                row_n, col_n = neg_edges[:, 0], neg_edges[:, 1]
                edge_richness_n = deg[row_n] + deg[col_n]
                sorted_idx_n_te = torch.argsort(edge_richness_n)
                step_ctx_n_te = torch.linspace(0, len(sorted_idx_n_te) - 1, steps=actual_ctx_te).long()
                ctx_n_edges_te = neg_edges[sorted_idx_n_te[step_ctx_n_te]].t()
            else:
                ctx_n_edges_te = negative_sampling(adj_ctx_mp, N_te, actual_ctx_te)

            ctx_p_tk, ctx_p_st = get_separated_features(adj_ctx_mp, N_te, ctx_p_edges_te, args.tk_hops,
                                                        args.struct_hops, device)
            ctx_n_tk, ctx_n_st = get_separated_features(adj_ctx_mp, N_te, ctx_n_edges_te, args.tk_hops,
                                                        args.struct_hops, device)

            actual_s_te = min(32, ctx_p_tk.size(0))

            sp_idx_te = deterministic_structural_sampling(ctx_p_st, actual_s_te)
            sp_tk, sp_st = ctx_p_tk[sp_idx_te], ctx_p_st[sp_idx_te]

            sn_idx_te = deterministic_structural_sampling(ctx_n_st, actual_s_te)
            sn_tk, sn_st = ctx_n_tk[sn_idx_te], ctx_n_st[sn_idx_te]

            query_p_edges = split_edge_te['test']['edge'].t()
            query_n_edges = split_edge_te['test']['edge_neg'].t()

            full_tr_adj_te = to_undirected(test_train_edges.t())

            qp_tk, qp_st = get_separated_features(full_tr_adj_te, N_te, query_p_edges, args.tk_hops, args.struct_hops,
                                                  device)
            qn_tk, qn_st = get_separated_features(full_tr_adj_te, N_te, query_n_edges, args.tk_hops, args.struct_hops,
                                                  device)

            pos_s = get_scores_in_batches(model, sp_tk, sn_tk, sp_st, sn_st, qp_tk, qp_st, ctx_p_tk, ctx_n_tk, ctx_p_st,
                                          ctx_n_st)
            neg_s = get_scores_in_batches(model, sp_tk, sn_tk, sp_st, sn_st, qn_tk, qn_st, ctx_p_tk, ctx_n_tk, ctx_p_st,
                                          ctx_n_st)

            cross_domain_name = f"Multi-Train -> {test_name}"

            # ==== Directly receive returned results, no need to recalculate rankings ====
            hit_50 = print_comprehensive_report(pos_s.flatten(), neg_s.flatten(), split_edge_te, N_te,
                                                cross_domain_name,
                                                "Dual-PathCount-PA-LOG1P")

            test_results[test_name] = hit_50

    print(f"\n{'=' * 60}")
    print(f"🏁 [Post-run Stats] Trainable parameters: {total_params:,} | Total positive training samples: {total_train_samples:,}")
    print(f"{'=' * 60}\n")

    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--train_c', nargs='+',
                        default=['polblogs', 'Power', 'Router', 'Twitch', 'Github', 'Physics', 'Pubmed', 'Citeseer',
                                 'Ecoli', 'Yeast', ])
    parser.add_argument('--test_c', nargs='+', default=['Celegans', 'USAir', 'PB', 'NS', 'CS', 'Cora', 'facebook'])
    parser.add_argument('--tk_hops', type=int, default=12)
    parser.add_argument('--struct_hops', type=int, default=3)
    parser.add_argument('--k_shot', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    all_results = {test_name: [] for test_name in args.test_c}

    for seed in range(5):
        print(f"\n{'=' * 50}")
        print(f"🌟 Starting execution for Seed: {seed}")
        print(f"{'=' * 50}\n")

        seed_results = run_experiment(args, seed)

        for test_name, hit_val in seed_results.items():
            all_results[test_name].append(hit_val)

    print("\n\n" + "=" * 70)
    print(f"🏆 Final Summary of 5 Seeds (Hit@50)")
    print("=" * 70)
    for test_name in args.test_c:
        hit_list = all_results[test_name]
        mean_hit = np.mean(hit_list)
        std_hit = np.std(hit_list)
        details_str = ", ".join([f"{h:.4f}" for h in hit_list])
        print(f"📊 Dataset: {test_name:<10} | Avg Hit@50: {(mean_hit*100):.2f}% ± {(std_hit*100):.2f}%  |  Details (5 runs): [{details_str}]")
    print("=" * 70 + "\n")