import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DomainPromptGenerator(nn.Module):
    """
    Use cross-attention mechanism based on the Perceiver architecture
    Condense thousands of Context samples into K pure Domain Prompt Tokens
    """

    def __init__(self, d_model, num_prompts=8):
        super(DomainPromptGenerator, self).__init__()
        self.num_prompts = num_prompts
        self.d_model = d_model

        # 🌟 Learnable latent variable query vectors (Latent Queries)
        self.latent_queries = nn.Parameter(torch.randn(num_prompts, d_model))

        # Cross-attention components
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, context_features):
        # context_features: (batch_size=1, num_context, d_model)
        # latent_queries: (batch_size=1, num_prompts, d_model)
        B = context_features.size(0)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        # Latent Tokens cross-attend to massive Context samples to extract graph-level "environmental prompts"
        attn_out, _ = self.cross_attn(query=queries, key=context_features, value=context_features)

        # Residual and feed-forward networks
        out1 = self.norm(queries + attn_out)
        out2 = self.norm2(out1 + self.ffn(out1))

        # Return the condensed Domain Prompts: (B, num_prompts, d_model)
        return out2


class DualStreamInContextPredictor(nn.Module):
    def __init__(self, tk_dim, struct_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(DualStreamInContextPredictor, self).__init__()
        self.d_model = d_model
        self.num_prompts = 16 # Number of condensed domain prompts

        # 1. Dual-stream preprocessing (Keep features pure, do not add arbitrary masks)
        self.tk_encoder = nn.Sequential(
            nn.Linear(tk_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, d_model // 2)
        )
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, d_model // 2)
        )

        # Role encoding: 0: Positive Support, 1: Negative Support, 2: Query, 3: Domain Prompt
        self.role_emb = nn.Embedding(4, d_model)

        # 2. 🌟 Domain prompt generator (Takes over the 10,000 Context samples)
        self.prompt_generator = DomainPromptGenerator(d_model, num_prompts=self.num_prompts)

        # 3. Main Transformer interaction layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Scoring layer
        self.metric_proj = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def _generate_inductive_mask(self, num_prompts, kp, kn, bsz, device):
        # Sequence composition: [Prompts, Supp_P, Supp_N, Query]
        total_len = num_prompts + kp + kn + bsz
        mask = torch.full((total_len, total_len), float('-inf'), device=device)
        k_sup_end = num_prompts + kp + kn

        # Prompts and Support are fully visible to each other, Query can see Prompts and Support
        mask[:k_sup_end, :k_sup_end] = 0.0
        mask[k_sup_end:, :k_sup_end] = 0.0

        # Queries are invisible to each other (Inductive constraint)
        query_self_mask = torch.eye(bsz, device=device)
        query_self_mask = query_self_mask.masked_fill(query_self_mask == 0, float('-inf'))
        query_self_mask = query_self_mask.masked_fill(query_self_mask == 1, 0.0)
        mask[k_sup_end:, k_sup_end:] = query_self_mask

        return mask

    def forward(self, supp_p_tk, supp_n_tk, query_tk, supp_p_st, supp_n_st, query_st,
                ctx_p_tk=None, ctx_n_tk=None, ctx_p_st=None, ctx_n_st=None):
        kp = supp_p_tk.size(0)
        kn = supp_n_tk.size(0)
        bsz = query_tk.size(0)
        device = supp_p_tk.device

        # Fallback logic
        if ctx_p_tk is None: ctx_p_tk = supp_p_tk
        if ctx_n_tk is None: ctx_n_tk = supp_n_tk
        if ctx_p_st is None: ctx_p_st = supp_p_st
        if ctx_n_st is None: ctx_n_st = supp_n_st

        # Base embedding function
        def embed_fn(tk, st, role_idx):
            feat = torch.cat([self.tk_encoder(tk), self.struct_encoder(st)], dim=-1)
            return feat + self.role_emb(torch.tensor(role_idx, device=device))

        # ====================================================
        # 🌟 1. Generate Domain Prompts using Context
        # ====================================================
        # Preliminary encoding of massive Context features
        ctx_all_tk = torch.cat([ctx_p_tk, ctx_n_tk], dim=0)
        ctx_all_st = torch.cat([ctx_p_st, ctx_n_st], dim=0)

        ctx_all_feat = torch.cat([self.tk_encoder(ctx_all_tk), self.struct_encoder(ctx_all_st)], dim=-1)
        # Add batch dimension and feed into Cross-Attention
        ctx_all_feat = ctx_all_feat.unsqueeze(0) # (1, num_ctx, d_model)

        # Generate the condensed domain prompts (1, num_prompts, d_model)
        domain_prompts = self.prompt_generator(ctx_all_feat).squeeze(0)
        domain_prompts = domain_prompts + self.role_emb(torch.tensor(3, device=device))

        # ====================================================
        # 2. Encode Support and Query
        # ====================================================
        emb_sp = embed_fn(supp_p_tk, supp_p_st, 0)
        emb_sn = embed_fn(supp_n_tk, supp_n_st, 1)
        emb_q = embed_fn(query_tk, query_st, 2)

        # ====================================================
        # 🌟 3. Sequence concatenation and Transformer interaction
        # At this point, the sequence contains: graph-level environmental prompts + few-shot references + samples to be tested
        # ====================================================
        full_sequence = torch.cat([domain_prompts, emb_sp, emb_sn, emb_q], dim=0).unsqueeze(0)
        src_mask = self._generate_inductive_mask(self.num_prompts, kp, kn, bsz, device)

        enc_output = self.transformer(full_sequence, mask=src_mask).squeeze(0)

        # Strip features, do not use Prompts for scoring, only use Prompts as Attention keys/values
        feats_p = enc_output[self.num_prompts: self.num_prompts + kp]
        feats_n = enc_output[self.num_prompts + kp: self.num_prompts + kp + kn]
        feats_q = enc_output[self.num_prompts + kp + kn:]

        feats_p_p = F.normalize(self.metric_proj(feats_p), dim=-1)
        feats_n_p = F.normalize(self.metric_proj(feats_n), dim=-1)
        feats_q_p = F.normalize(self.metric_proj(feats_q), dim=-1)

        sim_p = torch.matmul(feats_q_p, feats_p_p.T)
        sim_n = torch.matmul(feats_q_p, feats_n_p.T)

        sim_p_mean = sim_p.mean(dim=1, keepdim=True)
        sim_n_mean = sim_n.mean(dim=1, keepdim=True)

        logits = (sim_p_mean - sim_n_mean) * self.logit_scale.exp() + self.logit_bias

        return logits