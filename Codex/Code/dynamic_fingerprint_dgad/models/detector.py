from __future__ import annotations

import torch
from torch import nn


class DynamicFingerprintDetector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, in_dim] or [batch, seq_len, in_dim]
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        z = self.input_proj(x)
        z = self.encoder(z)
        logits = self.score_head(z).squeeze(-1)
        return logits.squeeze(0) if squeeze else logits


class FeatureTokenDetector(nn.Module):
    """Encodes fingerprint groups as semantic feature tokens before scoring.

    Input layout is expected to be:
    [nCN, CP1..CPK, optional temporal features, d_nCN, d_CP1..d_CPK, optional temporal deltas].
    """

    def __init__(
        self,
        cheb_order: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 4,
        include_edge_surprise: bool = False,
        include_node_activity: bool = False,
    ):
        super().__init__()
        self.cheb_order = cheb_order
        self.hidden_dim = hidden_dim
        self.use_moe = use_moe
        self.include_edge_surprise = include_edge_surprise
        self.include_node_activity = include_node_activity
        if use_moe and num_experts != 4:
            raise ValueError("semantic feature-token MoE expects exactly 4 experts")

        self.local_struct = self._group_encoder(1, hidden_dim)
        self.global_struct = self._group_encoder(cheb_order, hidden_dim)
        local_temp_dim = 1 + int(include_node_activity) + (2 if include_edge_surprise else 0)
        self.local_temp = self._group_encoder(local_temp_dim, hidden_dim)
        self.global_temp = self._group_encoder(cheb_order, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb = nn.Parameter(torch.randn(1, 5, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.feature_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if use_moe:
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(num_experts)
                ]
            )
            self.gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_experts))

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _group_encoder(in_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        k = self.cheb_order
        ncn = x[:, 0:1]
        cp = x[:, 1 : 1 + k]
        cursor = 1 + k
        local_temporal_parts = []
        if self.include_node_activity:
            local_temporal_parts.append(x[:, cursor : cursor + 1])
            cursor += 1
        if self.include_edge_surprise:
            local_temporal_parts.append(x[:, cursor : cursor + 1])
            cursor += 1
        dncn = x[:, cursor : cursor + 1]
        dcp = x[:, cursor + 1 : cursor + 1 + k]
        local_temporal_parts.append(dncn)
        cursor = cursor + 1 + k
        if self.include_edge_surprise:
            local_temporal_parts.append(x[:, cursor : cursor + 1])
        local_temporal = torch.cat(local_temporal_parts, dim=1)

        tokens = torch.stack(
            [
                self.local_struct(ncn),
                self.global_struct(cp),
                self.local_temp(local_temporal),
                self.global_temp(dcp),
            ],
            dim=1,
        )
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return tokens + self.type_emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_edges, feature_dim]
        tokens = self._tokens(x)
        encoded = self.feature_encoder(tokens)
        z = encoded[:, 0]
        if self.use_moe:
            weights = torch.softmax(self.gate(z), dim=-1)
            expert_inputs = [encoded[:, i + 1] for i in range(len(self.experts))]
            expert_out = torch.stack(
                [expert(token_z) for expert, token_z in zip(self.experts, expert_inputs)],
                dim=1,
            )
            z = torch.sum(weights.unsqueeze(-1) * expert_out, dim=1)
        return z

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.score_head(z).squeeze(-1)
        if return_embedding:
            return logits, z
        return logits


class FeatureTokenProtoMoE(nn.Module):
    """Feature-token encoder with prototype routing over anomaly score experts."""

    def __init__(
        self,
        cheb_order: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_experts: int = 4,
        router_temperature: float = 1.0,
        router_top_k: int = 0,
        router_entropy_lambda: float = 0.0,
        load_balance_lambda: float = 0.0,
        include_edge_surprise: bool = False,
        include_node_activity: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router_temperature = router_temperature
        self.router_top_k = router_top_k
        self.router_entropy_lambda = router_entropy_lambda
        self.load_balance_lambda = load_balance_lambda
        self.encoder = FeatureTokenDetector(
            cheb_order=cheb_order,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_moe=False,
            include_edge_surprise=include_edge_surprise,
            include_node_activity=include_node_activity,
        )
        self.prototypes = nn.Parameter(torch.randn(num_experts, hidden_dim) * 0.02)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_experts)
            ]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def _route(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = torch.nn.functional.normalize(z, dim=-1)
        proto_norm = torch.nn.functional.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(z_norm, proto_norm.t()) / max(self.router_temperature, 1e-6)
        return torch.softmax(logits, dim=-1)

    def _sparse_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.router_top_k <= 0 or self.router_top_k >= self.num_experts:
            return weights
        top_values, top_indices = torch.topk(weights, k=self.router_top_k, dim=-1)
        sparse = torch.zeros_like(weights).scatter(dim=-1, index=top_indices, src=top_values)
        return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _router_aux_loss(self, weights: torch.Tensor) -> torch.Tensor:
        aux = torch.tensor(0.0, device=weights.device)
        if self.router_entropy_lambda > 0.0:
            entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
            aux = aux - self.router_entropy_lambda * entropy
        if self.load_balance_lambda > 0.0:
            usage = weights.mean(dim=0)
            balance = self.num_experts * torch.sum(usage ** 2)
            aux = aux + self.load_balance_lambda * balance
        return aux

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        weights = self._route(z)
        sparse_weights = self._sparse_weights(weights)
        expert_logits = torch.stack([expert(z).squeeze(-1) for expert in self.experts], dim=1)
        logits = torch.sum(sparse_weights * expert_logits, dim=1)
        outputs: list[torch.Tensor] = [logits]
        if return_embedding:
            outputs.append(z)
        if return_aux:
            outputs.append(self._router_aux_loss(weights))
        return tuple(outputs) if len(outputs) > 1 else logits


class CrossAttentionBlock(nn.Module):
    """Bidirectional attention between structure tokens and temporal tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.struct_norm = nn.LayerNorm(hidden_dim)
        self.temp_norm = nn.LayerNorm(hidden_dim)
        self.struct_to_temp = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temp_to_struct = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.struct_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.temp_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, struct_tokens: torch.Tensor, temp_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_query = self.struct_norm(struct_tokens)
        t_context = self.temp_norm(temp_tokens)
        s_update, _ = self.struct_to_temp(s_query, t_context, t_context, need_weights=False)
        struct_tokens = struct_tokens + self.dropout(s_update)

        t_query = self.temp_norm(temp_tokens)
        s_context = self.struct_norm(struct_tokens)
        t_update, _ = self.temp_to_struct(t_query, s_context, s_context, need_weights=False)
        temp_tokens = temp_tokens + self.dropout(t_update)

        struct_tokens = struct_tokens + self.dropout(self.struct_ffn(struct_tokens))
        temp_tokens = temp_tokens + self.dropout(self.temp_ffn(temp_tokens))
        return struct_tokens, temp_tokens


class CrossAttentionDetector(nn.Module):
    """Separates structure and temporal feature tokens, then lets them attend to each other."""

    def __init__(
        self,
        cheb_order: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        include_edge_surprise: bool = False,
        include_node_activity: bool = False,
    ):
        super().__init__()
        self.cheb_order = cheb_order
        self.include_edge_surprise = include_edge_surprise
        self.include_node_activity = include_node_activity

        self.local_struct = self._group_encoder(1, hidden_dim)
        self.global_struct = self._group_encoder(cheb_order, hidden_dim)
        local_temp_dim = 1 + int(include_node_activity) + (2 if include_edge_surprise else 0)
        self.local_temp = self._group_encoder(local_temp_dim, hidden_dim)
        self.global_temp = self._group_encoder(cheb_order, hidden_dim)

        self.struct_type_emb = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.temp_type_emb = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _group_encoder(in_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def _split_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.cheb_order
        ncn = x[:, 0:1]
        cp = x[:, 1 : 1 + k]
        cursor = 1 + k
        local_temporal_parts = []
        if self.include_node_activity:
            local_temporal_parts.append(x[:, cursor : cursor + 1])
            cursor += 1
        if self.include_edge_surprise:
            local_temporal_parts.append(x[:, cursor : cursor + 1])
            cursor += 1
        dncn = x[:, cursor : cursor + 1]
        dcp = x[:, cursor + 1 : cursor + 1 + k]
        local_temporal_parts.append(dncn)
        cursor = cursor + 1 + k
        if self.include_edge_surprise:
            local_temporal_parts.append(x[:, cursor : cursor + 1])
        local_temporal = torch.cat(local_temporal_parts, dim=1)

        struct_tokens = torch.stack(
            [
                self.local_struct(ncn),
                self.global_struct(cp),
            ],
            dim=1,
        )
        temp_tokens = torch.stack(
            [
                self.local_temp(local_temporal),
                self.global_temp(dcp),
            ],
            dim=1,
        )
        return struct_tokens + self.struct_type_emb, temp_tokens + self.temp_type_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        struct_tokens, temp_tokens = self._split_tokens(x)
        for block in self.blocks:
            struct_tokens, temp_tokens = block(struct_tokens, temp_tokens)
        z = torch.cat([struct_tokens.mean(dim=1), temp_tokens.mean(dim=1)], dim=1)
        return self.score_head(z).squeeze(-1)
