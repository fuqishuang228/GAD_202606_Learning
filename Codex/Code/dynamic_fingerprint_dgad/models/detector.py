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


class MechanismSoftMoE(nn.Module):
    """Soft MoE whose experts correspond to explicit fingerprint anomaly mechanisms.

    Default 10-dim core layout:
    [nCN, CP1, CP2, CP3, CP4, d_nCN, d_CP1, d_CP2, d_CP3, d_CP4].
    Optional temporal features may be present in the input for compatibility, but the
    four mechanism experts only consume the core structural and delta fingerprints.
    """

    def __init__(
        self,
        cheb_order: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        include_edge_surprise: bool = False,
        include_node_activity: bool = False,
        router_mode: str = "learned",
    ):
        super().__init__()
        if cheb_order != 4:
            raise ValueError("MechanismSoftMoE currently expects cheb_order=4 for CP1..CP4 experts")
        if router_mode not in {"learned", "uniform", "global"}:
            raise ValueError(f"unknown mechanism router mode: {router_mode}")
        self.cheb_order = cheb_order
        self.include_edge_surprise = include_edge_surprise
        self.include_node_activity = include_node_activity
        self.num_experts = 4
        self.router_mode = router_mode
        self.input_dim = 2 + 2 * cheb_order + int(include_node_activity) + (2 if include_edge_surprise else 0)

        self.local_structure_expert = self._expert(2, hidden_dim, dropout)
        self.low_order_spectral_expert = self._expert(4, hidden_dim, dropout)
        self.high_order_spectral_expert = self._expert(4, hidden_dim, dropout)
        self.temporal_delta_expert = self._expert(5, hidden_dim, dropout)
        self.router = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_experts),
        )
        self.global_router_logits = nn.Parameter(torch.zeros(self.num_experts))

    @staticmethod
    def _expert(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        expert_hidden = max(8, hidden_dim // 2)
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, 1),
        )

    def _core_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.cheb_order
        ncn = x[:, 0:1]
        cp = x[:, 1 : 1 + k]
        cursor = 1 + k
        if self.include_node_activity:
            cursor += 1
        if self.include_edge_surprise:
            cursor += 1
        dncn = x[:, cursor : cursor + 1]
        dcp = x[:, cursor + 1 : cursor + 1 + k]
        return ncn, cp, torch.cat([dncn, dcp], dim=1)

    def mechanism_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ncn, cp, delta = self._core_features(x)
        dncn = delta[:, 0:1]
        dcp = delta[:, 1:5]
        local_structure = torch.cat([ncn, dncn], dim=1)
        low_order = torch.cat([cp[:, 0:2], dcp[:, 0:2]], dim=1)
        high_order = torch.cat([cp[:, 2:4], dcp[:, 2:4]], dim=1)
        temporal_delta = delta
        return local_structure, low_order, high_order, temporal_delta

    def expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        local_structure, low_order, high_order, temporal_delta = self.mechanism_inputs(x)
        return torch.cat(
            [
                self.local_structure_expert(local_structure),
                self.low_order_spectral_expert(low_order),
                self.high_order_spectral_expert(high_order),
                self.temporal_delta_expert(temporal_delta),
            ],
            dim=1,
        )

    def route(self, x: torch.Tensor) -> torch.Tensor:
        if self.router_mode == "uniform":
            return torch.full((x.size(0), self.num_experts), 1.0 / self.num_experts, device=x.device, dtype=x.dtype)
        if self.router_mode == "global":
            weights = torch.softmax(self.global_router_logits, dim=0)
            return weights.unsqueeze(0).expand(x.size(0), -1)
        return torch.softmax(self.router(x), dim=-1)

    def set_router_mode(self, router_mode: str) -> None:
        if router_mode not in {"learned", "uniform", "global"}:
            raise ValueError(f"unknown mechanism router mode: {router_mode}")
        self.router_mode = router_mode

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_aux: bool = False,
        return_router: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        weights = self.route(x)
        expert_logits = self.expert_logits(x)
        logits = torch.sum(weights * expert_logits, dim=1)
        outputs: list[torch.Tensor] = [logits]
        if return_embedding:
            outputs.append(torch.cat([expert_logits, weights], dim=1))
        if return_aux:
            outputs.append(torch.tensor(0.0, device=x.device))
        if return_router:
            outputs.append(weights)
        return tuple(outputs) if len(outputs) > 1 else logits


class MechanismContextSoftMoE(MechanismSoftMoE):
    """Mechanism-guided Soft MoE with a shared full-fingerprint context.

    Each expert keeps its mechanism-specific features, but also receives a compact
    embedding of the full fingerprint so the expert is guided rather than isolated.
    """

    def __init__(
        self,
        cheb_order: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        include_edge_surprise: bool = False,
        include_node_activity: bool = False,
        router_mode: str = "learned",
        context_dim: int | None = None,
    ):
        super().__init__(
            cheb_order=cheb_order,
            hidden_dim=hidden_dim,
            dropout=dropout,
            include_edge_surprise=include_edge_surprise,
            include_node_activity=include_node_activity,
            router_mode=router_mode,
        )
        self.context_dim = context_dim or max(8, hidden_dim // 4)
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_structure_expert = self._expert(2 + self.context_dim, hidden_dim, dropout)
        self.low_order_spectral_expert = self._expert(4 + self.context_dim, hidden_dim, dropout)
        self.high_order_spectral_expert = self._expert(4 + self.context_dim, hidden_dim, dropout)
        self.temporal_delta_expert = self._expert(5 + self.context_dim, hidden_dim, dropout)

    def mechanism_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_inputs = super().mechanism_inputs(x)
        context = self.context_encoder(x)
        return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)


class MechanismPrototypeRouterMoE(nn.Module):
    """Mechanism-context experts routed by anomaly-vs-normal prototype evidence."""

    def __init__(
        self,
        cheb_order: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        include_edge_surprise: bool = False,
        include_node_activity: bool = False,
        router_temperature: float = 1.0,
        prototype_momentum: float = 0.9,
        context_dim: int | None = None,
        use_context: bool = True,
        snapshot_relative_features: bool = False,
        relative_feature_mode: str = "none",
        high_order_view: str = "raw",
        mechanism_feature_mode: str = "raw10",
    ):
        super().__init__()
        if cheb_order != 4:
            raise ValueError("MechanismPrototypeRouterMoE currently expects cheb_order=4 for CP1..CP4 experts")
        self.cheb_order = cheb_order
        self.include_edge_surprise = include_edge_surprise
        self.include_node_activity = include_node_activity
        self.num_experts = 4
        self.router_temperature = router_temperature
        self.prototype_momentum = prototype_momentum
        self.relative_feature_mode = "all" if snapshot_relative_features else relative_feature_mode
        if self.relative_feature_mode not in {"none", "all", "local"}:
            raise ValueError(f"unknown relative_feature_mode: {self.relative_feature_mode}")
        if high_order_view not in {"raw", "multiview"}:
            raise ValueError(f"unknown high_order_view: {high_order_view}")
        if mechanism_feature_mode not in {"raw10", "atlas_local_k2", "atlas_local_k2_fast", "atlas_local_k1"}:
            raise ValueError(f"unknown mechanism_feature_mode: {mechanism_feature_mode}")
        self.high_order_view = high_order_view
        self.mechanism_feature_mode = mechanism_feature_mode
        self.snapshot_relative_features = self.relative_feature_mode == "all"
        if self.mechanism_feature_mode == "atlas_local_k2":
            self.raw_input_dim = 32
            self.input_dim = 32
        elif self.mechanism_feature_mode == "atlas_local_k2_fast":
            self.raw_input_dim = 26
            self.input_dim = 26
        elif self.mechanism_feature_mode == "atlas_local_k1":
            self.raw_input_dim = 14
            self.input_dim = 14
        else:
            self.raw_input_dim = 2 + 2 * cheb_order + int(include_node_activity) + (2 if include_edge_surprise else 0)
            if self.relative_feature_mode == "all":
                self.input_dim = self.raw_input_dim * 2
            elif self.relative_feature_mode == "local":
                self.input_dim = self.raw_input_dim + 2
            else:
                self.input_dim = self.raw_input_dim
        self.use_context = use_context
        self.context_dim = context_dim or max(8, hidden_dim // 4)
        self.rep_dim = max(8, hidden_dim // 2)

        self.context_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.mechanism_feature_mode == "atlas_local_k2":
            mechanism_dims = [12, 4, 4, 16]
        elif self.mechanism_feature_mode == "atlas_local_k2_fast":
            mechanism_dims = [9, 4, 4, 13]
        elif self.mechanism_feature_mode == "atlas_local_k1":
            mechanism_dims = [3, 4, 4, 7]
        elif self.relative_feature_mode == "all":
            mechanism_dims = [4, 8, 8, 10]
        elif self.relative_feature_mode == "local":
            mechanism_dims = [4, 4, 10 if self.high_order_view == "multiview" else 4, 5]
        else:
            mechanism_dims = [2, 4, 10 if self.high_order_view == "multiview" else 4, 5]
        context_extra_dim = self.context_dim if self.use_context else 0
        self.expert_encoders = nn.ModuleList(
            [self._expert_encoder(dim + context_extra_dim, self.rep_dim, dropout) for dim in mechanism_dims]
        )
        self.logit_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.rep_dim),
                    nn.Linear(self.rep_dim, 1),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.register_buffer("normal_prototypes", torch.zeros(self.num_experts, self.rep_dim))
        self.register_buffer("anomaly_prototypes", torch.zeros(self.num_experts, self.rep_dim))

    @staticmethod
    def _expert_encoder(in_dim: int, rep_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, rep_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _core_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.cheb_order
        ncn = x[:, 0:1]
        cp = x[:, 1 : 1 + k]
        cursor = 1 + k
        if self.include_node_activity:
            cursor += 1
        if self.include_edge_surprise:
            cursor += 1
        dncn = x[:, cursor : cursor + 1]
        dcp = x[:, cursor + 1 : cursor + 1 + k]
        return ncn, cp, torch.cat([dncn, dcp], dim=1)

    @staticmethod
    def _high_order_multiview(cp: torch.Tensor, dcp: torch.Tensor) -> torch.Tensor:
        cp1 = cp[:, 0:1]
        cp2 = cp[:, 1:2]
        cp3 = cp[:, 2:3]
        cp4 = cp[:, 3:4]
        dcp1 = dcp[:, 0:1]
        dcp2 = dcp[:, 1:2]
        dcp3 = dcp[:, 2:3]
        dcp4 = dcp[:, 3:4]
        high_mean = 0.5 * (cp3 + cp4)
        d_high_mean = 0.5 * (dcp3 + dcp4)
        high_gap = cp4 - cp3
        d_high_gap = dcp4 - dcp3
        high_low_contrast = high_mean - 0.5 * (cp1 + cp2)
        d_high_low_contrast = d_high_mean - 0.5 * (dcp1 + dcp2)
        return torch.cat(
            [
                cp3,
                cp4,
                dcp3,
                dcp4,
                high_mean,
                d_high_mean,
                high_gap,
                d_high_gap,
                high_low_contrast,
                d_high_low_contrast,
            ],
            dim=1,
        )

    def mechanism_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mechanism_feature_mode == "atlas_local_k2":
            base_inputs = (
                x[:, 0:12],
                x[:, [12, 13, 28, 29]],
                x[:, [14, 15, 30, 31]],
                x[:, 16:32],
            )
            if not self.use_context:
                return base_inputs
            context = self.context_encoder(x)
            return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)

        if self.mechanism_feature_mode == "atlas_local_k2_fast":
            base_inputs = (
                x[:, 0:9],
                x[:, [9, 10, 22, 23]],
                x[:, [11, 12, 24, 25]],
                x[:, 13:26],
            )
            if not self.use_context:
                return base_inputs
            context = self.context_encoder(x)
            return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)

        if self.mechanism_feature_mode == "atlas_local_k1":
            base_inputs = (
                x[:, 0:3],
                x[:, [3, 4, 10, 11]],
                x[:, [5, 6, 12, 13]],
                x[:, 7:14],
            )
            if not self.use_context:
                return base_inputs
            context = self.context_encoder(x)
            return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)

        if self.relative_feature_mode == "all":
            local_structure = x[:, [0, 5, 10, 15]]
            low_order = x[:, [1, 2, 6, 7, 11, 12, 16, 17]]
            high_order = x[:, [3, 4, 8, 9, 13, 14, 18, 19]]
            temporal_delta = x[:, [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]]
            base_inputs = (local_structure, low_order, high_order, temporal_delta)
            if not self.use_context:
                return base_inputs
            context = self.context_encoder(x)
            return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)

        if self.relative_feature_mode == "local":
            high_order = x[:, [3, 4, 8, 9]]
            if self.high_order_view == "multiview":
                cp = x[:, 1:5]
                dcp = x[:, 6:10]
                high_order = self._high_order_multiview(cp, dcp)
            base_inputs = (
                x[:, [0, 5, 10, 11]],
                x[:, [1, 2, 6, 7]],
                high_order,
                x[:, [5, 6, 7, 8, 9]],
            )
            if not self.use_context:
                return base_inputs
            context = self.context_encoder(x)
            return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)

        ncn, cp, delta = self._core_features(x)
        dncn = delta[:, 0:1]
        dcp = delta[:, 1:5]
        high_order = torch.cat([cp[:, 2:4], dcp[:, 2:4]], dim=1)
        if self.high_order_view == "multiview":
            high_order = self._high_order_multiview(cp, dcp)
        base_inputs = (
            torch.cat([ncn, dncn], dim=1),
            torch.cat([cp[:, 0:2], dcp[:, 0:2]], dim=1),
            high_order,
            delta,
        )
        if not self.use_context:
            return base_inputs
        context = self.context_encoder(x)
        return tuple(torch.cat([mechanism_x, context], dim=1) for mechanism_x in base_inputs)

    def expert_representations(self, x: torch.Tensor) -> torch.Tensor:
        mechanism_inputs = self.mechanism_inputs(x)
        reps = [encoder(mechanism_x) for encoder, mechanism_x in zip(self.expert_encoders, mechanism_inputs)]
        return torch.stack(reps, dim=1)

    def expert_logits_from_representations(self, reps: torch.Tensor) -> torch.Tensor:
        logits = [head(reps[:, idx]).squeeze(-1) for idx, head in enumerate(self.logit_heads)]
        return torch.stack(logits, dim=1)

    def expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert_logits_from_representations(self.expert_representations(x))

    def prototype_evidence_from_representations(self, reps: torch.Tensor) -> torch.Tensor:
        reps = torch.nn.functional.normalize(reps, dim=-1)
        anomaly_proto = torch.nn.functional.normalize(self.anomaly_prototypes, dim=-1)
        normal_proto = torch.nn.functional.normalize(self.normal_prototypes, dim=-1)
        anomaly_sim = torch.sum(reps * anomaly_proto.unsqueeze(0), dim=-1)
        normal_sim = torch.sum(reps * normal_proto.unsqueeze(0), dim=-1)
        return anomaly_sim - normal_sim

    def prototype_evidence(self, x: torch.Tensor) -> torch.Tensor:
        return self.prototype_evidence_from_representations(self.expert_representations(x))

    def route_from_evidence(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.softmax(evidence / max(self.router_temperature, 1e-6), dim=-1)

    @torch.no_grad()
    def update_prototypes(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        if not self.training:
            return
        reps = torch.nn.functional.normalize(self.expert_representations(x), dim=-1)
        normal_mask = labels <= 0.5
        anomaly_mask = labels > 0.5
        self._update_class_prototypes(self.normal_prototypes, reps, normal_mask)
        self._update_class_prototypes(self.anomaly_prototypes, reps, anomaly_mask)

    @torch.no_grad()
    def _update_class_prototypes(self, prototypes: torch.Tensor, reps: torch.Tensor, mask: torch.Tensor) -> None:
        if mask.sum() == 0:
            return
        means = torch.nn.functional.normalize(reps[mask].mean(dim=0), dim=-1)
        current_norm = prototypes.norm(dim=-1, keepdim=True)
        initialized = current_norm > 1e-8
        updated = self.prototype_momentum * prototypes + (1.0 - self.prototype_momentum) * means
        updated = torch.nn.functional.normalize(updated, dim=-1)
        prototypes.copy_(torch.where(initialized, updated, means))

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_aux: bool = False,
        return_router: bool = False,
        return_evidence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        reps = self.expert_representations(x)
        expert_logits = self.expert_logits_from_representations(reps)
        evidence = self.prototype_evidence_from_representations(reps)
        weights = self.route_from_evidence(evidence)
        logits = torch.sum(weights * expert_logits, dim=1)
        outputs: list[torch.Tensor] = [logits]
        if return_embedding:
            outputs.append(reps.mean(dim=1))
        if return_aux:
            outputs.append(torch.tensor(0.0, device=x.device))
        if return_router:
            outputs.append(weights)
        if return_evidence:
            outputs.append(evidence)
        return tuple(outputs) if len(outputs) > 1 else logits


class ContextRepresentationMoE(nn.Module):
    """Mechanism expert representations fused with optional unlabeled target context."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        context_mode: str = "none",
    ):
        super().__init__()
        if context_mode not in {"none", "target_mean"}:
            raise ValueError(f"unknown context mode: {context_mode}")
        self.context_mode = context_mode
        self.hidden_dim = hidden_dim
        self.num_experts = 3
        self.expert_input_dims = [2, 4, 4]
        self.expert_names = ["local_structure", "low_order_spectral", "high_order_spectral"]

        self.local_encoder = self._expert_encoder(2, hidden_dim, dropout)
        self.low_encoder = self._expert_encoder(4, hidden_dim, dropout)
        self.high_encoder = self._expert_encoder(4, hidden_dim, dropout)
        self.type_emb = nn.Parameter(torch.randn(1, 4, hidden_dim) * 0.02)

        if context_mode == "target_mean":
            self.context_encoder = nn.Sequential(
                nn.Linear(20, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.context_encoder = None
        self.register_buffer("target_context_stats", torch.zeros(20))

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.fusion_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _expert_encoder(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def set_target_context(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        stats = torch.cat([mean.reshape(-1), std.reshape(-1)], dim=0).to(self.target_context_stats.device)
        if stats.numel() != 20:
            raise ValueError(f"target context stats must have 20 values, got {stats.numel()}")
        self.target_context_stats.copy_(stats.float())

    def mechanism_representations(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_encoder(x[:, [0, 5]])
        low = self.low_encoder(x[:, [1, 2, 6, 7]])
        high = self.high_encoder(x[:, [3, 4, 8, 9]])
        return torch.stack([local, low, high], dim=1)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        expert_tokens = self.mechanism_representations(x)
        if self.context_mode == "target_mean":
            if self.context_encoder is None:
                raise RuntimeError("context_encoder is not initialized")
            context = self.context_encoder(self.target_context_stats.unsqueeze(0))
            context = context.unsqueeze(1).expand(x.size(0), -1, -1)
            tokens = torch.cat([context, expert_tokens], dim=1)
            return tokens + self.type_emb[:, :4]
        return expert_tokens + self.type_emb[:, 1:4]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fusion_encoder(self.tokens(x))
        return z.mean(dim=1)

    def expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        reps = self.mechanism_representations(x)
        logits = [self.final_head(reps[:, idx]).squeeze(-1) for idx in range(self.num_experts)]
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def context_diagnostics(self) -> dict:
        stats = self.target_context_stats.detach().cpu()
        diagnostics = {
            "context_mode": self.context_mode,
            "target_labels_used_for_context": False,
            "hidden_dim": self.hidden_dim,
            "num_experts": self.num_experts,
            "expert_input_dims": self.expert_input_dims,
        }
        if self.context_mode == "target_mean":
            diagnostics["target_context_mean"] = stats[:10].tolist()
            diagnostics["target_context_std"] = stats[10:].tolist()
            if self.context_encoder is not None:
                context_token = self.context_encoder(self.target_context_stats.unsqueeze(0))
                diagnostics["context_token_norm"] = float(context_token.norm(dim=-1).item())
        return diagnostics

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.final_head(z).squeeze(-1)
        if return_embedding:
            return logits, z
        return logits


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
