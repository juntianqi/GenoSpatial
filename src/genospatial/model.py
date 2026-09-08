"""Frozen GenoSpatial v12_r main-model architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


class GenoSpatialModel(nn.Module):
    """Sequence-to-expression model conditioned on cell, environment and CCC."""

    def __init__(
        self,
        seq_len: int = 624,
        seq_channels: int = 1920,
        num_major_types: int = 10,
        cell_embed_dim: int = 128,
        env_dim: int = 128,
        ccc_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_factors: int = 8,
        branch_dropout_probability: float = 0.25,
        predictor_dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.seq_channels = seq_channels
        self.hidden_dim = hidden_dim
        self.num_factors = num_factors
        self.branch_dropout_probability = branch_dropout_probability

        self.seq_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=seq_channels,
                out_channels=512,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(
                in_channels=512,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.cell_embedding = nn.Embedding(
            num_embeddings=num_major_types, embedding_dim=cell_embed_dim
        )
        self.film_generator = nn.Sequential(
            nn.Linear(cell_embed_dim, hidden_dim * 2), nn.GELU()
        )

        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, hidden_dim * num_factors),
            nn.LayerNorm(hidden_dim * num_factors),
        )
        self.ccc_proj = nn.Sequential(
            nn.Linear(ccc_dim, hidden_dim * num_factors),
            nn.LayerNorm(hidden_dim * num_factors),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.ccc_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(predictor_dropout_probability),
            nn.Linear(64, 1),
        )

        self.saved_env_attn_weights = None
        self.saved_ccc_attn_weights = None

    def _drop_context_branches(
        self, environment_output: torch.Tensor, ccc_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the notebook's independent sample-level branch masks."""

        batch_size = environment_output.size(0)
        environment_mask = (
            torch.rand(batch_size, 1, device=environment_output.device)
            < self.branch_dropout_probability
        )
        ccc_mask = (
            torch.rand(batch_size, 1, device=ccc_output.device)
            < self.branch_dropout_probability
        )
        environment_output = environment_output.masked_fill(environment_mask, 0.0)
        ccc_output = ccc_output.masked_fill(ccc_mask, 0.0)
        return environment_output, ccc_output

    def forward(
        self,
        seq_feat: torch.Tensor,
        major_cell_idx: torch.Tensor,
        env_feat: torch.Tensor,
        ccc_feat: torch.Tensor,
    ) -> torch.Tensor:
        seq_feat = seq_feat.permute(0, 2, 1)
        z_seq = self.seq_conv(seq_feat)

        z_cell = self.cell_embedding(major_cell_idx)
        film_params = self.film_generator(z_cell)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        z_seq_film = z_seq * (1.0 + gamma) + beta
        z_seq_film = z_seq_film.permute(0, 2, 1)
        z_baseline = z_seq_film.mean(dim=1)

        batch_size = z_seq_film.size(0)
        z_env_flat = self.env_proj(env_feat)
        z_ccc_flat = self.ccc_proj(ccc_feat)
        query_environment = z_env_flat.view(
            batch_size, self.num_factors, self.hidden_dim
        )
        query_ccc = z_ccc_flat.view(
            batch_size, self.num_factors, self.hidden_dim
        )

        environment_output, environment_weights = self.cross_attn(
            query=query_environment, key=z_seq_film, value=z_seq_film
        )
        ccc_output, ccc_weights = self.ccc_attn(
            query=query_ccc, key=z_seq_film, value=z_seq_film
        )

        if not self.training:
            self.saved_env_attn_weights = environment_weights.detach()
            self.saved_ccc_attn_weights = ccc_weights.detach()

        environment_output = environment_output.sum(dim=1)
        ccc_output = ccc_output.sum(dim=1)
        if self.training:
            environment_output, ccc_output = self._drop_context_branches(
                environment_output, ccc_output
            )

        z_final = self.fusion_norm(z_baseline + environment_output + ccc_output)
        expression_prediction = self.predictor(z_final)
        return expression_prediction.squeeze(-1)

