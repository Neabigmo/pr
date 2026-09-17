"""Small zero-initialized cross-chain residual Adapter."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AdapterConfig:
    geometry: str = "G2"
    aggregation: str = "A2"
    hidden_dim: int = 128
    edge_dim: int = 64
    token_dim: int = 32
    message_dim: int = 128
    layers: int = 1
    dropout: float = 0.1
    rbf_bins: int = 16
    separate_edge_encoders: bool = False


class _Direction(nn.Module):
    def __init__(self, hidden_dim: int, token_dim: int, edge_dim: int, message_dim: int, partner_vocab: int, output_vocab: int, aggregation: str, layers: int, dropout: float):
        super().__init__()
        # The partner token vocabulary and the target output vocabulary are
        # different in the reciprocal directions: RNA(4)->protein(20) and
        # protein(20)->RNA(4).  Keeping them separate prevents the P->R
        # branch from trying to embed amino-acid ids with a four-row table.
        self.token_embedding = nn.Embedding(partner_vocab, token_dim)
        input_dim = hidden_dim * 2 + token_dim + edge_dim
        blocks: list[nn.Module] = []
        for index in range(max(1, layers)):
            blocks.append(nn.Linear(input_dim if index == 0 else message_dim, message_dim))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
        self.message = nn.Sequential(*blocks)
        self.attention = nn.Linear(message_dim, 1) if aggregation in {"A1", "A2"} else None
        self.null_logit = nn.Parameter(torch.zeros(())) if aggregation == "A2" else None
        self.norm = nn.LayerNorm(message_dim)
        self.output = nn.Linear(message_dim, output_vocab)
        # This is the ControlNet-style safety boundary: before training, the
        # adapter is mathematically absent regardless of the random branch.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.aggregation = aggregation

    def forward(self, target_h: Tensor, partner_h: Tensor, partner_tokens: Tensor, edge_index: Tensor, edge_features: Tensor, token_off: bool = False) -> Tensor:
        length = target_h.shape[0]
        if edge_index.numel() == 0:
            return target_h.new_zeros((length, self.output.out_features))
        target_index, partner_index = edge_index[0], edge_index[1]
        token = torch.zeros((partner_index.shape[0], self.token_embedding.embedding_dim), device=target_h.device, dtype=target_h.dtype) if token_off else self.token_embedding(partner_tokens[partner_index])
        message = self.message(torch.cat([target_h[target_index], partner_h[partner_index], edge_features, token], dim=-1))
        if self.aggregation == "A0":
            aggregate = target_h.new_zeros((length, message.shape[-1]))
            aggregate.index_add_(0, target_index, message)
            count = target_h.new_zeros((length, 1))
            count.index_add_(0, target_index, target_h.new_ones((target_index.shape[0], 1)))
            aggregate = aggregate / count.clamp_min(1.0)
        else:
            scores = self.attention(message).squeeze(-1)
            maximum = target_h.new_full((length,), -torch.inf)
            maximum.scatter_reduce_(0, target_index, scores, reduce="amax", include_self=True)
            weights = torch.exp(scores - maximum[target_index])
            denominator = target_h.new_zeros((length,))
            denominator.index_add_(0, target_index, weights)
            if self.null_logit is not None:
                denominator = denominator + torch.exp(self.null_logit.to(dtype=target_h.dtype))
            aggregate = target_h.new_zeros((length, message.shape[-1]))
            aggregate.index_add_(0, target_index, message * (weights / denominator[target_index]).unsqueeze(-1))
        return self.output(self.norm(aggregate))


class ReciprocalAdapter(nn.Module):
    """R→P and P→R residuals over frozen upstream hidden states."""

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        self.config = config or AdapterConfig()
        dims = {"G0": self.config.rbf_bins + 1, "G1": 6 * self.config.rbf_bins + 6, "G2": 6 * self.config.rbf_bins + 6 + 3 + 3 + 6}
        if self.config.geometry not in dims:
            raise ValueError(f"unknown geometry {self.config.geometry}")
        self.raw_geometry_dim = dims[self.config.geometry]
        self.shared_edge_encoder = nn.Sequential(nn.Linear(self.raw_geometry_dim, self.config.edge_dim), nn.GELU())
        self.edge_encoder_r2p = (
            nn.Sequential(nn.Linear(self.raw_geometry_dim, self.config.edge_dim), nn.GELU())
            if self.config.separate_edge_encoders else None
        )
        self.edge_encoder_p2r = (
            nn.Sequential(nn.Linear(self.raw_geometry_dim, self.config.edge_dim), nn.GELU())
            if self.config.separate_edge_encoders else None
        )
        self.r2p = _Direction(self.config.hidden_dim, self.config.token_dim, self.config.edge_dim, self.config.message_dim, 4, 20, self.config.aggregation, self.config.layers, self.config.dropout)
        self.p2r = _Direction(self.config.hidden_dim, self.config.token_dim, self.config.edge_dim, self.config.message_dim, 20, 4, self.config.aggregation, self.config.layers, self.config.dropout)

    def _select_geometry(self, full_geometry: Tensor) -> Tensor:
        bins = self.config.rbf_bins
        if self.config.geometry == "G2":
            return full_geometry
        if self.config.geometry == "G1":
            return full_geometry[:, : 6 * bins + 6]
        return torch.cat([full_geometry[:, :bins], full_geometry[:, 6 * bins : 6 * bins + 1]], dim=-1)

    def _residual(self, direction: str, p_h: Tensor, r_h: Tensor, p_tokens: Tensor, r_tokens: Tensor, edge_index: Tensor, full_geometry: Tensor, token_off: bool) -> Tensor:
        geometry = self._select_geometry(full_geometry)
        if self.config.separate_edge_encoders:
            encoder = self.edge_encoder_r2p if direction == "protein" else self.edge_encoder_p2r
            assert encoder is not None
        else:
            encoder = self.shared_edge_encoder
        edge = encoder(geometry)
        if direction == "protein":
            return self.r2p(p_h, r_h, r_tokens, edge_index, edge, token_off)
        if direction == "rna":
            return self.p2r(r_h, p_h, p_tokens, edge_index[[1, 0]], edge, token_off)
        raise ValueError(direction)

    def forward(
        self,
        p_base: Tensor,
        r_base: Tensor,
        p_h: Tensor,
        r_h: Tensor,
        p_tokens: Tensor,
        r_tokens: Tensor,
        edge_index_r2p: Tensor,
        edge_geometry_r2p: Tensor,
        edge_index_p2r: Tensor | None = None,
        edge_geometry_p2r: Tensor | None = None,
        token_off: bool = False,
    ) -> dict[str, Tensor]:
        # Keep the old single-edge call valid for existing correctness tests and
        # old diagnostic scripts. New experiments pass independent directional
        # edge sets and geometries explicitly.
        if edge_index_p2r is None:
            edge_index_p2r = edge_index_r2p
        if edge_geometry_p2r is None:
            edge_geometry_p2r = edge_geometry_r2p
        p_delta = self._residual("protein", p_h, r_h, p_tokens, r_tokens, edge_index_r2p, edge_geometry_r2p, token_off)
        r_delta = self._residual("rna", p_h, r_h, p_tokens, r_tokens, edge_index_p2r, edge_geometry_p2r, token_off)
        return {"protein_logits": p_base + p_delta, "rna_logits": r_base + r_delta, "protein_delta": p_delta, "rna_delta": r_delta}
