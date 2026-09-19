"""Explicit AA--base compatibility adapter used by the frozen A0--A2 protocol.

The module intentionally keeps the upstream priors outside the trainable graph.
It only consumes sequence-neutral prior encoder states, G2 edge features, and
the native partner token at the final matrix-indexing operation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .geometry import geometry_dimension


@dataclass(frozen=True)
class ExplicitMatrixConfig:
    variant: str = "A2"
    geometry: str = "G2"
    hidden_dim: int = 128
    projection_dim: int = 64
    edge_dim: int = 64
    mlp_hidden_dim: int = 128
    dropout: float = 0.1
    rbf_bins: int = 16
    gate_init: float = 0.1


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("gate_init must be strictly between zero and one")
    return math.log(probability / (1.0 - probability))


class ExplicitSelectionAdapter(nn.Module):
    """A0 baseline, A1 global matrix, or A2 global + dynamic matrix.

    Edge indices are always stored as ``(protein_index, rna_index)``.  The
    Protein direction aggregates on the first row and selects an RNA-base
    column; the RNA direction aggregates on the second row and selects a
    protein-AA row.  The same per-edge matrix is used in both directions.
    """

    def __init__(self, config: ExplicitMatrixConfig | None = None):
        super().__init__()
        self.config = config or ExplicitMatrixConfig()
        if self.config.variant not in {"A0", "A1", "A2", "B1", "B2", "B3"}:
            raise ValueError(f"unknown explicit matrix variant: {self.config.variant}")
        if self.config.geometry not in {"G0", "G1", "G2", "G3"}:
            raise ValueError(f"unknown geometry: {self.config.geometry}")

        self.global_matrix = nn.Parameter(torch.zeros(20, 4))
        beta = _logit(float(self.config.gate_init))
        self.protein_gate_logit = nn.Parameter(torch.tensor(beta))
        self.rna_gate_logit = nn.Parameter(torch.tensor(beta))

        if self.config.variant in {"A2", "B1", "B2", "B3"}:
            raw_dim = geometry_dimension(self.config.geometry, self.config.rbf_bins)
            self.use_structural_hidden = self.config.variant != "B2"
            if self.use_structural_hidden:
                self.protein_norm = nn.LayerNorm(self.config.hidden_dim)
                self.rna_norm = nn.LayerNorm(self.config.hidden_dim)
                self.protein_projector = nn.Linear(self.config.hidden_dim, self.config.projection_dim)
                self.rna_projector = nn.Linear(self.config.hidden_dim, self.config.projection_dim)
            self.shared_edge_encoder = nn.Sequential(
                nn.Linear(raw_dim, self.config.edge_dim),
                nn.GELU(),
            )
            input_dim = self.config.edge_dim + (2 * self.config.projection_dim if self.use_structural_hidden else 0)
            self.delta_mlp = nn.Sequential(
                nn.Linear(input_dim, self.config.mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.mlp_hidden_dim, 80),
            )
            # Exact prior-preserving start: C = 0 and Delta-C = 0.
            nn.init.zeros_(self.delta_mlp[-1].weight)
            nn.init.zeros_(self.delta_mlp[-1].bias)

    def _geometry(self, full_geometry: Tensor) -> Tensor:
        expected = geometry_dimension(self.config.geometry, self.config.rbf_bins)
        if full_geometry.shape[-1] == expected:
            return full_geometry
        if self.config.geometry == "G2":
            return full_geometry[:, :expected]
        if self.config.geometry == "G1":
            return full_geometry[:, : 6 * self.config.rbf_bins + 6]
        if self.config.geometry == "G0":
            bins = self.config.rbf_bins
            return torch.cat([full_geometry[:, :bins], full_geometry[:, 6 * bins : 6 * bins + 1]], dim=-1)
        raise ValueError("G3 requires a cache built with G3 geometry")

    def _matrices(
        self,
        p_h: Tensor,
        r_h: Tensor,
        edge_index: Tensor,
        edge_geometry: Tensor,
        token_off: bool,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return p_h.new_zeros((0, 20, 4))
        if token_off:
            # Token-off is a diagnostic control, not a training mode.  It
            # removes the interaction correction rather than pretending that
            # a single alphabet token is a neutral identity control.
            return p_h.new_zeros((edge_index.shape[1], 20, 4))
        if self.config.variant == "A1":
            return self.global_matrix.to(dtype=p_h.dtype).unsqueeze(0).expand(edge_index.shape[1], -1, -1)
        p_index, r_index = edge_index[0], edge_index[1]
        edge = self.shared_edge_encoder(self._geometry(edge_geometry))
        if self.use_structural_hidden:
            p_input = self.protein_projector(self.protein_norm(p_h))
            r_input = self.rna_projector(self.rna_norm(r_h))
            mlp_input = torch.cat([p_input[p_index], r_input[r_index], edge], dim=-1)
        else:
            mlp_input = edge
        delta = self.delta_mlp(mlp_input).reshape(-1, 20, 4)
        if self.config.variant == "B1":
            return delta
        return self.global_matrix.to(dtype=delta.dtype).unsqueeze(0) + delta

    @staticmethod
    def _mean_messages(values: Tensor, target_index: Tensor, length: int) -> Tensor:
        result = values.new_zeros((length, values.shape[-1]))
        if values.numel() == 0:
            return result
        result.index_add_(0, target_index, values)
        count = values.new_zeros((length, 1))
        count.index_add_(0, target_index, values.new_ones((target_index.shape[0], 1)))
        return result / count.clamp_min(1.0)

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
        edge_index_p2r: Tensor,
        edge_geometry_p2r: Tensor,
        token_off: bool = False,
    ) -> dict[str, Tensor]:
        # One matrix is computed independently for each selected edge list,
        # but the parameters are shared. The two lists may differ because
        # K_R->P=8 and K_P->R=12 are direction-specific.
        m_r2p = self._matrices(p_h, r_h, edge_index_r2p, edge_geometry_r2p, token_off)
        m_p2r = self._matrices(p_h, r_h, edge_index_p2r, edge_geometry_p2r, token_off)

        if edge_index_r2p.numel():
            r_partner = r_tokens[edge_index_r2p[1]].long()
            p_values = m_r2p.gather(2, r_partner[:, None, None].expand(-1, 20, 1)).squeeze(-1)
            p_delta = self._mean_messages(p_values, edge_index_r2p[0], p_base.shape[0])
        else:
            p_delta = p_base.new_zeros(p_base.shape)
        if edge_index_p2r.numel():
            p_partner = p_tokens[edge_index_p2r[0]].long()
            r_values = m_p2r.gather(1, p_partner[:, None, None].expand(-1, 1, 4)).squeeze(1)
            r_delta = self._mean_messages(r_values, edge_index_p2r[1], r_base.shape[0])
        else:
            r_delta = r_base.new_zeros(r_base.shape)

        p_gate = torch.sigmoid(self.protein_gate_logit).to(dtype=p_base.dtype)
        r_gate = torch.sigmoid(self.rna_gate_logit).to(dtype=r_base.dtype)
        return {
            "protein_logits": p_base + p_gate * p_delta,
            "rna_logits": r_base + r_gate * r_delta,
            "protein_delta": p_delta,
            "rna_delta": r_delta,
            "protein_gate": p_gate,
            "rna_gate": r_gate,
        }
