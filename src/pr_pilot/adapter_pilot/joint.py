"""Adapter V2 bridge for the existing mixed-order/SPIR sampler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import ReciprocalAdapter


@dataclass
class AdapterJointPayload:
    """Frozen prior/cache tensors needed by one complex during decoding."""

    protein_base: Tensor
    rna_base: Tensor
    protein_hidden: Tensor
    rna_hidden: Tensor
    edge_index_r2p: Tensor
    edge_geometry_r2p: Tensor
    edge_index_p2r: Tensor
    edge_geometry_p2r: Tensor


class AdapterJointModel(nn.Module):
    """Expose :class:`ReciprocalAdapter` through the legacy sampler contract.

    The sampler supplies known masks for both polymers.  Unknown partner
    positions are passed as ``False`` masks, so their identity contributes no
    Adapter correction.  The frozen single-molecule logits remain the base
    distribution and are never updated by this bridge.
    """

    def __init__(self, adapter: ReciprocalAdapter, device: torch.device | str = "cpu"):
        super().__init__()
        self.adapter = adapter
        self.device = torch.device(device)
        self._payloads: dict[int, AdapterJointPayload] = {}

    def register(self, sample: Any, payload: AdapterJointPayload) -> None:
        self._payloads[id(sample.pr)] = payload

    def clear(self) -> None:
        self._payloads.clear()

    def _payload(self, pr: Any) -> AdapterJointPayload:
        try:
            return self._payloads[id(pr)]
        except KeyError as exc:
            raise KeyError("sample.pr was not registered with AdapterJointModel") from exc

    def forward(
        self,
        protein_node_x: Tensor,
        protein_edge_index: Tensor,
        protein_edge_x: Tensor,
        rna_node_x: Tensor,
        rna_edge_index: Tensor,
        rna_edge_x: Tensor,
        pr: Any,
        protein_tokens: Tensor,
        rna_tokens: Tensor,
        protein_known: Tensor,
        rna_known: Tensor,
        use_delta: bool = True,
        learned_alpha: bool = True,
    ) -> dict[str, Tensor]:
        del protein_node_x, protein_edge_index, protein_edge_x
        del rna_node_x, rna_edge_index, rna_edge_x, use_delta, learned_alpha
        payload = self._payload(pr)
        device = protein_tokens.device
        return self.adapter(
            payload.protein_base.to(device),
            payload.rna_base.to(device),
            payload.protein_hidden.to(device),
            payload.rna_hidden.to(device),
            protein_tokens.to(device),
            rna_tokens.to(device),
            payload.edge_index_r2p.to(device),
            payload.edge_geometry_r2p.to(device),
            payload.edge_index_p2r.to(device),
            payload.edge_geometry_p2r.to(device),
            protein_known=protein_known.to(device),
            rna_known=rna_known.to(device),
        )
