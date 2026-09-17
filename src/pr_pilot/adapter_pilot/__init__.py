"""Frozen-prior conditional Adapter pilot.

This package is intentionally separate from the legacy DM-ICF model.  The
pilot wraps the pinned upstream ProteinMPNN and NA-MPNN implementations, caches
their frozen representations, and trains only a small cross-chain residual.
"""

from .geometry import CrossEdgeSet, build_cross_edges, heavy_contact_audit
from .model import ReciprocalAdapter

__all__ = [
    "CrossEdgeSet",
    "ReciprocalAdapter",
    "build_cross_edges",
    "heavy_contact_audit",
]
