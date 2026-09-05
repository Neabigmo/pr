import pytest
import torch

from pr_pilot.runtime.dataset_adapter import PolymerGraph
from pr_pilot.runtime.manifest_dataset import _assert_frozen_sequence


def _graph(tokens):
    n = len(tokens)
    return PolymerGraph(
        node_x=torch.zeros(n, 2),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_x=torch.zeros(0, 1),
        sequence=torch.tensor(tokens, dtype=torch.long),
        interface=torch.zeros(n, dtype=torch.bool),
        valid=torch.ones(n, dtype=torch.bool),
        fixed=torch.zeros(n, dtype=torch.bool),
        reference_xyz=torch.zeros(n, 3),
        chain_index=torch.zeros(n, dtype=torch.long),
        residue_ids=[f"A:{i+1}:X" for i in range(n)],
    )


def test_protein_runtime_sequence_must_equal_frozen_manifest():
    graph = _graph([0, 1, 2])
    # project protein alphabet begins A,C,D
    _assert_frozen_sequence(graph, "ACD", "protein", "p1")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        _assert_frozen_sequence(graph, "ACE", "protein", "p1")


def test_rna_runtime_sequence_normalizes_t_to_u_but_not_identity_changes():
    graph = _graph([0, 1, 2, 3])
    # project canonical RNA alphabet is A,U,G,C
    _assert_frozen_sequence(graph, "ATGC", "rna", "r1")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        _assert_frozen_sequence(graph, "AGUC", "rna", "r1")
