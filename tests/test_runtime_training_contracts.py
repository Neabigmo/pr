import numpy as np
import torch

from pr_pilot.model.dmicf import PRBatch, SimpleSparseBackboneEncoder, SparseSequenceContextDecoder, stochastic_depth
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph
from pr_pilot.runtime.gemmi_adapter import P_ATOMS, R_ATOMS, feature_dimensions
from pr_pilot.training.corruption import curriculum_bounds, generate_corruption
from pr_pilot.training.engine import _drop_intra_edges, _drop_pr_edges


def _graph(n: int = 6, node_dim: int = 3, edge_dim: int = 18) -> PolymerGraph:
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_x = torch.zeros(len(edges), edge_dim)
    # Mark adjacent bidirectional edges as covalent in the final edge feature.
    for k, (i, j) in enumerate(edges):
        if abs(i - j) == 1:
            edge_x[k, -1] = 1.0
    return PolymerGraph(
        node_x=torch.randn(n, node_dim),
        edge_index=edge_index,
        edge_x=edge_x,
        sequence=torch.arange(n) % 4,
        interface=torch.tensor([True, True] + [False] * (n - 2)),
        valid=torch.ones(n, dtype=torch.bool),
        fixed=torch.zeros(n, dtype=torch.bool),
        reference_xyz=torch.randn(n, 3),
        chain_index=torch.zeros(n, dtype=torch.long),
        residue_ids=[str(i) for i in range(n)],
    )


def test_feature_dimensions_are_single_source_of_truth():
    dims = feature_dimensions(24, True)
    assert dims["protein_node"] == 15
    assert dims["rna_node"] == 20
    assert len(P_ATOMS) == 5
    assert len(R_ATOMS) == 12
    assert dims["pr_atom_pairs"] == 60
    assert dims["pr_edge"] == 60 * 24 + 60 + 3 + 3 + 9


def test_mask_curriculum_matches_declared_three_bands():
    assert curriculum_bounds(0.0, 0.10, 1.0) == (0.10, 0.40)
    assert curriculum_bounds(0.5, 0.10, 1.0) == (0.20, 0.70)
    assert curriculum_bounds(0.9, 0.10, 1.0) == (0.10, 1.00)


def test_explicit_full_mask_is_reachable_and_exact():
    graph = _graph()
    corruption = generate_corruption(
        graph,
        alphabet_size=4,
        sample_id="x",
        epoch=99,
        seed=1,
        progress=1.0,
        full_mask_probability=1.0,
        wrong_token_fraction=0.0,
    )
    assert corruption.mode.startswith("full:")
    assert corruption.target_mask.all()
    assert not corruption.known.any()
    assert corruption.sampled_fraction == 1.0


def test_intra_edge_dropout_preserves_covalent_edges_and_outgoing_connectivity():
    graph = _graph()
    original = graph.edge_index.shape[1]
    _drop_intra_edges(graph, probability=0.90, seed=7)
    assert graph.edge_index.shape[1] < original
    src = graph.edge_index[0]
    for node in range(graph.node_x.shape[0]):
        assert (src == node).any()
    # Every retained/required covalent directed edge is still present.
    pairs = set(map(tuple, graph.edge_index.T.tolist()))
    for i in range(graph.node_x.shape[0] - 1):
        assert (i, i + 1) in pairs
        assert (i + 1, i) in pairs


def test_pr_dropout_preserves_nearest_edge_per_protein_and_rna_node():
    p = _graph(n=3)
    r = _graph(n=3)
    pr = PRBatch(
        protein_index=torch.tensor([0, 0, 1, 1, 2, 2]),
        rna_index=torch.tensor([0, 1, 1, 2, 0, 2]),
        edge_features=torch.randn(6, 5),
        effective_distance=torch.tensor([2.0, 4.0, 2.5, 5.0, 3.0, 6.0]),
    )
    sample = ComplexTensorSample("x", p, r, pr)
    _drop_pr_edges(sample, probability=0.99, seed=3)
    for node in torch.unique(pr.protein_index):
        assert (sample.pr.protein_index == node).any()
    for node in torch.unique(pr.rna_index):
        assert (sample.pr.rna_index == node).any()


def test_stochastic_depth_is_identity_in_eval_and_zero_at_probability_one():
    x = torch.randn(4, 8)
    assert torch.equal(stochastic_depth(x, 0.5, training=False), x)
    assert torch.count_nonzero(stochastic_depth(x, 1.0, training=True)) == 0


def test_sparse_reductions_support_bfloat16_autocast():
    graph = _graph(n=4, node_dim=3, edge_dim=2)
    encoder = SimpleSparseBackboneEncoder(node_in=3, edge_in=2, hidden=8, layers=1)
    decoder = SparseSequenceContextDecoder(alphabet=4, edge_in=2, hidden=8, layers=1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        encoded = encoder(graph.node_x, graph.edge_index, graph.edge_x)
        decoded = decoder(encoded, graph.edge_index, graph.edge_x, graph.sequence, graph.valid)
    assert encoded.dtype in {torch.float32, torch.bfloat16}
    assert decoded.shape == (4, 8)
