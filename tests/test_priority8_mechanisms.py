import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from run_priority8_mechanisms import _degree_preserving_rewire, _drop_edges  # noqa: E402


def test_degree_preserving_rewire_keeps_both_endpoint_degree_sequences():
    edges = np.asarray([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 3]], dtype=np.int64)
    rewired = _degree_preserving_rewire(edges, np.random.default_rng(17))
    assert rewired.shape == edges.shape
    assert len(set(map(tuple, rewired.T))) == edges.shape[1]
    for axis in (0, 1):
        assert np.array_equal(np.bincount(edges[axis]), np.bincount(rewired[axis]))


def test_edge_dropout_preserves_each_active_target():
    edges = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=torch.long)
    geometry = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    dropped = _drop_edges(edges, geometry, probability=1.0, seed=17, target_axis=0)
    assert set(dropped["edge_index"][0].tolist()) == {0, 1}
    assert dropped["edge_index"].shape[1] == dropped["geometry"].shape[0]
