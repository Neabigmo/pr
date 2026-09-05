from pathlib import Path

import pytest

from pr_pilot.runtime.manifest_dataset import canonical_interface_ids


def test_canonical_interface_uses_full_heavy_atom_geometry_independent_of_pr_graph(tmp_path: Path):
    pdb = tmp_path / "tiny.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1    N ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2   CA ALA A   1       0.500   0.000   0.000  1.00 20.00           C",
                "ATOM      3    C ALA A   1       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4    O ALA A   1       1.500   0.000   0.000  1.00 20.00           O",
                "ATOM      5    P   A B   1       6.900   0.000   0.000  1.00 20.00           P",
                "ATOM      6  C1'   A B   1       6.700   0.000   0.000  1.00 20.00           C",
                "ATOM      7  C3'   A B   1       6.600   0.500   0.000  1.00 20.00           C",
                "ATOM      8  C4'   A B   1       6.400   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    p_ids, r_ids = canonical_interface_ids(pdb, ["A"], ["B"], cutoff=6.0)
    assert any(x.startswith("A:1:ALA") for x in p_ids)
    assert any(x.startswith("B:1:A") for x in r_ids)

    with pytest.raises(ValueError, match="No canonical heavy-atom"):
        canonical_interface_ids(pdb, ["A"], ["B"], cutoff=4.0)
