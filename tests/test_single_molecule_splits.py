import pandas as pd
import pytest

from pr_pilot.data.manifest import assert_single_molecule_split_disjoint, single_molecule_group_split


def _protein_rows(groups=15, per_group=10):
    rows = []
    for g in range(groups):
        for k in range(per_group):
            rows.append(
                {
                    "sample_id": f"p{g}_{k}",
                    "structure_path": "p.cif",
                    "sequence": "A" * (30 + (k % 3)),
                    "sequence_hash": f"ph{g}_{k}",
                    "protein_cluster_p30": f"P{g}",
                }
            )
    return pd.DataFrame(rows)


def _rna_rows(groups=15, per_group=10):
    rows = []
    for g in range(groups):
        for k in range(per_group):
            rows.append(
                {
                    "sample_id": f"r{g}_{k}",
                    "structure_path": "r.cif",
                    "sequence": "AUGC" + "A" * (k % 3),
                    "sequence_hash": f"rh{g}_{k}",
                    "rna_cluster_r80": f"R{g}",
                    "rfam_family": f"RF{g}",
                }
            )
    return pd.DataFrame(rows)


def test_protein_train_val_is_p30_and_exact_disjoint():
    frame = _protein_rows()
    train, val = single_molecule_group_split(frame, "protein", 100, 20, seed=11)
    assert len(train) == 100 and len(val) == 20
    assert_single_molecule_split_disjoint(train, val, "protein")
    assert set(train.protein_cluster_p30).isdisjoint(set(val.protein_cluster_p30))


def test_rna_train_val_is_r80_rfam_and_exact_disjoint():
    frame = _rna_rows()
    train, val = single_molecule_group_split(frame, "rna", 100, 20, seed=12)
    assert len(train) == 100 and len(val) == 20
    assert_single_molecule_split_disjoint(train, val, "rna")
    assert set(train.rna_cluster_r80).isdisjoint(set(val.rna_cluster_r80))
    assert set(train.rfam_family).isdisjoint(set(val.rfam_family))


def test_single_molecule_audit_rejects_shared_cluster():
    train = _protein_rows(groups=2, per_group=2).iloc[:2].copy()
    val = _protein_rows(groups=2, per_group=2).iloc[2:3].copy()
    val.loc[:, "protein_cluster_p30"] = train.iloc[0].protein_cluster_p30
    with pytest.raises(AssertionError, match="P30 leakage"):
        assert_single_molecule_split_disjoint(train, val, "protein")
