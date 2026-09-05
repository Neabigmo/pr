import pandas as pd
import pytest

from pr_pilot.data.manifest import bilateral_group_split, assert_no_test_leakage, assert_pretraining_disjoint


def _complex_df():
    rows=[]
    # 12 independent bilateral components, 10 rows each -> exact 20-row holdout possible.
    for g in range(12):
        for k in range(10):
            rows.append({"sample_id":f"s{g}_{k}","structure_path":"x.cif","protein_sequence":"AAAA","rna_sequence":"AUGC","protein_hash":f"ph{g}_{k}","rna_hash":f"rh{g}_{k}","protein_cluster_p30":f"P{g}","rna_cluster_r80":f"R{g}","rfam_family":f"RF{g}","mother_sample_id":f"m{g}_{k}","experimental":True})
    return pd.DataFrame(rows)


def test_bilateral_split_uses_whole_components_and_is_order_invariant():
    df=_complex_df(); remain,test=bilateral_group_split(df,20,seed=7)
    remain2,test2=bilateral_group_split(df.sample(frac=1,random_state=3).reset_index(drop=True),20,seed=7)
    assert set(test.sample_id)==set(test2.sample_id)
    assert_no_test_leakage(remain,pd.DataFrame(columns=remain.columns),test,True)
    assert len(test)==20 and len(remain)==100


def test_pretraining_disjoint_catches_family_leakage():
    test=_complex_df().iloc[:1].copy()
    proteins=pd.DataFrame([{"sample_id":"p","structure_path":"p.cif","sequence":"AAAA","sequence_hash":"other","protein_cluster_p30":test.iloc[0].protein_cluster_p30}])
    rnas=pd.DataFrame([{"sample_id":"r","structure_path":"r.cif","sequence":"AUGC","sequence_hash":"other","rna_cluster_r80":"OTHER","rfam_family":"OTHER"}])
    with pytest.raises(AssertionError):
        assert_pretraining_disjoint(proteins,rnas,test)
