import torch

from pr_pilot.inference.sampler import sample_joint
from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch
from pr_pilot.runtime.dataset_adapter import PolymerGraph, ComplexTensorSample


def _graph(n,node_dim,edge_dim,alphabet,interface):
    edge_index=torch.tensor([[0,1],[1,0]],dtype=torch.long) if n>=2 else torch.zeros((2,0),dtype=torch.long)
    edge_x=torch.randn(edge_index.shape[1],edge_dim)
    return PolymerGraph(torch.randn(n,node_dim),edge_index,edge_x,torch.arange(n)%alphabet,torch.tensor(interface,dtype=torch.bool),torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool),torch.randn(n,3),torch.zeros(n,dtype=torch.long),[str(i) for i in range(n)])


def test_spir_changes_only_interface_positions():
    p=_graph(2,3,4,20,[True,False]); r=_graph(2,5,4,4,[True,False])
    pr=PRBatch(torch.tensor([0]),torch.tensor([0]),torch.randn(1,6),torch.tensor([3.0]))
    s=ComplexTensorSample("x",p,r,pr); s.validate()
    m=JointPriorAndFieldModel(3,4,5,4,6,hidden=8,protein_layers=1,rna_layers=1,decoder_layers=1)
    c=sample_joint(m,s,candidates=3,temperature=0.8,seed=3,spir_enabled=True,spir_reopen_fraction=1.0,spir_temperature=0.5,spir_cycles=1)
    for x in c:
        p_changed=x.protein_tokens!=x.pre_spir_protein
        r_changed=x.rna_tokens!=x.pre_spir_rna
        assert not bool(p_changed[~p.interface].any())
        assert not bool(r_changed[~r.interface].any())
