import torch

from pr_pilot.model.dmicf import JointPriorAndFieldModel
from pr_pilot.runtime.dataset_adapter import PolymerGraph
from pr_pilot.runtime.manifest_dataset import _apply_canonical_interface
from pr_pilot.training.refit import schedule_progress
from pr_pilot.training.stages import Stage, apply_joint_unfreezing, configure_stage
from tools.evaluate_official_baselines import _rna_columns


def _model() -> JointPriorAndFieldModel:
    return JointPriorAndFieldModel(
        protein_node_in=3,
        protein_edge_in=2,
        rna_node_in=4,
        rna_edge_in=2,
        pr_edge_in=5,
        hidden=8,
        protein_layers=2,
        rna_layers=2,
        decoder_layers=1,
        interaction_layers=1,
        drop_path=0.0,
    )


def _graph() -> PolymerGraph:
    return PolymerGraph(
        node_x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_x=torch.randn(4, 2),
        sequence=torch.tensor([0, 1, 2]),
        interface=torch.tensor([False, True, False]),
        valid=torch.ones(3, dtype=torch.bool),
        fixed=torch.zeros(3, dtype=torch.bool),
        reference_xyz=torch.randn(3, 3),
        chain_index=torch.zeros(3, dtype=torch.long),
        residue_ids=["A:1:ALA", "A:2:GLY", "A:3:SER"],
    )


def test_nampnn_shared_token_columns_are_canonical_augc_not_acgu():
    mapping = {
        "DA": 21,
        "DC": 22,
        "DG": 23,
        "DT": 24,
        "A": 21,
        "C": 22,
        "G": 23,
        "U": 24,
    }
    assert _rna_columns(mapping) == [21, 24, 23, 22]


def test_alpha_stage_trains_relevance_only_inside_dmicf():
    model = _model()
    # Make the model unambiguously post-Delta rather than a fresh scratch model.
    with torch.no_grad():
        model.dmicf.delta.out.weight[0, 0] = 1.0
    configure_stage(model, Stage.ALPHA)
    assert any(p.requires_grad for p in model.dmicf.relevance.parameters())
    assert not any(p.requires_grad for p in model.dmicf.interaction.parameters())
    assert not any(p.requires_grad for p in model.dmicf.delta.parameters())
    assert not model.dmicf.global_c.raw.requires_grad


def test_primary_joint_keeps_global_c_frozen():
    model = _model()
    with torch.no_grad():
        model.dmicf.delta.out.weight[0, 0] = 1.0
        model.dmicf.relevance.score.weight[0, 0] = 1.0
    configure_stage(model, Stage.JOINT)
    assert not getattr(model, "_scratch_joint_mode")
    assert not model.dmicf.global_c.raw.requires_grad
    apply_joint_unfreezing(model, 1.0)
    assert not model.dmicf.global_c.raw.requires_grad


def test_fresh_scratch_joint_is_all_trainable_from_step_zero():
    model = _model()
    configure_stage(model, Stage.JOINT)
    assert getattr(model, "_scratch_joint_mode")
    assert all(p.requires_grad for p in model.parameters())
    apply_joint_unfreezing(model, 0.5)
    assert all(p.requires_grad for p in model.parameters())


def test_refit_schedule_replays_development_prefix_not_compressed():
    horizon = 150
    selected = 20
    # Refit epoch 19 must remain at development progress 19/149, not 100%.
    assert schedule_progress(selected - 1, horizon) == (selected - 1) / (horizon - 1)
    assert schedule_progress(selected - 1, horizon) < 0.2


def test_canonical_interface_mask_is_residue_id_based_not_graph_based():
    graph = _graph()
    # Deliberately replace the adapter-created interface labels.
    _apply_canonical_interface(graph, {"A:1:ALA", "A:3:SER"}, "protein")
    assert graph.interface.tolist() == [True, False, True]
