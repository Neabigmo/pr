import torch

from pr_pilot.model.dmicf import JointPriorAndFieldModel
from pr_pilot.training.stages import Stage, configure_stage


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


def test_explicit_pretrained_joint_does_not_misclassify_zero_unused_heads_as_scratch():
    model = _model()
    # A valid dual-prior checkpoint may still have exact-zero DeltaC/alpha heads.
    assert torch.count_nonzero(model.dmicf.delta.out.weight) == 0
    assert torch.count_nonzero(model.dmicf.relevance.score.weight) == 0
    configure_stage(model, Stage.JOINT, scratch_joint=False)
    assert not getattr(model, "_scratch_joint_mode")
    assert not model.dmicf.global_c.raw.requires_grad
    assert not all(parameter.requires_grad for parameter in model.parameters())


def test_explicit_scratch_joint_is_fully_trainable_from_step_zero():
    model = _model()
    configure_stage(model, Stage.JOINT, scratch_joint=True)
    assert getattr(model, "_scratch_joint_mode")
    assert all(parameter.requires_grad for parameter in model.parameters())
