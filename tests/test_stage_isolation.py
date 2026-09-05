import torch

from pr_pilot.model.dmicf import JointPriorAndFieldModel
from pr_pilot.training.stages import Stage, apply_joint_unfreezing, build_optimizer, configure_stage


def _model():
    return JointPriorAndFieldModel(3, 4, 5, 4, 6, hidden=8, protein_layers=2, rna_layers=2, decoder_layers=1)


def _ids(params):
    return {id(p) for p in params}


def test_alpha_stage_only_trains_relevance():
    model = _model()
    configure_stage(model, Stage.ALPHA)
    trainable = _ids(p for p in model.parameters() if p.requires_grad)
    assert trainable == _ids(model.dmicf.relevance.parameters())
    optimizer = build_optimizer(model, Stage.ALPHA)
    optimized = _ids(p for group in optimizer.param_groups for p in group["params"])
    assert optimized == trainable


def test_joint_never_unfreezes_global_c_anchor():
    model = _model()
    configure_stage(model, Stage.JOINT)
    assert not model.dmicf.global_c.raw.requires_grad
    for progress in (0.0, 0.25, 0.8, 1.0):
        apply_joint_unfreezing(model, progress)
        assert not model.dmicf.global_c.raw.requires_grad


def test_joint_optimizer_excludes_global_c_anchor():
    model = _model()
    configure_stage(model, Stage.JOINT)
    optimizer = build_optimizer(model, Stage.JOINT)
    optimized = _ids(p for group in optimizer.param_groups for p in group["params"])
    assert id(model.dmicf.global_c.raw) not in optimized
    assert id(model.dmicf.raw_lambda_p) in optimized
    assert id(model.dmicf.raw_lambda_r) in optimized
