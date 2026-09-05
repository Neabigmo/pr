from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pr_pilot.data.manifest import _single_components
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.gemmi_adapter_legacy import feature_dimensions
from pr_pilot.model.dmicf import JointPriorAndFieldModel
from pr_pilot.training.refit import selected_epoch_count, selected_schedule_horizon
from pr_pilot.training.stages import Stage, apply_joint_unfreezing, configure_stage


def _load_tool(name: str):
    path = Path("tools") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model() -> JointPriorAndFieldModel:
    dims = feature_dimensions(4, True)
    return JointPriorAndFieldModel(
        dims["protein_node"],
        dims["protein_edge"],
        dims["rna_node"],
        dims["rna_edge"],
        dims["pr_edge"],
        hidden=16,
        protein_layers=2,
        rna_layers=2,
        decoder_layers=1,
        interaction_layers=1,
    )


def test_na_mpnn_ppm_columns_are_project_augc_order():
    tool = _load_tool("evaluate_official_baselines")
    shared = {"DA": 21, "DC": 22, "DG": 23, "DT": 24}
    assert tool._rna_columns(shared) == [21, 24, 23, 22]
    direct = {"A": 26, "C": 27, "G": 28, "U": 29}
    assert tool._rna_columns(direct) == [26, 29, 28, 27]


def test_alpha_stage_owns_only_relevance_and_joint_keeps_c_frozen():
    model = _tiny_model()
    configure_stage(model, Stage.ALPHA)
    assert any(p.requires_grad for p in model.dmicf.relevance.parameters())
    assert not any(p.requires_grad for p in model.dmicf.interaction.parameters())
    assert not any(p.requires_grad for p in model.dmicf.delta.parameters())
    assert not model.dmicf.global_c.raw.requires_grad

    configure_stage(model, Stage.JOINT, joint_unfreezing_mode="pretrained_gradual")
    assert not model.dmicf.global_c.raw.requires_grad
    assert model.dmicf.raw_lambda_p.requires_grad
    assert model.dmicf.raw_lambda_r.requires_grad


def test_scratch_joint_unfreezing_releases_random_encoders_immediately():
    model = _tiny_model()
    configure_stage(model, Stage.JOINT, joint_unfreezing_mode="all_trainable_from_start")
    released = apply_joint_unfreezing(model, 0.0, mode="all_trainable_from_start")
    assert released["protein"] == 2
    assert released["rna"] == 2
    assert all(p.requires_grad for p in model.protein_encoder.parameters())
    assert all(p.requires_grad for p in model.rna_encoder.parameters())
    assert not model.dmicf.global_c.raw.requires_grad


def test_refit_reads_selected_prefix_and_original_horizon(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 11,
            "selected_epoch_count": 12,
            "schedule_horizon_epochs": 150,
        },
        checkpoint,
    )
    cfg = {"training_stages": {"joint": {"max_epochs": 150}}}
    assert selected_epoch_count(checkpoint) == 12
    assert selected_schedule_horizon(checkpoint, cfg, Stage.JOINT) == 150


def test_canonical_interface_is_independent_of_message_graph_hyperparameters():
    near_p = SimpleNamespace(atoms={"CA": np.array([0.0, 0.0, 0.0], dtype=np.float32)})
    far_p = SimpleNamespace(atoms={"CA": np.array([30.0, 0.0, 0.0], dtype=np.float32)})
    near_r = SimpleNamespace(atoms={"C4'": np.array([5.5, 0.0, 0.0], dtype=np.float32)})
    far_r = SimpleNamespace(atoms={"C4'": np.array([50.0, 0.0, 0.0], dtype=np.float32)})
    a = GemmiStructureAdapter(pr_cutoff_angstrom=7.0, pr_max_neighbors=1, canonical_interface_cutoff_angstrom=6.0)
    b = GemmiStructureAdapter(pr_cutoff_angstrom=12.0, pr_max_neighbors=99, canonical_interface_cutoff_angstrom=6.0)
    pa, ra = a._canonical_interface_masks([near_p, far_p], [near_r, far_r])
    pb, rb = b._canonical_interface_masks([near_p, far_p], [near_r, far_r])
    assert torch.equal(pa, pb)
    assert torch.equal(ra, rb)
    assert pa.tolist() == [True, False]
    assert ra.tolist() == [True, False]


def test_prior_component_builder_uses_p30_and_union_of_r80_rfam():
    protein = pd.DataFrame(
        {
            "sample_id": ["p1", "p2", "p3"],
            "protein_cluster_p30": ["P1", "P1", "P2"],
        }
    )
    p_components = sorted(sorted(x) for x in _single_components(protein, "protein"))
    assert p_components == [["p1", "p2"], ["p3"]]

    rna = pd.DataFrame(
        {
            "sample_id": ["r1", "r2", "r3"],
            "rna_cluster_r80": ["R1", "R2", "R3"],
            "rfam_family": ["RF1", "RF1", "RF2"],
        }
    )
    r_components = sorted(sorted(x) for x in _single_components(rna, "rna"))
    assert r_components == [["r1", "r2"], ["r3"]]
