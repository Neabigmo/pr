from pathlib import Path

import torch
import yaml

from pr_pilot.baselines.wrappers import ensure_lock_file, pinned_upstream
from pr_pilot.evaluation.battery import MANDATORY_TESTS
from pr_pilot.training.engine import build_model_from_config
from pr_pilot.training.stages import Stage, build_optimizer, configure_stage, make_joint_fully_trainable
from tools.evaluate_official_baselines import _rna_columns
from tools.run_official_baselines import _na_command, _protein_command


REPO = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((REPO / "configs" / "pilot.yaml").read_text(encoding="utf-8"))


def test_locked_upstreams_are_immutable_and_expected():
    lock = ensure_lock_file(REPO)
    assert pinned_upstream("ProteinMPNN", lock).commit == "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
    assert pinned_upstream("NA-MPNN", lock).commit == "9fabc2482092b725e067969fba21297a806b6fda"


def test_na_shared_token_columns_follow_project_augc_order():
    mapping = {"DA": 0, "DC": 1, "DG": 2, "DT": 3}
    assert _rna_columns(mapping) == [0, 3, 2, 1]
    rna_mapping = {"A": 4, "C": 5, "G": 6, "U": 7}
    assert _rna_columns(rna_mapping) == [4, 7, 6, 5]


def test_upstream_command_contracts_have_no_removed_flags():
    p = _protein_command(REPO, Path("ProteinMPNN"), Path("prepared_root"), Path("out"), 1, 2, 900)
    forwarded = p[p.index("--") + 1 :]
    assert "--path_for_training_data" in forwarded
    assert forwarded[forwarded.index("--path_for_training_data") + 1] == "prepared_root"
    assert "--path_for_training_clusters" not in forwarded
    assert "--path_for_valid_clusters" not in forwarded
    assert "--path_for_test_clusters" not in forwarded
    assert "--seed" not in forwarded
    assert forwarded[forwarded.index("--mixed_precision") + 1] == "True"

    n = _na_command(REPO, Path("NA-MPNN"), Path("config.json"), 1)
    assert n[n.index("--") + 1 :] == ["config.json"]


def test_alpha_stage_is_relevance_only_and_global_c_remains_frozen():
    cfg = _config()
    model = build_model_from_config(cfg)
    configure_stage(model, Stage.ALPHA)
    assert all(p.requires_grad for p in model.dmicf.relevance.parameters())
    assert not any(p.requires_grad for p in model.dmicf.interaction.parameters())
    assert not any(p.requires_grad for p in model.dmicf.delta.parameters())
    assert not model.dmicf.global_c.raw.requires_grad


def test_scratch_joint_can_train_every_parameter_including_c():
    cfg = _config()
    model = build_model_from_config(cfg)
    configure_stage(model, Stage.JOINT)
    make_joint_fully_trainable(model, include_global_c=True)
    assert all(p.requires_grad for p in model.parameters())
    opt = cfg["optimization"]
    optimizer = build_optimizer(
        model,
        Stage.JOINT,
        float(opt["lr_heads"]),
        float(opt["lr_projections"]),
        float(opt["lr_encoder_top"]),
        float(opt["lr_encoder_bottom"]),
        float(opt["lr_global_c_joint"]),
        float(opt["weight_decay"]),
        float(opt["layerwise_lr_decay"]),
    )
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.dmicf.global_c.raw) in ids


def test_primary_config_preserves_clean_c_anchor_and_five_hypotheses():
    cfg = _config()
    assert cfg["training_stages"]["global_c"]["hard_context_fraction_late"] == 0.0
    assert cfg["training_stages"]["alpha"]["freeze_interaction_and_delta"] is True
    assert cfg["training_stages"]["joint"]["freeze_global_c"] is True
    assert cfg["training_stages"]["joint"]["unfreezing_mode"] == "gradual"
    expected = set(cfg["evaluation"]["primary_hypotheses"])
    primary = {item.name for item in MANDATORY_TESTS if item.primary}
    assert len(primary) == 5
    assert primary == expected


def test_primary_joint_optimizer_excludes_frozen_global_c():
    cfg = _config()
    model = build_model_from_config(cfg)
    configure_stage(model, Stage.JOINT)
    assert not model.dmicf.global_c.raw.requires_grad
    opt = cfg["optimization"]
    optimizer = build_optimizer(
        model,
        Stage.JOINT,
        float(opt["lr_heads"]),
        float(opt["lr_projections"]),
        float(opt["lr_encoder_top"]),
        float(opt["lr_encoder_bottom"]),
        float(opt["lr_global_c_joint"]),
        float(opt["weight_decay"]),
        float(opt["layerwise_lr_decay"]),
    )
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.dmicf.global_c.raw) not in ids
