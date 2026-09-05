"""Audited training entrypoint for the six-stage DM-ICF mini-pilot.

The original implementation is retained verbatim in ``engine_legacy.py`` as a
low-level helper library. This module owns the scientifically sensitive control
flow added by the v3 audit:

- explicit pretrained-vs-scratch joint unfreezing semantics;
- global C remains frozen after Stage C;
- validation-selected checkpoints use a sequential teacher-forced joint metric
  rather than a single all-unknown forward pass;
- schedule metadata is recorded for exact full-1000 prefix replay.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training import engine_legacy as _legacy
from pr_pilot.training.losses import balanced_sequence_loss
from pr_pilot.training.stages import (
    Stage,
    apply_joint_unfreezing,
    build_optimizer,
    configure_stage,
    trainable_parameter_report,
)

# Public/semipublic helpers used by existing tools. Keep one implementation of
# tensor augmentation/corruption in engine_legacy while v3 owns orchestration.
build_model_from_config = _legacy.build_model_from_config
_adapter = _legacy._adapter
_autocast = _legacy._autocast
_cosine_schedule = _legacy._cosine_schedule
_one_training_loss = _legacy._one_training_loss
_move_graph = _legacy._move_graph
_move_complex = _legacy._move_complex
_all_interface_corruption = _legacy._all_interface_corruption


def _joint_mode(cfg: dict) -> str:
    return str(
        cfg.get("training_stages", {})
        .get("joint", {})
        .get("unfreezing_mode", "pretrained_gradual")
    )


def _stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest()


def _ordered_design_positions(sample, sample_id: str, mode: str) -> list[tuple[str, int]]:
    protein = [
        ("protein", int(i))
        for i in torch.where(sample.protein.valid & ~sample.protein.fixed)[0]
    ]
    rna = [
        ("rna", int(i))
        for i in torch.where(sample.rna.valid & ~sample.rna.fixed)[0]
    ]

    def ordered(items: list[tuple[str, int]], salt: str) -> list[tuple[str, int]]:
        return sorted(
            items,
            key=lambda item: hashlib.sha256(
                f"{sample_id}|{salt}|{item[0]}|{item[1]}".encode()
            ).hexdigest(),
        )

    p = ordered(protein, mode)
    r = ordered(rna, mode)
    if mode == "protein_first":
        return p + r
    if mode == "rna_first":
        return r + p
    if mode == "mixed":
        return ordered(p + r, "mixed")
    raise ValueError(mode)


@torch.no_grad()
def sequential_joint_normalized_nll(
    model,
    sample,
    order_modes: tuple[str, ...] = ("mixed", "protein_first", "rna_first"),
) -> float:
    """Teacher-forced sequential pseudo-NLL matching the real joint information flow.

    Future positions remain unknown. After a native token has been scored, it is
    revealed to subsequent steps exactly as a teacher-forced decoding prefix. The
    current target token is never marked known before its score is recorded.
    """
    order_scores: list[float] = []
    for mode in order_modes:
        pt = sample.protein.sequence.clone()
        rt = sample.rna.sequence.clone()
        pk = sample.protein.fixed & sample.protein.valid
        rk = sample.rna.fixed & sample.rna.valid
        p_nll: list[float] = []
        r_nll: list[float] = []
        for polymer, index in _ordered_design_positions(sample, sample.sample_id, mode):
            out = model(
                sample.protein.node_x,
                sample.protein.edge_index,
                sample.protein.edge_x,
                sample.rna.node_x,
                sample.rna.edge_index,
                sample.rna.edge_x,
                sample.pr,
                pt,
                rt,
                pk,
                rk,
                use_delta=True,
                learned_alpha=True,
            )
            if polymer == "protein":
                if bool(pk[index]):
                    raise AssertionError("Current Protein target leaked into known prefix")
                logp = F.log_softmax(out["protein_logits"][index].float(), -1)
                p_nll.append(float(-logp[int(pt[index])].cpu()) / math.log(20.0))
                pk[index] = True
            else:
                if bool(rk[index]):
                    raise AssertionError("Current RNA target leaked into known prefix")
                logp = F.log_softmax(out["rna_logits"][index].float(), -1)
                r_nll.append(float(-logp[int(rt[index])].cpu()) / math.log(4.0))
                rk[index] = True
        if not p_nll or not r_nll:
            raise ValueError("Joint validation requires designable positions on both polymers")
        order_scores.append(0.5 * (float(np.mean(p_nll)) + float(np.mean(r_nll))))
    return float(np.mean(order_scores))


@torch.no_grad()
def validate_stage(model, stage: Stage, manifest: ManifestTable, cfg: dict, device: torch.device) -> float:
    """Validation metric used for checkpoint selection.

    Non-joint stages retain the audited legacy metric. Joint selection combines
    Protein-conditional interface NLL, RNA-conditional interface NLL and a
    deterministic three-order sequential joint pseudo-NLL. The expensive sequential
    term is evaluated on a fixed hash-selected validation subset, never selected by
    performance.
    """
    if stage != Stage.JOINT:
        return _legacy.validate_stage(model, stage, manifest, cfg, device)

    model.eval()
    adapter = _adapter(cfg, 0, training=False)
    seed = int(cfg["experiment"]["pilot_seed"])
    subset_n = int(cfg.get("optimization", {}).get("joint_validation_complexes", 20))
    all_ids = [str(x) for x in manifest.df["sample_id"].tolist()]
    selected = set(sorted(all_ids, key=lambda sid: _stable_rank(seed + 7717, sid))[: min(subset_n, len(all_ids))])

    protein_values: list[float] = []
    rna_values: list[float] = []
    joint_values: list[float] = []
    for row in manifest.rows():
        sample = _move_complex(load_complex_row(adapter, row), device)
        ptok, pknown, pmask = _all_interface_corruption(sample.protein)
        rtok, rknown, rmask = _all_interface_corruption(sample.rna)

        outp = model(
            sample.protein.node_x,
            sample.protein.edge_index,
            sample.protein.edge_x,
            sample.rna.node_x,
            sample.rna.edge_index,
            sample.rna.edge_x,
            sample.pr,
            ptok,
            sample.rna.sequence,
            pknown,
            sample.rna.valid,
            use_delta=True,
            learned_alpha=True,
        )
        bp = balanced_sequence_loss(
            outp["protein_logits"],
            sample.protein.sequence,
            pmask,
            sample.protein.interface,
            None,
            None,
            None,
            None,
            "protein",
            0.0,
            0.0,
        )
        protein_values.append(float(bp.total))

        outr = model(
            sample.protein.node_x,
            sample.protein.edge_index,
            sample.protein.edge_x,
            sample.rna.node_x,
            sample.rna.edge_index,
            sample.rna.edge_x,
            sample.pr,
            sample.protein.sequence,
            rtok,
            sample.protein.valid,
            rknown,
            use_delta=True,
            learned_alpha=True,
        )
        br = balanced_sequence_loss(
            None,
            None,
            None,
            None,
            outr["rna_logits"],
            sample.rna.sequence,
            rmask,
            sample.rna.interface,
            "rna",
            0.0,
            0.0,
        )
        rna_values.append(float(br.total))

        if row.sample_id in selected:
            joint_values.append(sequential_joint_normalized_nll(model, sample))

    if not protein_values or not rna_values or not joint_values:
        raise ValueError("Joint validation produced an empty metric group")
    return float(
        (np.mean(protein_values) + np.mean(rna_values) + np.mean(joint_values)) / 3.0
    )


def train_stage(
    cfg: dict,
    stage: Stage,
    train_manifest: Path,
    val_manifest: Path,
    out_dir: Path,
    init_checkpoint: Path | None = None,
    device: str | None = None,
) -> Path:
    """Train one development stage and return the validation-selected checkpoint."""
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model_from_config(cfg).to(dev)
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])

    joint_mode = _joint_mode(cfg)
    configure_stage(model, stage, joint_unfreezing_mode=joint_mode)
    optcfg = cfg["optimization"]
    optimizer = build_optimizer(
        model,
        stage,
        float(optcfg["lr_heads"]),
        float(optcfg["lr_projections"]),
        float(optcfg["lr_encoder_top"]),
        float(optcfg["lr_encoder_bottom"]),
        float(optcfg.get("lr_global_c_joint", 0.0)),
        float(optcfg["weight_decay"]),
        float(optcfg["layerwise_lr_decay"]),
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    train_table = ManifestTable(train_manifest)
    val_table = ManifestTable(val_manifest)
    max_epochs = int(
        cfg["training_stages"][stage.value].get(
            "max_epochs", cfg["optimization"].get("max_epochs_default", 100)
        )
    )
    patience = int(optcfg["early_stopping_patience"])
    total_steps = max_epochs * max(1, len(train_table))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best.pt"
    best = float("inf")
    bad_epochs = 0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        progress = epoch / max(1, max_epochs - 1)
        if stage == Stage.JOINT:
            apply_joint_unfreezing(model, progress, mode=joint_mode)
        adapter = _adapter(cfg, epoch, training=True)
        rows = list(train_table.rows())
        random.Random(seed + epoch * 104729).shuffle(rows)
        running: list[float] = []
        mask_fractions: list[float] = []
        hard_count = 0
        task_counts: dict[str, int] = {}

        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, dev):
                loss, detail = _one_training_loss(
                    model, row, adapter, stage, cfg, epoch, progress, dev
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at {stage.value} epoch={epoch} sample={row.sample_id}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                float(optcfg["grad_clip_norm"]),
            )
            optimizer.step()
            global_step += 1
            _cosine_schedule(
                optimizer,
                global_step,
                total_steps,
                float(optcfg["warmup_fraction"]),
                base_lrs,
            )
            running.append(float(loss.detach().cpu()))
            if "mask_fraction" in detail:
                mask_fractions.append(float(detail["mask_fraction"]))
            if detail.get("hard_context"):
                hard_count += 1
            if "task" in detail:
                task = str(detail["task"])
                task_counts[task] = task_counts.get(task, 0) + 1

        val = validate_stage(model, stage, val_table, cfg, dev)
        record = {
            "stage": stage.value,
            "epoch": epoch,
            "schedule_horizon_epochs": max_epochs,
            "schedule_progress": progress,
            "joint_unfreezing_mode": joint_mode if stage == Stage.JOINT else None,
            "train_loss": float(np.mean(running)),
            "val_metric": val,
            "mean_mask_fraction": float(np.mean(mask_fractions)) if mask_fractions else None,
            "hard_context_samples": hard_count,
            "task_counts": task_counts,
            "trainable": trainable_parameter_report(model),
            "precision": (
                "bf16"
                if dev.type == "cuda"
                and str(optcfg.get("precision", "")).lower() == "bf16"
                and torch.cuda.is_bf16_supported()
                else "fp32"
            ),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        if val < best - 1e-6:
            best = val
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "stage": stage.value,
                    "epoch": epoch,
                    "selected_epoch_count": epoch + 1,
                    "schedule_horizon_epochs": max_epochs,
                    "schedule_progress_at_stop": progress,
                    "joint_unfreezing_mode": joint_mode if stage == Stage.JOINT else None,
                    "val_metric": val,
                    "config": cfg,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if not best_path.exists():
        raise RuntimeError("No checkpoint was saved")
    return best_path
