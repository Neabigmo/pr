"""Development-only gradient-conflict audit for final joint coordination.

The audit is diagnostic and never uses the final 100 complexes. It measures
pairwise cosine similarity among Protein-conditional, RNA-conditional and joint
loss gradients on parameters genuinely shared across the two polymers:
backbone encoders and DM-ICF. Task-specific sequence heads/decoders are excluded
from the cosine vector because zero blocks would dilute the interpretation.

If more than the configured fraction of task pairs are negative in repeated
audits, a PCGrad variant is recommended as an explicitly reported optimization
ablation. The primary run does not silently switch optimizers based on test data.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import yaml

from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.corruption import generate_corruption
from pr_pilot.training.engine import _adapter, _move_complex, build_model_from_config
from pr_pilot.training.losses import balanced_sequence_loss
from pr_pilot.training.stages import Stage, apply_joint_unfreezing, configure_stage


TASKS = ("protein_conditional", "rna_conditional", "joint")
PAIRS = (
    ("protein_conditional", "rna_conditional"),
    ("protein_conditional", "joint"),
    ("rna_conditional", "joint"),
)


def _shared_parameters(model):
    prefixes = ("protein_encoder.", "rna_encoder.", "dmicf.")
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith(prefixes)]


def _loss(model, sample, task: str, sample_id: str, seed: int):
    pvalid = sample.protein.valid & ~sample.protein.fixed
    rvalid = sample.rna.valid & ~sample.rna.fixed
    if task == "protein_conditional":
        pt = sample.protein.sequence.clone()
        pk = sample.protein.valid.clone().bool()
        pk[pvalid] = False
        rt = sample.rna.sequence.clone()
        rk = sample.rna.valid.clone().bool()
        out = model(
            sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x,
            sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr,
            pt, rt, pk, rk,
        )
        return balanced_sequence_loss(
            out["protein_logits"], sample.protein.sequence, pvalid, sample.protein.interface,
            None, None, None, None, "protein", 0.0, 0.0,
        ).total
    if task == "rna_conditional":
        pt = sample.protein.sequence.clone()
        pk = sample.protein.valid.clone().bool()
        rt = sample.rna.sequence.clone()
        rk = sample.rna.valid.clone().bool()
        rk[rvalid] = False
        out = model(
            sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x,
            sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr,
            pt, rt, pk, rk,
        )
        return balanced_sequence_loss(
            None, None, None, None,
            out["rna_logits"], sample.rna.sequence, rvalid, sample.rna.interface,
            "rna", 0.0, 0.0,
        ).total
    if task == "joint":
        pc = generate_corruption(
            sample.protein, 20, sample_id, 0, seed,
            force_fraction=0.50, wrong_token_fraction=0.0,
            use_curriculum=False, full_mask_probability=0.0,
        )
        rc = generate_corruption(
            sample.rna, 4, sample_id, 0, seed + 17,
            force_fraction=0.50, wrong_token_fraction=0.0,
            use_curriculum=False, full_mask_probability=0.0,
        )
        out = model(
            sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x,
            sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr,
            pc.input_tokens, rc.input_tokens, pc.known, rc.known,
        )
        return balanced_sequence_loss(
            out["protein_logits"], sample.protein.sequence, pc.target_mask, sample.protein.interface,
            out["rna_logits"], sample.rna.sequence, rc.target_mask, sample.rna.interface,
            "joint", 0.0, 0.0,
        ).total
    raise ValueError(task)


def _gradient_vector(loss: torch.Tensor, params) -> torch.Tensor:
    tensors = [p for _, p in params]
    grads = torch.autograd.grad(loss, tensors, retain_graph=False, allow_unused=True)
    flat = []
    for parameter, grad in zip(tensors, grads):
        flat.append(torch.zeros_like(parameter).reshape(-1) if grad is None else grad.detach().reshape(-1))
    return torch.cat(flat).float()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) <= 1e-20:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def audit_gradients(
    config: dict,
    checkpoint: Path,
    manifest_path: Path,
    out_dir: Path,
    *,
    device: str | None = None,
    max_complexes: int = 32,
    negative_fraction_trigger: float = 0.30,
) -> dict:
    """Run one frozen audit on complex-development validation data."""
    if "test" in manifest_path.name.lower():
        raise ValueError("Gradient audit must never run on the final test manifest")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(config).to(dev)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    configure_stage(model, Stage.JOINT)
    apply_joint_unfreezing(model, 1.0)
    model.train(False)
    params = _shared_parameters(model)
    if not params:
        raise RuntimeError("No shared parameters selected for gradient audit")

    adapter = _adapter(config, 0, training=False)
    table = ManifestTable(manifest_path)
    rows = []
    seed = int(config["experiment"]["pilot_seed"])
    for index, row in enumerate(table.rows()):
        if index >= max_complexes:
            break
        sample = _move_complex(load_complex_row(adapter, row), dev)
        vectors = {}
        losses = {}
        for task in TASKS:
            model.zero_grad(set_to_none=True)
            loss = _loss(model, sample, task, row.sample_id, seed + index * 101)
            vectors[task] = _gradient_vector(loss, params)
            losses[task] = float(loss.detach().cpu())
        for a, b in PAIRS:
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "task_a": a,
                    "task_b": b,
                    "cosine": _cosine(vectors[a], vectors[b]),
                    "loss_a": losses[a],
                    "loss_b": losses[b],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No gradient-audit rows were produced")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "gradient_cosines.tsv", sep="\t", index=False)
    finite = frame[np.isfinite(frame.cosine)]
    negative_fraction = float((finite.cosine < 0).mean()) if len(finite) else float("nan")
    by_pair = (
        finite.groupby(["task_a", "task_b"])["cosine"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    by_pair.to_csv(out_dir / "gradient_cosine_by_pair.tsv", sep="\t", index=False)
    summary = {
        "manifest": str(manifest_path),
        "n_complexes": int(frame.sample_id.nunique()),
        "n_task_pairs": int(len(finite)),
        "negative_fraction": negative_fraction,
        "trigger_threshold": float(negative_fraction_trigger),
        "pcgrad_ablation_recommended": bool(np.isfinite(negative_fraction) and negative_fraction > negative_fraction_trigger),
        "statistical_use": "development diagnostic only; never select from final-100 test",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--max-complexes", type=int, default=32)
    parser.add_argument("--negative-fraction-trigger", type=float, default=0.30)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            audit_gradients(
                cfg,
                args.checkpoint,
                args.manifest,
                args.out,
                device=args.device,
                max_complexes=args.max_complexes,
                negative_fraction_trigger=args.negative_fraction_trigger,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
