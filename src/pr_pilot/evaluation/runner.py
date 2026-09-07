"""Executable held-out evaluation for the frozen 100 complexes.

This module evaluates one checkpoint. Cross-model/seed statistics live in
``compare_runs.py``. The PMI accumulated here is explicitly a *model-graph contact
proxy* because it is derived from capped PR graph edges; an independent full
heavy-atom empirical-contact analysis is run separately for biological validation.
"""
from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F

from pr_pilot.evaluation.battery import empirical_pmi, matrix_correlations, token_metrics
from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch, double_center
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.engine import build_model_from_config


def _move(sample: ComplexTensorSample, device: torch.device) -> ComplexTensorSample:
    for graph in [sample.protein, sample.rna]:
        for name in [
            "node_x",
            "edge_index",
            "edge_x",
            "sequence",
            "interface",
            "valid",
            "fixed",
            "reference_xyz",
            "chain_index",
        ]:
            setattr(graph, name, getattr(graph, name).to(device))
    sample.pr = PRBatch(
        sample.pr.protein_index.to(device),
        sample.pr.rna_index.to(device),
        sample.pr.edge_features.to(device),
        sample.pr.effective_distance.to(device),
        None if sample.pr.edge_batch is None else sample.pr.edge_batch.to(device),
    )
    return sample


def _adapter(
    cfg: dict,
    noise: float = 0.0,
    rich: bool | None = None,
    seed_offset: int = 0,
) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    return GemmiStructureAdapter(
        int(g["rbf_bins"]),
        int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]),
        int(g["pr_max_neighbors"]),
        noise,
        int(cfg["experiment"]["pilot_seed"]) + seed_offset,
        bool(g["rich_pr_geometry"] if rich is None else rich),
    )


def load_model(checkpoint: Path, cfg: dict, device: torch.device) -> JointPriorAndFieldModel:
    model = build_model_from_config(cfg).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _forward(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    pt: Tensor,
    rt: Tensor,
    pk: Tensor,
    rk: Tensor,
) -> dict[str, Tensor]:
    return model(
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
    )


def _rows_from_logits(
    sample_id: str,
    polymer: str,
    logits: Tensor,
    native: Tensor,
    mask: Tensor,
    interface: Tensor,
    model_name: str,
    seed: int,
) -> list[dict]:
    logp = F.log_softmax(logits.float(), -1)
    prob = logp.exp()
    pred = logits.argmax(-1)
    rows = []
    for idx in torch.where(mask)[0]:
        i = int(idx)
        n = int(native[i])
        p = int(pred[i])
        row = {
            "sample_id": sample_id,
            "polymer": polymer,
            "position": i,
            "native_token": n,
            "predicted_token": p,
            "native_log_probability": float(logp[i, n].cpu()),
            "max_probability": float(prob[i].max().cpu()),
            "is_interface": bool(interface[i]),
            "model": model_name,
            "seed": seed,
        }
        row.update({f"probability_{j}": float(value.cpu()) for j, value in enumerate(prob[i])})
        rows.append(row)
    return rows


@torch.no_grad()
def score_conditional(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    target: str,
    interface_only: bool = False,
    partner_hide: float = 0.0,
    seed: int = 0,
    model_name: str = "DMICF",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pt = sample.protein.sequence.clone()
    rt = sample.rna.sequence.clone()
    if target == "protein":
        mask = (sample.protein.interface if interface_only else sample.protein.valid) & sample.protein.valid & ~sample.protein.fixed
        pk = sample.protein.valid.clone().bool()
        pk[mask] = False
        rk = sample.rna.valid.clone().bool()
        if partner_hide > 0:
            candidates = torch.where(rk & ~sample.rna.fixed)[0].cpu().numpy()
            n = int(round(len(candidates) * partner_hide))
            if n:
                selected = torch.tensor(rng.choice(candidates, n, replace=False), device=rk.device)
                rk[selected] = False
        out = _forward(model, sample, pt, rt, pk, rk)
        return pd.DataFrame(
            _rows_from_logits(
                sample.sample_id,
                "protein",
                out["protein_logits"],
                sample.protein.sequence,
                mask,
                sample.protein.interface,
                model_name,
                seed,
            )
        )
    if target == "rna":
        mask = (sample.rna.interface if interface_only else sample.rna.valid) & sample.rna.valid & ~sample.rna.fixed
        rk = sample.rna.valid.clone().bool()
        rk[mask] = False
        pk = sample.protein.valid.clone().bool()
        if partner_hide > 0:
            candidates = torch.where(pk & ~sample.protein.fixed)[0].cpu().numpy()
            n = int(round(len(candidates) * partner_hide))
            if n:
                selected = torch.tensor(rng.choice(candidates, n, replace=False), device=pk.device)
                pk[selected] = False
        out = _forward(model, sample, pt, rt, pk, rk)
        return pd.DataFrame(
            _rows_from_logits(
                sample.sample_id,
                "rna",
                out["rna_logits"],
                sample.rna.sequence,
                mask,
                sample.rna.interface,
                model_name,
                seed,
            )
        )
    raise ValueError(target)


@torch.no_grad()
def score_joint_teacher_forced(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    orders: int = 5,
    seed: int = 0,
    model_name: str = "DMICF",
) -> pd.DataFrame:
    """Mixed-order native pseudo-NLL without exposing the current target token."""
    rows = []
    design = [
        ("protein", int(i))
        for i in torch.where(sample.protein.valid & ~sample.protein.fixed)[0]
    ] + [
        ("rna", int(i))
        for i in torch.where(sample.rna.valid & ~sample.rna.fixed)[0]
    ]
    for order_idx in range(orders):
        rng = random.Random(seed + 104729 * order_idx)
        order = design.copy()
        rng.shuffle(order)
        pk = sample.protein.fixed & sample.protein.valid
        rk = sample.rna.fixed & sample.rna.valid
        pt = sample.protein.sequence.clone()
        rt = sample.rna.sequence.clone()
        for polymer, i in order:
            out = _forward(model, sample, pt, rt, pk, rk)
            logits = out["protein_logits"][i] if polymer == "protein" else out["rna_logits"][i]
            logp = F.log_softmax(logits.float(), -1)
            prob = logp.exp()
            native = int(pt[i] if polymer == "protein" else rt[i])
            pred = int(logits.argmax())
            interface = bool(sample.protein.interface[i] if polymer == "protein" else sample.rna.interface[i])
            row = {
                "sample_id": sample.sample_id,
                "polymer": polymer,
                "position": i,
                "order": order_idx,
                "native_token": native,
                "predicted_token": pred,
                "native_log_probability": float(logp[native].cpu()),
                "max_probability": float(prob.max().cpu()),
                "is_interface": interface,
                "model": model_name,
                "seed": seed,
            }
            row.update({f"probability_{j}": float(value.cpu()) for j, value in enumerate(prob)})
            rows.append(row)
            if polymer == "protein":
                pk[i] = True
            else:
                rk[i] = True
    return pd.DataFrame(rows)


def _permute_tensor(x: Tensor, valid: Tensor, seed: int) -> Tensor:
    out = x.clone()
    idx = torch.where(valid)[0].cpu().numpy()
    rng = np.random.default_rng(seed)
    perm = idx.copy()
    rng.shuffle(perm)
    out[torch.tensor(idx, device=x.device)] = x[torch.tensor(perm, device=x.device)]
    return out


@torch.no_grad()
def partner_scramble(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    repeats: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for direction in ["protein", "rna"]:
        native = score_conditional(model, sample, direction, interface_only=True, seed=seed)
        native_nll = -native.native_log_probability.mean()
        for k in range(repeats):
            pt = sample.protein.sequence.clone()
            rt = sample.rna.sequence.clone()
            if direction == "protein":
                rt = _permute_tensor(rt, sample.rna.valid, seed + 1000 + k)
                mask = sample.protein.interface & sample.protein.valid & ~sample.protein.fixed
                pk = sample.protein.valid.clone()
                pk[mask] = False
                rk = sample.rna.valid.clone()
                out = _forward(model, sample, pt, rt, pk, rk)
                logp = F.log_softmax(out["protein_logits"].float(), -1)
                nll = float(-logp[mask, sample.protein.sequence[mask]].mean().cpu())
            else:
                pt = _permute_tensor(pt, sample.protein.valid, seed + 2000 + k)
                mask = sample.rna.interface & sample.rna.valid & ~sample.rna.fixed
                rk = sample.rna.valid.clone()
                rk[mask] = False
                pk = sample.protein.valid.clone()
                out = _forward(model, sample, pt, rt, pk, rk)
                logp = F.log_softmax(out["rna_logits"].float(), -1)
                nll = float(-logp[mask, sample.rna.sequence[mask]].mean().cpu())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "direction": direction,
                    "repeat": k,
                    "native_nll": float(native_nll),
                    "scrambled_nll": nll,
                    "delta_nll": nll - float(native_nll),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def counterfactual_partner_mutation(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    seed: int = 0,
    max_partner_sites: int = 16,
) -> pd.DataFrame:
    """Symmetric single-token counterfactuals; current target tokens remain hidden."""
    rng = np.random.default_rng(seed)
    rows = []

    rsites = torch.unique(sample.pr.rna_index).cpu().numpy()
    rng.shuffle(rsites)
    rsites = rsites[:max_partner_sites]
    base_pk = sample.protein.valid.clone()
    base_pk[sample.protein.interface & ~sample.protein.fixed] = False
    rk = sample.rna.valid.clone()
    base = _forward(model, sample, sample.protein.sequence, sample.rna.sequence, base_pk, rk)
    p0 = F.softmax(base["protein_logits"].float(), -1)
    for j in rsites:
        native = int(sample.rna.sequence[j])
        for alt in [x for x in range(4) if x != native]:
            rt = sample.rna.sequence.clone()
            rt[j] = alt
            out = _forward(model, sample, sample.protein.sequence, rt, base_pk, rk)
            p1 = F.softmax(out["protein_logits"].float(), -1)
            kl = (p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(-1)
            for i in torch.where(sample.protein.interface)[0]:
                edge = (sample.pr.protein_index == i) & (sample.pr.rna_index == int(j))
                dist = (
                    float(sample.pr.effective_distance[edge].min().cpu())
                    if edge.any()
                    else float(torch.linalg.vector_norm(sample.protein.reference_xyz[i] - sample.rna.reference_xyz[j]).cpu())
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "mutated_polymer": "rna",
                        "partner_position": int(j),
                        "native": native,
                        "alternative": alt,
                        "responding_polymer": "protein",
                        "position": int(i),
                        "distance": dist,
                        "kl": float(kl[i].cpu()),
                    }
                )

    psites = torch.unique(sample.pr.protein_index).cpu().numpy()
    rng.shuffle(psites)
    psites = psites[:max_partner_sites]
    pk = sample.protein.valid.clone()
    base_rk = sample.rna.valid.clone()
    base_rk[sample.rna.interface & ~sample.rna.fixed] = False
    base = _forward(model, sample, sample.protein.sequence, sample.rna.sequence, pk, base_rk)
    r0 = F.softmax(base["rna_logits"].float(), -1)
    for i in psites:
        native = int(sample.protein.sequence[i])
        alts = [x for x in range(20) if x != native]
        rng.shuffle(alts)
        for alt in alts[:4]:
            pt = sample.protein.sequence.clone()
            pt[i] = alt
            out = _forward(model, sample, pt, sample.rna.sequence, pk, base_rk)
            r1 = F.softmax(out["rna_logits"].float(), -1)
            kl = (r0 * (r0.clamp_min(1e-12).log() - r1.clamp_min(1e-12).log())).sum(-1)
            for j in torch.where(sample.rna.interface)[0]:
                edge = (sample.pr.protein_index == int(i)) & (sample.pr.rna_index == j)
                dist = (
                    float(sample.pr.effective_distance[edge].min().cpu())
                    if edge.any()
                    else float(torch.linalg.vector_norm(sample.protein.reference_xyz[i] - sample.rna.reference_xyz[j]).cpu())
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "mutated_polymer": "protein",
                        "partner_position": int(i),
                        "native": native,
                        "alternative": alt,
                        "responding_polymer": "rna",
                        "position": int(j),
                        "distance": dist,
                        "kl": float(kl[j].cpu()),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def field_tables(model: JointPriorAndFieldModel, sample: ComplexTensorSample) -> tuple[pd.DataFrame, np.ndarray]:
    hp, hr = model.encode_backbones(
        sample.protein.node_x,
        sample.protein.edge_index,
        sample.protein.edge_x,
        sample.rna.node_x,
        sample.rna.edge_index,
        sample.rna.edge_x,
    )
    field = model.dmicf.field(hp, hr, sample.pr)
    cnorm = float(torch.linalg.vector_norm(field["C"]).cpu())
    rows = []
    graph_proxy_counts = np.zeros((20, 4), dtype=np.int64)
    for edge_index in range(len(sample.pr.protein_index)):
        i = int(sample.pr.protein_index[edge_index])
        j = int(sample.pr.rna_index[edge_index])
        aa = int(sample.protein.sequence[i])
        base = int(sample.rna.sequence[j])
        distance = float(sample.pr.effective_distance[edge_index].cpu())
        if distance <= 5.0:
            graph_proxy_counts[aa, base] += 1
        delta_norm = float(torch.linalg.vector_norm(field["DeltaC"][edge_index]).cpu())
        rows.append(
            {
                "sample_id": sample.sample_id,
                "edge": edge_index,
                "protein_position": i,
                "rna_position": j,
                "distance": distance,
                "aa": aa,
                "base": base,
                "delta_fro": delta_norm,
                "delta_over_c": delta_norm / (cnorm + 1e-12),
                "alpha_p": float(field["alpha_p"][edge_index].cpu()),
                "alpha_r": float(field["alpha_r"][edge_index].cpu()),
                "score": float(field["scores"][edge_index].cpu()),
            }
        )
    return pd.DataFrame(rows), graph_proxy_counts


@torch.no_grad()
def evaluate_holdout(
    cfg: dict,
    checkpoint: Path,
    manifest_path: Path,
    out_dir: Path,
    device: str | None = None,
    model_name: str = "DMICF",
) -> dict:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, cfg, device_obj)
    adapter = _adapter(cfg)
    table = ManifestTable(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_p = []
    cond_r = []
    joint = []
    scramble = []
    counterfactual = []
    fields = []
    graph_counts = np.zeros((20, 4), dtype=np.int64)
    seed = int(cfg["experiment"]["pilot_seed"])

    for row in table.rows():
        sample = _move(load_complex_row(adapter, row), device_obj)
        cond_p.append(score_conditional(model, sample, "protein", False, seed=seed, model_name=model_name))
        cond_r.append(score_conditional(model, sample, "rna", False, seed=seed, model_name=model_name))
        joint.append(
            score_joint_teacher_forced(
                model,
                sample,
                int(cfg["evaluation"].get("joint_teacher_forced_orders", 5)),
                seed,
                model_name,
            )
        )
        scramble.append(partner_scramble(model, sample, int(cfg["evaluation"].get("scramble_repeats", 20)), seed))
        counterfactual.append(
            counterfactual_partner_mutation(
                model,
                sample,
                seed,
                int(cfg["evaluation"].get("counterfactual_max_partner_sites", 16)),
            )
        )
        field_df, counts = field_tables(model, sample)
        fields.append(field_df)
        graph_counts += counts

    outputs = {
        "conditional_protein": pd.concat(cond_p, ignore_index=True),
        "conditional_rna": pd.concat(cond_r, ignore_index=True),
        "joint_teacher_forced": pd.concat(joint, ignore_index=True),
        "partner_scramble": pd.concat(scramble, ignore_index=True),
        "counterfactual": pd.concat(counterfactual, ignore_index=True),
        "field_edges": pd.concat(fields, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)

    c_full = model.dmicf.global_c().detach().cpu()
    c_interaction = double_center(c_full)
    graph_pmi = empirical_pmi(graph_counts)
    graph_pmi_interaction = double_center(torch.from_numpy(graph_pmi)).numpy()
    np.save(out_dir / "C_global_centered.npy", c_full.numpy())
    np.save(out_dir / "C_interaction_only.npy", c_interaction.numpy())
    np.save(out_dir / "graph_contact_pmi.npy", graph_pmi)
    np.save(out_dir / "graph_contact_pmi_interaction_only.npy", graph_pmi_interaction)
    np.save(out_dir / "graph_contact_counts_20x4.npy", graph_counts)

    summary = {
        "protein": token_metrics(outputs["conditional_protein"], 20),
        "rna": token_metrics(outputs["conditional_rna"], 4),
        "joint_protein": token_metrics(outputs["joint_teacher_forced"].query("polymer=='protein'"), 20),
        "joint_rna": token_metrics(outputs["joint_teacher_forced"].query("polymer=='rna'"), 4),
        "C_vs_graph_PMI_full": matrix_correlations(c_full.numpy(), graph_pmi),
        "C_vs_graph_PMI_interaction_only": matrix_correlations(c_interaction.numpy(), graph_pmi_interaction),
        "pmi_warning": "Graph-contact PMI is a capped-model-graph proxy, not the independent heavy-atom biological validation.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
