"""Command-line interface for the complete mini-pilot.

The CLI intentionally exposes the whole scientific lifecycle:
RCSB discovery -> coordinate download -> structural screening -> joint sequence
clustering/Rfam annotation -> leakage-safe freezing -> audit -> six-stage
training -> joint sampling -> held-out evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import json

import pandas as pd
import torch
import yaml

from pr_pilot.data.clustering import annotate_all_candidates
from pr_pilot.data.manifest import (
    FrozenCounts,
    assert_no_test_leakage,
    assert_pretraining_disjoint,
    freeze_complex_pool,
    freeze_single_molecule_pool,
)
from pr_pilot.data.rcsb_io import discover_rcsb, download_rcsb_candidates, download_rfam_resources
from pr_pilot.data.screening import ScreenConfig, screen_download_manifest
from pr_pilot.evaluation.battery import mandatory_test_registry
from pr_pilot.evaluation.runner import evaluate_holdout, load_model, _move
from pr_pilot.inference.sampler import sample_joint
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.engine import train_stage
from pr_pilot.training.stages import Stage

PAA = "ACDEFGHIKLMNPQRSTVWY"
RNA = "AUGC"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _counts(cfg: dict) -> FrozenCounts:
    s = cfg["sampling"]
    return FrozenCounts(
        int(s["protein_pool_size"]),
        int(s["protein_train"]),
        int(s["protein_val"]),
        int(s["rna_pool_size"]),
        int(s["rna_train"]),
        int(s["rna_val"]),
        int(s["complex_pool_size"]),
        int(s["complex_dev"]),
        int(s["complex_test"]),
        int(s["complex_train"]),
        int(s["complex_val"]),
    )


def _screen_config(cfg: dict) -> ScreenConfig:
    raw = cfg.get("structure_filters", {})
    valid = {f.name for f in fields(ScreenConfig)}
    mapping = {
        "protein_min_length": raw.get("protein_min_length", 30),
        "protein_max_length": raw.get("protein_max_length", 1000),
        "rna_min_length": raw.get("rna_min_length", 5),
        "rna_max_length": raw.get("rna_max_length", 500),
        "max_total_tokens": raw.get("max_total_tokens", 1000),
        "max_resolution_angstrom": raw.get("max_resolution_angstrom", 4.0),
        "allow_nmr_without_resolution": raw.get("allow_nmr_without_resolution", True),
        "interface_contact_angstrom": raw.get("interface_contact_angstrom", 6.0),
        "min_interfacial_residue_pairs": raw.get("min_interfacial_residue_pairs", 3),
        "max_interface_missing_fraction": raw.get("max_interface_missing_fraction", 0.10),
        "exclude_large_rnp_keywords": bool(raw.get("exclude_ribosome", True) or raw.get("exclude_spliceosome", True)),
    }
    return ScreenConfig(**{k: v for k, v in mapping.items() if k in valid})


def cmd_discover(args: argparse.Namespace) -> None:
    frame = discover_rcsb(args.kind, args.out)
    print(f"RCSB {args.kind}: discovered {len(frame)} candidate entries -> {args.out}")


def cmd_download(args: argparse.Namespace) -> None:
    frame = download_rcsb_candidates(
        args.candidates,
        args.out,
        seed=args.seed,
        max_candidates=args.max_candidates,
        biological_assembly=(args.kind == "complex"),
    )
    print(f"{args.kind}: downloaded {len(frame)} structures -> {args.out}")


def cmd_download_rfam(args: argparse.Namespace) -> None:
    outputs = download_rfam_resources(args.out)
    print(json.dumps(outputs, indent=2))


def cmd_screen(args: argparse.Namespace) -> None:
    cfg = _screen_config(load_config(args.config))
    eligible, rejected = screen_download_manifest(args.download_manifest, args.kind, args.out, cfg)
    print(f"{args.kind}: eligible={len(eligible)} rejected={len(rejected)} -> {args.out}")


def cmd_annotate(args: argparse.Namespace) -> None:
    p, r, c = annotate_all_candidates(
        args.proteins,
        args.rnas,
        args.complexes,
        args.out,
        rfam_cm_gz=args.rfam_cm_gz,
        rfam_clanin=args.rfam_clanin,
        cmscan_cpu=args.cpu,
    )
    print(f"Annotated candidates:\n  protein={p}\n  RNA={r}\n  complexes={c}")


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze complex test first, then purge it from both prior pools."""
    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["pilot_seed"])
    counts = _counts(cfg)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    paths = freeze_complex_pool(
        args.complexes,
        out,
        seed + 202,
        counts,
        require_strict_bilateral=bool(cfg["experiment"]["strict_mode"]),
        strict_validation=bool(cfg["leakage"].get("strict_validation", True)),
    )
    freeze_single_molecule_pool(args.proteins, out, "protein", seed, paths["test"], counts)
    freeze_single_molecule_pool(args.rnas, out, "rna", seed + 101, paths["test"], counts)
    print(f"Frozen manifests written to {out}; final test was frozen before prior pools.")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def cmd_audit_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    root = args.manifest_root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = _counts(cfg)
    expected = {
        "protein_pool.tsv": counts.protein_pool,
        "protein_train.tsv": counts.protein_train,
        "protein_val.tsv": counts.protein_val,
        "rna_pool.tsv": counts.rna_pool,
        "rna_train.tsv": counts.rna_train,
        "rna_val.tsv": counts.rna_val,
        "complex_pool.tsv": counts.complex_pool,
        "complex_dev.tsv": counts.complex_dev,
        "complex_train.tsv": counts.complex_train,
        "complex_val.tsv": counts.complex_val,
        "complex_test.tsv": counts.complex_test,
    }
    report = {"counts": {}, "errors": [], "warnings": []}
    for name, n in expected.items():
        p = root / name
        if not p.exists():
            report["errors"].append(f"missing {name}")
            continue
        got = len(_read(p))
        report["counts"][name] = got
        if got != n:
            report["errors"].append(f"{name}: expected {n}, got {got}")

    if not report["errors"]:
        train = _read(root / "complex_train.tsv")
        val = _read(root / "complex_val.tsv")
        test = _read(root / "complex_test.tsv")
        try:
            assert_no_test_leakage(train, val, test, strict_cluster_check=bool(cfg["experiment"]["strict_mode"]))
            assert_pretraining_disjoint(_read(root / "protein_pool.tsv"), _read(root / "rna_pool.tsv"), test)
        except Exception as exc:
            report["errors"].append(str(exc))
        if not test["experimental"].astype(bool).all():
            report["errors"].append("Final test contains non-experimental structures")

    (out / "manifest_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    mandatory_test_registry().to_csv(out / "mandatory_test_registry.tsv", sep="\t", index=False)
    if report["errors"]:
        raise SystemExit("Data audit FAILED; inspect manifest_audit.json")
    print("Data audit passed, including final-test purge from both structural-prior pools.")


def _default_manifests(root: Path, stage: Stage) -> tuple[Path, Path]:
    if stage == Stage.PROTEIN_PRIOR:
        return root / "protein_train.tsv", root / "protein_val.tsv"
    if stage == Stage.RNA_PRIOR:
        return root / "rna_train.tsv", root / "rna_val.tsv"
    return root / "complex_train.tsv", root / "complex_val.tsv"


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    stage = Stage(args.stage)
    train, val = (
        (args.manifest, args.validation)
        if args.manifest and args.validation
        else _default_manifests(args.manifest_root, stage)
    )
    best = train_stage(cfg, stage, train, val, args.out, args.init_checkpoint, args.device)
    print(best)


def cmd_train_all(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    root = args.manifest_root
    ckpt = None
    order = [Stage.PROTEIN_PRIOR, Stage.RNA_PRIOR, Stage.GLOBAL_C, Stage.DELTA_C, Stage.ALPHA, Stage.JOINT]
    for stage in order:
        if not bool(cfg["training_stages"][stage.value].get("enabled", True)):
            continue
        train, val = _default_manifests(root, stage)
        stage_out = args.out / stage.value
        ckpt = train_stage(cfg, stage, train, val, stage_out, ckpt, args.device)
        print(f"{stage.value}: {ckpt}")
    if ckpt is None:
        raise SystemExit("No stage enabled")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "FINAL_CHECKPOINT.txt").write_text(str(ckpt), encoding="utf-8")


def _adapter_for_eval(cfg: dict) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    return GemmiStructureAdapter(
        int(g["rbf_bins"]),
        int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]),
        int(g["pr_max_neighbors"]),
        0.0,
        int(cfg["experiment"]["pilot_seed"]),
        bool(g["rich_pr_geometry"]),
    )


def cmd_sample(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, cfg, device)
    adapter = _adapter_for_eval(cfg)
    table = ManifestTable(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    icfg = cfg["inference"]
    spir = icfg["spir"]
    for row in table.rows():
        s = _move(load_complex_row(adapter, row), device)
        cands = sample_joint(
            model,
            s,
            int(icfg["candidates_per_complex"]),
            float(icfg["initial_temperature"]),
            int(cfg["experiment"]["pilot_seed"]),
            bool(spir["enabled"]),
            float(spir["reopen_fraction"]),
            float(spir["temperature"]),
            int(spir["cycles"]),
            float(spir["reverse_direction_fraction"]),
        )
        for c in cands:
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "candidate_id": c.candidate_id,
                    "protein_sequence": "".join(PAA[int(x)] for x in c.protein_tokens.cpu()),
                    "rna_sequence": "".join(RNA[int(x)] for x in c.rna_tokens.cpu()),
                    "pre_spir_protein": "".join(PAA[int(x)] for x in c.pre_spir_protein.cpu()),
                    "pre_spir_rna": "".join(RNA[int(x)] for x in c.pre_spir_rna.cpu()),
                    "spir_direction": c.spir_direction,
                    "spir_cycles": c.spir_cycles,
                    "mean_generation_logprob": sum(c.token_logprobs) / max(1, len(c.token_logprobs)),
                }
            )
    pd.DataFrame(rows).to_csv(args.out / "candidates.tsv", sep="\t", index=False)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    summary = evaluate_holdout(cfg, args.checkpoint, args.manifest, args.out, args.device, args.model_name)
    print(json.dumps(summary, indent=2))


def cmd_registry(args: argparse.Namespace) -> None:
    print(mandatory_test_registry().to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pr-pilot")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover-rcsb")
    d.add_argument("--kind", choices=["protein", "rna", "complex"], required=True)
    d.add_argument("--out", type=Path, required=True)
    d.set_defaults(func=cmd_discover)

    dl = sub.add_parser("download-rcsb")
    dl.add_argument("--kind", choices=["protein", "rna", "complex"], required=True)
    dl.add_argument("--candidates", type=Path, required=True)
    dl.add_argument("--out", type=Path, required=True)
    dl.add_argument("--seed", type=int, default=20260905)
    dl.add_argument("--max-candidates", type=int, required=True)
    dl.set_defaults(func=cmd_download)

    rf = sub.add_parser("download-rfam")
    rf.add_argument("--out", type=Path, required=True)
    rf.set_defaults(func=cmd_download_rfam)

    sc = sub.add_parser("screen")
    sc.add_argument("--kind", choices=["protein", "rna", "complex"], required=True)
    sc.add_argument("--config", type=Path, required=True)
    sc.add_argument("--download-manifest", type=Path, required=True)
    sc.add_argument("--out", type=Path, required=True)
    sc.set_defaults(func=cmd_screen)

    an = sub.add_parser("annotate")
    an.add_argument("--proteins", type=Path, required=True)
    an.add_argument("--rnas", type=Path, required=True)
    an.add_argument("--complexes", type=Path, required=True)
    an.add_argument("--rfam-cm-gz", type=Path, required=True)
    an.add_argument("--rfam-clanin", type=Path, required=True)
    an.add_argument("--cpu", type=int, default=4)
    an.add_argument("--out", type=Path, required=True)
    an.set_defaults(func=cmd_annotate)

    f = sub.add_parser("freeze")
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--proteins", type=Path, required=True)
    f.add_argument("--rnas", type=Path, required=True)
    f.add_argument("--complexes", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(func=cmd_freeze)

    a = sub.add_parser("audit-data")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--manifest-root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_audit_data)

    t = sub.add_parser("train")
    t.add_argument("--stage", required=True, choices=[x.value for x in Stage])
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--manifest-root", type=Path, default=Path("manifests"))
    t.add_argument("--manifest", type=Path)
    t.add_argument("--validation", type=Path)
    t.add_argument("--init-checkpoint", type=Path)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device")
    t.set_defaults(func=cmd_train)

    ta = sub.add_parser("train-all")
    ta.add_argument("--config", type=Path, required=True)
    ta.add_argument("--manifest-root", type=Path, required=True)
    ta.add_argument("--out", type=Path, required=True)
    ta.add_argument("--device")
    ta.set_defaults(func=cmd_train_all)

    s = sub.add_parser("sample-joint")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device")
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("evaluate")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--device")
    e.add_argument("--model-name", default="DMICF")
    e.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("test-registry")
    r.set_defaults(func=cmd_registry)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
