#!/usr/bin/env python3
"""Screen a complex manifest in disjoint worker shards with auditable progress."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def _new_lines(path: Path, offset: int) -> tuple[int, int]:
    if not path.exists():
        return 0, offset
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
        return data.count(b"\n"), handle.tell()


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, sep="\t")
    if manifest.empty:
        raise ValueError("The complex manifest is empty")
    if manifest["pdb_id"].astype(str).duplicated().any():
        raise ValueError("The complex manifest contains duplicate pdb_id values")
    args.shard_root.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    shards: list[Path] = []
    for index in range(args.shards):
        path = args.shard_root / f"complex_shard_{index:02d}.tsv"
        manifest.iloc[index::args.shards].to_csv(path, sep="\t", index=False)
        shards.append(path)

    def run_one(item: tuple[int, Path]) -> None:
        index, shard = item
        shard_out = args.out / f"shard_{index:02d}"
        shard_out.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python), "-m", "pr_pilot.cli", "screen",
            "--kind", "complex", "--config", str(args.config),
            "--download-manifest", str(shard), "--out", str(shard_out),
            "--progress-log", str(shard_out / "progress.jsonl"),
            "--progress-label", f"complex shard {index:02d}", "--no-progress",
        ]
        environment = {**os.environ, "PYTHONPATH": str(args.repo_root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        with (shard_out / "worker.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=args.repo_root, env=environment, check=True, stdout=log, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(max_workers=args.shards) as pool:
        futures = {pool.submit(run_one, item): item[0] for item in enumerate(shards)}
        logs = [args.out / f"shard_{i:02d}" / "progress.jsonl" for i in range(args.shards)]
        offsets = {path: 0 for path in logs}
        pending = set(futures)
        errors: list[tuple[int, Exception]] = []
        with tqdm(total=len(manifest), desc=f"complex screening ({args.shards} shards)", unit="sample", dynamic_ncols=True) as bar:
            while pending:
                completed = 0
                for path in logs:
                    count, offsets[path] = _new_lines(path, offsets[path])
                    completed += count
                if completed:
                    bar.update(completed)
                for future in [future for future in pending if future.done()]:
                    pending.remove(future)
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append((futures[future], exc))
                if pending:
                    time.sleep(1)
            for path in logs:
                count, offsets[path] = _new_lines(path, offsets[path])
                if count:
                    bar.update(count)
        if errors:
            shard, error = errors[0]
            raise RuntimeError(f"Complex screening shard {shard:02d} failed; inspect worker.log") from error
        if bar.n != len(manifest):
            raise RuntimeError(f"Progress coverage mismatch: completed={bar.n} expected={len(manifest)}")

    eligible = pd.concat([pd.read_csv(args.out / f"shard_{i:02d}" / "complex_eligible.tsv", sep="\t") for i in range(args.shards)], ignore_index=True)
    rejected = pd.concat([pd.read_csv(args.out / f"shard_{i:02d}" / "complex_rejected.tsv", sep="\t") for i in range(args.shards)], ignore_index=True)
    if eligible["pdb_id"].astype(str).duplicated().any() or rejected["pdb_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate IDs found while merging screening shards")
    if len(eligible) + len(rejected) != len(manifest):
        raise RuntimeError("Screening shard coverage is incomplete")
    eligible.to_csv(args.out / "complex_eligible.tsv", sep="\t", index=False)
    rejected.to_csv(args.out / "complex_rejected.tsv", sep="\t", index=False)
    summary = {
        "kind": "complex",
        "manifest": str(args.manifest),
        "config": str(args.config),
        "shards": args.shards,
        "downloaded": len(manifest),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "rejection_counts": rejected["reason"].value_counts().to_dict(),
    }
    (args.out / "complex_screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"MERGED complex: eligible={len(eligible)} rejected={len(rejected)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    run(args)


if __name__ == "__main__":
    main()
