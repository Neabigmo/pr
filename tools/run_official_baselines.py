#!/usr/bin/env python3
"""Clone/check pinned upstream baselines, prepare data, and run from-scratch training.

This runner never downloads published weights for the primary baseline.
It records exact commands and refuses a checkout whose HEAD differs from LOCK.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import shlex
import subprocess
import sys


def run(cmd:list[str],cwd:Path|None=None,log:Path|None=None)->None:
    if log is not None:
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w",encoding="utf-8") as f:
            f.write("COMMAND: "+" ".join(shlex.quote(x) for x in cmd)+"\n\n")
            p=subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,text=True)
    else:
        p=subprocess.run(cmd,cwd=cwd)
    if p.returncode!=0: raise RuntimeError(f"Command failed ({p.returncode}): {cmd}; log={log}")


def ensure_checkout(root:Path,spec:dict)->Path:
    checkout=(root/spec["checkout"]).resolve(); checkout.parent.mkdir(parents=True,exist_ok=True)
    if not checkout.exists(): run(["git","clone",spec["repository"],str(checkout)])
    got=subprocess.check_output(["git","rev-parse","HEAD"],cwd=checkout,text=True).strip()
    if got!=spec["commit"]:
        # Detached checkout is explicit and reproducible; working tree must be clean.
        dirty=subprocess.check_output(["git","status","--porcelain"],cwd=checkout,text=True).strip()
        if dirty: raise RuntimeError(f"Refusing to move dirty checkout {checkout}")
        run(["git","fetch","origin",spec["commit"]],cwd=checkout)
        run(["git","checkout","--detach",spec["commit"]],cwd=checkout)
    got=subprocess.check_output(["git","rev-parse","HEAD"],cwd=checkout,text=True).strip()
    if got!=spec["commit"]: raise RuntimeError(f"Commit mismatch {got} != {spec['commit']}")
    return checkout


def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",type=Path,default=Path(".")); p.add_argument("--manifest-root",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--python",default=sys.executable); p.add_argument("--prepare-only",action="store_true"); args=p.parse_args()
    root=args.repo_root.resolve(); lock=json.loads((root/"third_party/LOCK.json").read_text()); args.out.mkdir(parents=True,exist_ok=True)
    protein=ensure_checkout(root,lock["proteinmpnn"]); na=ensure_checkout(root,lock["rna_fixbb"])
    prep=args.out/"prepared"
    run([args.python,str(root/"tools/prepare_official_baselines.py"),"--protein-train",str(args.manifest_root/"protein_train.tsv"),"--protein-val",str(args.manifest_root/"protein_val.tsv"),"--rna-train",str(args.manifest_root/"rna_train.tsv"),"--rna-val",str(args.manifest_root/"rna_val.tsv"),"--out",str(prep),"--passes","150"],cwd=root,log=args.out/"prepare.log")
    if args.prepare_only: return
    p_cmd=[args.python,"training/training.py","--path_for_training_data",str((prep/"proteinmpnn").resolve()),"--path_for_outputs",str((args.out/"proteinmpnn_train").resolve()),"--previous_checkpoint","","--num_epochs","150","--batch_size","10000","--max_protein_length","1000","--hidden_dim","128","--num_encoder_layers","3","--num_neighbors","48","--dropout","0.1","--backbone_noise","0.10","--rescut","3.5","--gradient_norm","1.0","--mixed_precision","True"]
    run(p_cmd,cwd=protein,log=args.out/"proteinmpnn_train.log")
    na_cfg=(prep/"na_mpnn/na_mpnn_from_scratch.json").resolve()
    n_cmd=[args.python,"na_run.py",str(na_cfg)]
    run(n_cmd,cwd=na,log=args.out/"na_mpnn_train.log")
    summary={"proteinmpnn_commit":lock["proteinmpnn"]["commit"],"na_mpnn_commit":lock["rna_fixbb"]["commit"],"proteinmpnn_command":p_cmd,"na_mpnn_command":n_cmd,"primary_from_scratch":True}
    (args.out/"baseline_run_manifest.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__=="__main__": main()
