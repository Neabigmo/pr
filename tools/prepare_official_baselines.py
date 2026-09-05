#!/usr/bin/env python3
"""Prepare the exact frozen 900/100 pools for official ProteinMPNN and NA-MPNN.

This tool does NOT vendor or alter upstream model code. It converts only the
frozen manifests into the formats consumed by the pinned official repositories.
Primary baseline policy:
  - random initialization;
  - identical 900/100 sample IDs;
  - backbone noise 0.10 A;
  - about 150 dataset passes for both baselines;
  - published checkpoints are a separate reference track.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
import math
import string

import gemmi
import numpy as np
import pandas as pd
import torch

AA3 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}
RNA3 = {"A":"A","U":"U","G":"G","C":"C","RA":"A","RU":"U","RG":"G","RC":"C","ADE":"A","URA":"U","GUA":"G","CYT":"C"}
PROTEIN_ATOM_ORDER = ["N","CA","C","O","CB","CG","CG1","CG2","OG","OG1","SG","CD","CD1","CD2"]
RNA_BACKBONE = ["OP1","OP2","P","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'"]


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def parse_chain_list(value: object) -> list[str] | None:
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return None
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def _residue_atom_dict(res: gemmi.Residue) -> dict[str, tuple[np.ndarray,float,float]]:
    chosen={}
    for atom in res:
        name=atom.name.strip(); occ=float(atom.occ); xyz=np.array([atom.pos.x,atom.pos.y,atom.pos.z],dtype=np.float32); b=float(atom.b_iso)
        if not np.isfinite(xyz).all(): continue
        if name not in chosen or occ>chosen[name][1]: chosen[name]=(xyz,occ,b)
    return chosen


def extract_matching_chain(path: Path, expected_sequence: str, polymer: str, explicit_chains: list[str] | None = None):
    st=gemmi.read_structure(str(path)); model=st[0]; candidates=[]
    allowed=AA3 if polymer=="protein" else RNA3
    for chain in model:
        if explicit_chains is not None and str(chain.name) not in explicit_chains: continue
        seq=[]; residues=[]
        for res in chain:
            one=allowed.get(res.name.strip().upper())
            if one is None: continue
            seq.append(one); residues.append(res)
        if seq: candidates.append((str(chain.name),"".join(seq),residues))
    exact=[x for x in candidates if x[1]==expected_sequence]
    if len(exact)!=1:
        desc=[(c,s[:20],len(s)) for c,s,_ in candidates]
        raise ValueError(f"Expected exactly one {polymer} chain matching manifest sequence in {path}; candidates={desc}")
    return exact[0]


def synth_pdb_id(index: int) -> str:
    alphabet=string.digits+string.ascii_lowercase
    x=index; chars=[]
    for _ in range(4): chars.append(alphabet[x%36]); x//=36
    return "".join(reversed(chars))


def prepare_proteinmpnn(train_path: Path, val_path: Path, out: Path) -> dict:
    out.mkdir(parents=True,exist_ok=True); rows=[]; mapping=[]
    all_sets=[("train",read_manifest(train_path)),("val",read_manifest(val_path))]
    counter=0; val_clusters=[]
    for split,df in all_sets:
        for _,row in df.iterrows():
            seq=str(row["sequence"]).strip().upper(); chains=parse_chain_list(row.get("protein_chains",row.get("chains")))
            chain_name,got,residues=extract_matching_chain(Path(str(row["structure_path"])),seq,"protein",chains)
            pid=synth_pdb_id(counter+1); synthetic=f"{pid}_A"; cluster=(counter+1 if split=="train" else 100000+counter+1)
            prefix=out/"pdb"/pid[1:3]/pid; prefix.parent.mkdir(parents=True,exist_ok=True)
            L=len(residues); xyz=np.zeros((L,14,3),dtype=np.float32); mask=np.zeros((L,14),dtype=np.float32); bfac=np.zeros((L,14),dtype=np.float32); occ=np.zeros((L,14),dtype=np.float32)
            for i,res in enumerate(residues):
                atoms=_residue_atom_dict(res)
                for ai,name in enumerate(PROTEIN_ATOM_ORDER):
                    if name in atoms:
                        xyz[i,ai]=atoms[name][0]; mask[i,ai]=1.; occ[i,ai]=atoms[name][1]; bfac[i,ai]=atoms[name][2]
                if not bool(mask[i,:4].all()): raise ValueError(f"Missing N/CA/C/O in {row['sample_id']} residue {i}")
            torch.save({"seq":seq,"xyz":torch.from_numpy(xyz),"mask":torch.from_numpy(mask),"bfac":torch.from_numpy(bfac),"occ":torch.from_numpy(occ)},str(prefix)+"_A.pt")
            # Empty assemblies make official loader return this chain alone; tm retained for schema completeness.
            torch.save({"method":"pilot","date":"2000-01-01","resolution":1.0,"chains":["A"],"tm":np.array([[[1.0,1.0,0.0]]],dtype=np.float32).reshape(1,1,3),"asmb_ids":[],"asmb_details":[],"asmb_method":[],"asmb_chains":[]},str(prefix)+".pt")
            rows.append([synthetic,"2000-01-01","1.0",f"hash{counter}",str(cluster),seq])
            if split=="val": val_clusters.append(cluster)
            mapping.append({"sample_id":str(row["sample_id"]),"synthetic_chain":synthetic,"source_chain":chain_name,"split":split,"cluster":cluster})
            counter+=1
    with (out/"list.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["CHAINID","DEPOSITION","RESOLUTION","HASH","CLUSTER","SEQUENCE"]); w.writerows(rows)
    (out/"valid_clusters.txt").write_text("\n".join(map(str,val_clusters))+"\n",encoding="utf-8"); (out/"test_clusters.txt").write_text("",encoding="utf-8")
    pd.DataFrame(mapping).to_csv(out/"sample_mapping.tsv",sep="\t",index=False)
    return {"train":sum(x["split"]=="train" for x in mapping),"val":sum(x["split"]=="val" for x in mapping)}


def write_rna_only_pdb(path: Path, residues, sequence: str) -> None:
    lines=[]; serial=1
    for ri,(res,base) in enumerate(zip(residues,sequence),start=1):
        atoms=_residue_atom_dict(res)
        # Preserve all RNA atoms; upstream ATOMS_TO_LOAD=backbone controls what the model reads.
        for atom_name,(xyz,occ,bfac) in atoms.items():
            element="".join(x for x in atom_name if x.isalpha())[:1].upper() or "C"
            x,y,z=map(float,xyz)
            lines.append(f"ATOM  {serial:5d} {atom_name:>4s} {base:>3s} A{ri:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}")
            serial+=1
    lines += ["TER","END"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def _rna_preprocessed_paths(root: Path, stem: str, length: int) -> dict:
    pp=root/"preprocessed"; pp.mkdir(parents=True,exist_ok=True); key="1"; zeros=np.zeros(length,dtype=np.int32); zeros64=np.zeros(length,dtype=np.int64)
    values={
        "asmb_lengths_path":{key:(length,0,0,length)},
        "asmb_interface_masks_path":{key:zeros},
        "asmb_side_chain_interface_masks_path":{key:zeros},
        "asmb_nearest_protein_side_chain_index_path":{key:zeros64},
        "asmb_base_pair_masks_path":{key:zeros},
        "asmb_base_pair_index_path":{key:zeros64},
        "asmb_canonical_base_pair_masks_path":{key:zeros},
        "asmb_canonical_base_pair_index_path":{key:zeros64},
    }
    result={}
    for name,obj in values.items():
        p=pp/f"{stem}.{name}.npy"; np.save(p,obj); result[name]=str(p.resolve())
    return result


def _estimate_na_steps(lengths: list[int], batch_tokens: int, passes: int) -> int:
    # Mirrors upstream StructureLoader packing closely enough for budget matching.
    batch=[]; clusters=0
    for size in sorted(lengths):
        if size>batch_tokens: continue
        if size*(len(batch)+1)<=batch_tokens: batch.append(size)
        else:
            if batch: clusters+=1
            batch=[size]
    if batch: clusters+=1
    return max(1,clusters*passes)


def prepare_nampnn(train_path: Path, val_path: Path, out: Path, passes: int = 150, batch_tokens: int = 6000) -> dict:
    out.mkdir(parents=True,exist_ok=True); pdb_dir=out/"rna_pdb"; pdb_dir.mkdir(exist_ok=True); outputs={}
    train_lengths=[]
    for split,path in [("train",train_path),("valid",val_path)]:
        df=read_manifest(path); rows=[]
        for idx,row in df.iterrows():
            seq=str(row["sequence"]).strip().upper().replace("T","U"); chains=parse_chain_list(row.get("rna_chains",row.get("chains")))
            chain_name,got,residues=extract_matching_chain(Path(str(row["structure_path"])),seq,"rna",chains)
            stem=f"{split}_{idx:04d}_{row['sample_id']}"; pdb_path=(pdb_dir/f"{stem}.pdb").resolve(); write_rna_only_pdb(pdb_path,residues,seq)
            rec={"id":str(row["sample_id"]),"structure_path":str(pdb_path),"date":"2000-01-01","sampling_probability":1.0,"ppm_paths":"[]","source_chain":chain_name}
            rec.update(_rna_preprocessed_paths(out,stem,len(seq))); rows.append(rec)
            if split=="train": train_lengths.append(len(seq))
        out_csv=out/f"{split}.csv"; pd.DataFrame(rows).to_csv(out_csv,index=False); outputs[split]=str(out_csv.resolve())
    total_steps=_estimate_na_steps(train_lengths,batch_tokens,passes)
    config={
        "VOCAB_SIZE":33,"NUM_LETTERS":33,"PARSE_PROTEIN":0,"PARSE_DNA":0,"PARSE_RNA":1,"PARSE_RNA_AS_DNA":0,"NA_SHARED_TOKENS":1,"NA_REF_ATOM":"C1'","INCLUDE_PRED_NA_N":1,
        "PROTEIN_BACKBONE_OCC_CUTOFF":0.8,"PROTEIN_SIDE_CHAIN_OCC_CUTOFF":0.5,"DNA_BACKBONE_OCC_CUTOFF":0.8,"DNA_SIDE_CHAIN_OCC_CUTOFF":0.5,"RNA_BACKBONE_OCC_CUTOFF":0.8,"RNA_SIDE_CHAIN_OCC_CUTOFF":0.5,
        "EXCLUDED_ELEMENTS":[1],"DATE_CUTOFF":"2030-01-01","MAX_NUMBER_OF_PDBS_TRAIN":len(read_manifest(train_path)),"MAX_NUMBER_OF_PDBS_VALID":len(read_manifest(val_path)),"BATCH_TOKENS":batch_tokens,"LOSS_TOKENS":batch_tokens,
        "LABEL_SMOOTHING":0.1,"EXCLUDE_RES":["HOH","NA","CL","K","BR"],"MIN_PROTEIN_LENGTH_CUTOFF":1,"NUM_WORKERS":4,"TOTAL_STEPS":total_steps,"RANDOMIZE_NMR_MODEL":0,"CROP_LARGE_STRUCTURES":0,"MIN_OVERLAP_LENGTH":5,
        "DF_PATH_TRAIN":outputs["train"],"DF_PATH_VALID":outputs["valid"],"BASE_FOLDER":str((out/"model").resolve()),"PREV_CHECKPOINT":"","HIDDEN_DIM":128,"NUM_ENCODER_LAYERS":3,"NUM_DECODER_LAYERS":3,"NUM_NEIGHBORS":32,"DROPOUT":0.1,"DECODE_PROTEIN_FIRST":0,
        "PROTEIN_BACKBONE_NOISE":0.1,"DNA_BACKBONE_NOISE":0.1,"RNA_BACKBONE_NOISE":0.1,"PARSE_PPMS":0,"NA_ONLY_AS_UNIFORM_PPM":0,"DROP_PROTEIN_PROBABILITY":0,"PROTEIN_INTERFACE_RESIDUE_MUTATION_PROBABILITY":0,"MUTATE_BASE_PAIR_TOGETHER":0,"MUTATE_ENTIRE_SIDE_CHAIN_INTERFACE_PROBABILITY":0,"NA_NON_INTERFACE_AS_UNIFORM_PPM":0,
        "GRADIENT_NORM":1.0,"MIXED_PRECISION":1,"SAVE_EVERY_N_STEPS":max(1,total_steps//20),"ATOMS_TO_LOAD":"backbone","METRICS_TO_COMPUTE":"basic"
    }
    cfg_path=out/"na_mpnn_from_scratch.json"; cfg_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    return {"train":len(read_manifest(train_path)),"val":len(read_manifest(val_path)),"passes":passes,"estimated_total_steps":total_steps,"config":str(cfg_path)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--protein-train",type=Path,required=True); p.add_argument("--protein-val",type=Path,required=True); p.add_argument("--rna-train",type=Path,required=True); p.add_argument("--rna-val",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--passes",type=int,default=150); args=p.parse_args()
    result={"ProteinMPNN":prepare_proteinmpnn(args.protein_train,args.protein_val,args.out/"proteinmpnn"),"NA-MPNN":prepare_nampnn(args.rna_train,args.rna_val,args.out/"na_mpnn",args.passes)}
    (args.out/"baseline_preparation.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
