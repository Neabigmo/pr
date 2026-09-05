from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from pr_pilot.data import screening
from pr_pilot.data import clustering


def test_resolution_method_filter_can_be_disabled_for_a_round(monkeypatch, tmp_path):
    residue = SimpleNamespace(atoms={"N", "CA", "C", "O"})
    chain = SimpleNamespace(
        chain="A",
        polymer="protein",
        residues=[residue],
        sequence="A",
        modified_fraction=0.0,
    )
    monkeypatch.setattr(screening, "_metadata_header", lambda path: ("title", "X-RAY DIFFRACTION", 8.0, object()))
    monkeypatch.setattr(screening.gemmi, "make_structure_from_block", lambda block: object())
    monkeypatch.setattr(
        screening,
        "_chains",
        lambda structure: ([chain], [], False, False),
    )
    path = tmp_path / "1ABC.cif"

    rejected, reason = screening.screen_file(
        path,
        "protein",
        screening.ScreenConfig(
            protein_min_length=1,
            protein_max_length=10,
            apply_resolution_method_filter=True,
        ),
    )
    assert rejected is None
    assert reason == "resolution_or_method"

    accepted, reason = screening.screen_file(
        path,
        "protein",
        screening.ScreenConfig(
            protein_min_length=1,
            protein_max_length=10,
            apply_resolution_method_filter=False,
        ),
    )
    assert accepted is not None
    assert reason == ""
    assert accepted["resolution"] == 8.0


def test_spatial_index_matches_bruteforce_interface_pairs():
    def residue(chain: str, index: int, points: list[list[float]]) -> screening.ResidueRecord:
        return screening.ResidueRecord(
            chain=chain,
            index=index,
            name="A",
            token="A",
            modified=False,
            atoms=set(),
            heavy_xyz=np.asarray(points, dtype=np.float32),
        )

    proteins = [screening.ChainRecord("P", "protein", [residue("P", 0, [[0, 0, 0]]), residue("P", 1, [[20, 0, 0]])])]
    rnas = [screening.ChainRecord("R", "rna", [residue("R", 0, [[6, 0, 0]]), residue("R", 1, [[20.1, 0, 0]])])]

    expected = []
    for protein_chain in proteins:
        for rna_chain in rnas:
            for protein_residue in protein_chain.residues:
                for rna_residue in rna_chain.residues:
                    distance = screening._min_distance(protein_residue.heavy_xyz, rna_residue.heavy_xyz)
                    if distance <= 6.0:
                        expected.append((protein_residue.chain, protein_residue.index, rna_residue.chain, rna_residue.index, distance))

    actual = screening._interface_pairs(proteins, rnas, 6.0)
    actual_keys = [(p.chain, p.index, r.chain, r.index, distance) for p, r, distance in actual]
    assert actual_keys == expected


def test_screen_manifest_logs_each_completed_record(tmp_path, monkeypatch):
    manifest = tmp_path / "download_manifest.tsv"
    pd.DataFrame(
        [
            {"pdb_id": "A", "path": str(tmp_path / "a.cif")},
            {"pdb_id": "B", "path": str(tmp_path / "b.cif")},
        ]
    ).to_csv(manifest, sep="\t", index=False)

    def fake_screen(path, kind, cfg):
        if path.name == "a.cif":
            return {"sample_id": "A:1"}, ""
        return None, "test_rejection"

    monkeypatch.setattr(screening, "screen_file", fake_screen)
    progress_log = tmp_path / "logs" / "screen.jsonl"
    eligible, rejected = screening.screen_download_manifest(
        manifest,
        "protein",
        tmp_path / "output",
        screening.ScreenConfig(),
        progress_log=progress_log,
        show_progress=False,
    )

    events = [json.loads(line) for line in progress_log.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["record_complete", "record_complete"]
    assert [event["index"] for event in events] == [1, 2]
    assert [event["status"] for event in events] == ["eligible", "rejected"]
    assert len(eligible) == 1
    assert len(rejected) == 1


def test_rfam_fmt2_parser_uses_query_name_column(tmp_path, monkeypatch):
    def fake_run(command, log):
        (tmp_path / "rfam.tbl").write_text(
            "1\ttRNA\tRF00005\tcomplexR::1ABC::A\t-\tCL00001\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(clustering, "_require_executable", lambda name: "cmscan")
    monkeypatch.setattr(clustering, "_run", fake_run)
    hits = clustering.run_rfam_cmscan(
        tmp_path / "rna.fa",
        tmp_path / "Rfam.cm",
        tmp_path / "Rfam.clanin",
        tmp_path,
        cpu=1,
    )
    assert hits == {"complexR::1ABC::A": {"RF00005"}}
