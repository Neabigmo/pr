import pandas as pd

from pr_pilot.adapter_pilot.scientific import ScientificProtocol, choose_candidate, validate_length_bounds


def test_length_audit_reports_out_of_range_without_dropping_rows():
    frame = pd.DataFrame({"protein_length": [40, 2001], "rna_length": [10, 500]})
    report = validate_length_bounds(frame, ScientificProtocol())
    assert report["rows"] == 2
    assert report["protein_out_of_range"] == 1


def test_selection_prefers_ratio_gate_then_worst_direction():
    candidates = [
        {"name": "ungated", "metrics": {"protein_interface_ratio": 0.8, "rna_interface_ratio": 1.01, "specificity_mean": 0.4}},
        {"name": "gated", "metrics": {"protein_interface_ratio": 0.95, "rna_interface_ratio": 0.96, "specificity_mean": 0.0}},
    ]
    assert choose_candidate(candidates)["name"] == "gated"
