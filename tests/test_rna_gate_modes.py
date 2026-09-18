import torch

from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter


def _config(mode: str) -> AdapterConfig:
    return AdapterConfig(
        geometry="G2",
        aggregation="A0",
        interaction="multiplicative",
        residual="scalar_gate",
        separate_edge_encoders=True,
        sequence_independent_attention=True,
        modality_projector=True,
        rna_gate_mode=mode,
    )


def test_rna_gate_modes_keep_protein_direction_unchanged():
    baseline = ReciprocalAdapter(_config("baseline"))
    fixed = ReciprocalAdapter(_config("fixed_quarter"))
    capped = ReciprocalAdapter(_config("capped_quarter"))

    assert baseline.p2r.fixed_gate is None
    assert baseline.p2r.gate_scale == 1.0
    assert fixed.p2r.fixed_gate == 0.25
    assert not hasattr(fixed.p2r, "gate_logit")
    assert capped.p2r.fixed_gate is None
    assert capped.p2r.gate_scale == 0.25
    assert hasattr(baseline.r2p, "gate_logit")
    assert hasattr(fixed.r2p, "gate_logit")
    assert hasattr(capped.r2p, "gate_logit")


def test_capped_rna_gate_never_exceeds_quarter():
    model = ReciprocalAdapter(_config("capped_quarter"))
    model.p2r.gate_logit.data.fill_(20.0)
    reference = torch.zeros(3, 4)
    gate = model.p2r._gate(reference)
    assert float(gate) <= 0.25 + 1e-6
