# PR mini-pilot self-audit — current status

The original self-audit has been superseded by the deeper v3 review.

**Authoritative current audit:** `docs/AUDIT_V3_RESOLUTION.md`

The v3 review corrected baseline execution contracts, NA-MPNN AUGC probability mapping, refit schedule semantics, scratch-control unfreezing, C/DeltaC/alpha stage ownership, canonical interface definitions, joint checkpoint validation, prior validation leakage, RNA complex-chain view reuse, nested data-efficiency subsets, final-100 compute budgeting and confirmatory hypothesis scope.

Do not infer GPU readiness from CI alone. CI validates source contracts; the real experiment remains blocked until the pinned-baseline preflight and real frozen-data audit pass.

Current high-level state:

```text
DM-ICF scientific definition         frozen for mini-pilot
core model implementation            implemented
training/refit semantics             v3 audited
external baseline wrappers           v3 corrected, must still pass pinned-source preflight in execution environment
data download/screen/freeze code     implemented
real frozen pilot manifests          not yet produced
final 100 evaluation code            implemented with confirmatory/secondary separation
GPU training                         NO-GO until real-data and baseline execution gates pass
```

See also:

- `README.md` — current project overview;
- `RUNBOOK.md` — canonical execution order;
- `docs/DATA_PIPELINE.md` — data acquisition and freezing;
- `configs/hypotheses.yaml` — confirmatory test family;
- `docs/AUDIT_V3_RESOLUTION.md` — detailed corrections and remaining limitations.
