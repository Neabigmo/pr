# Documentation index

## Authoritative Review-3 documents

Use these for all new work:

1. `../RUNBOOK.md` — exact safe execution order.
2. `CODEX_EXECUTION_V3.md` — detailed agent/Codex implementation and execution contract.
3. `PROJECT_STATUS_FOR_YIHENG_V3.md` — user-facing project overview and current state.
4. `REVIEW3_FINAL_AUDIT.md` — deep audit findings, fixes and residual real-run risks.
5. `GO_NO_GO_V3.md` — formal code/data/GPU/final100 gates.
6. `DATA_PIPELINE.md` — source discovery, downloading, screening, clustering and freezing rationale.

## Historical documents

The following are retained only for provenance and must **not** be used as current execution instructions:

- `CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md`
- `PROJECT_STATUS_FOR_YIHENG.md`
- earlier `SELF_AUDIT.md` / experiment-spec iterations

Where a historical document conflicts with Review-3 code or the authoritative documents above, Review-3 wins.

## Safety reminder

A code-complete repository is not a completed experiment. Formal final100 access is allowed only after:

```text
PRIMARY_TRAINING_READY
+ CONTROL_TRAINING_READY
+ development-only runtime profile
+ EVALUATION_PROTOCOL_LOCK
```

No architecture, checkpoint, cutoff, SPIR setting, candidate budget or hypothesis may be selected using final100 outcomes.
