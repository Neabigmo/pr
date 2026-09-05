# PR Mini-Pilot GO / NO-GO Checklist V3

任何一项 NO 都不得用“先跑看看”替代。

## A. 代码门

- [ ] `python -m compileall -q src tests tools` = 0
- [ ] `python -m pytest -q` = 0
- [ ] `python tools/audit_config_usage.py --config configs/pilot.yaml` = 0
- [ ] `ruff check src tests tools --select E9,F63,F7,F82` = 0
- [ ] hardened CLI smoke 全部 = 0
- [ ] review3 PR CI = green

## B. 数据门

- [ ] Protein pool = 1000
- [ ] RNA pool = 1000
- [ ] complex development = 1000
- [ ] final test = 100
- [ ] final100 frozen before prior pools
- [ ] P30 leakage = 0
- [ ] R80 leakage = 0
- [ ] Rfam leakage = 0
- [ ] exact Protein/RNA leakage = 0
- [ ] mother-sample leakage = 0
- [ ] canonical interface schema PASS
- [ ] canonical interface cutoff uniformly 6.0 A
- [ ] post-freeze local similarity audit archived
- [ ] X-ray/cryo-EM/NMR composition archived

## C. Official baseline 门

- [ ] ProteinMPNN checkout SHA matches LOCK.json
- [ ] NA-MPNN checkout SHA matches LOCK.json
- [ ] baseline converters accept all frozen samples
- [ ] ProteinMPNN preflight PASS
- [ ] NA-MPNN preflight PASS
- [ ] one-example/one-batch real-data smoke PASS
- [ ] no sample silently dropped for only one method

## D. Primary GPU 门

Before long training:

- [ ] tiny end-to-end P->R->C->Delta->Alpha->Joint smoke PASS
- [ ] Stage C only trains C
- [ ] Stage Delta only trains q+DeltaC
- [ ] Stage Alpha only trains relevance+tau
- [ ] Primary Joint keeps C frozen
- [ ] Scratch Joint all trainable from step 0
- [ ] refit schedule-prefix test PASS
- [ ] joint validation sequential pseudo-NLL runs
- [ ] no final100 accessed

## E. Final100 unlock 门

- [ ] all 3 primary seeds development complete
- [ ] all 3 primary full1000 refits complete
- [ ] `PRIMARY_TRAINING_READY.json` complete
- [ ] partner-blind controls complete
- [ ] geometry-only controls complete
- [ ] `CONTROL_TRAINING_READY.json` complete
- [ ] development-only DeltaC drift audit complete
- [ ] development-only runtime profile complete
- [ ] Tier-B GPU budget accepted before final100
- [ ] config frozen
- [ ] final100 manifest frozen
- [ ] B/C/E/F checkpoint hashes frozen
- [ ] H2 control checkpoint hashes frozen
- [ ] H1-H4 frozen
- [ ] `EVALUATION_PROTOCOL_LOCK.json` created
- [ ] no pending model-selection decision

## F. Experiment completion 门

- [ ] Tier A final100 complete for all primary seeds
- [ ] Tier B complete for analysis seed
- [ ] official ProteinMPNN final100 output complete
- [ ] official NA-MPNN final100 output complete
- [ ] frozen B/C/E/F component evaluation complete
- [ ] H2 control final100 evaluation complete
- [ ] H1-H4 confirmatory statistics complete
- [ ] broad secondary/robustness tables complete
- [ ] exact git/config/manifest/upstream SHA archived
- [ ] environment + GPU + runtime metadata archived
- [ ] failures/rejections table archived

Only then can the mini-pilot be described as **experimentally complete**.
