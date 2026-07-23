# LG-R1b next-step decision

## Decision

LG-R1b has two simultaneous outcomes:

1. **Natural-failure data readiness: pass.** The pre-registered mechanical
   LG-R2 data gate passes with 320 valid rollouts, 27 natural failures, four
   failed tasks, 7,025 failure windows, 2,315 matched success windows, passed
   stage QA, a held-out-test failure, complete identities and zero leakage.
2. **SARM held-out-task generalization: Result C.** Zero-shot progress MAE is
   0.2112, Spearman is 0.3775 and pairwise accuracy is 0.5245. Lightweight
   task-level adaptation is only partially helpful and leaves major stage and
   temporal failures.

`artifacts/lg_r1b/lg_r2_gate.json` therefore preserves
`LG_R2_AUTHORIZED=true` exactly as defined by the pre-registered data
availability gate. This field means that the failure-data prerequisites are
met; it is not permission to start a failure-head run in the face of Result C.

## Recommendation

Do not start LG-R2 training directly. First design and obtain authorization
for a task-general progress representation with a unified, task-independent
stage ontology (or replace the current task-specific stage classifier). Its
minimum falsification test should reuse the frozen task-level split and
compare against the same frozen zero-shot and same-sample adaptation
baselines. It must improve progress ordering without worsening monotonicity or
stage transfer.

The failure corpus is sufficient to retain as an LG-R2 candidate dataset, but
its concentration in one task and one stage must be handled explicitly.

No RoboLab pivot is required for failure yield: standard LIBERO produced
enough natural failures under the frozen gate. RoboLab remains a future
option only if a later milestone needs broader failure modes or stronger
physical diversity. This milestone does not authorize or execute it.

No additional LIBERO episodes, SARM training, failure-head training,
candidate ranking, intervention, VLA-JEPA fine-tuning, RoboLab execution,
LangMani work, or final-seed access should occur without a new instruction.
