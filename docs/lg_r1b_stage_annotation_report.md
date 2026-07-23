# LG-R1b stage annotation report

## Result

The structured stage-annotation gate passed on all 320 completed episodes and
79,326 labeled steps.

| Check | Result |
|---|---:|
| Labeled episodes | 320 |
| Labeled steps | 79,326 |
| Structured reviewed episodes | 23 |
| Critical errors | 0 |
| Automated critical errors | 0 |
| Minor disagreements | 0 (0.0%) |
| Deterministic mismatches | 0 |
| Failed-task review shortages | 0 |
| Gate status | pass |

The review queue contains at least two episodes for every task. Every task
with a natural failure includes at least two reviewed successes and two
reviewed failures when available; the one-failure task includes its only
failure.

## Adapter semantics

The eight task adapters use the reviewed `pick_place`, `multi_place`,
`toggle_place`, and `place_close` declarative families. Stage and progress
derive from current task predicates, entity relations, contact and poses.
Step index is not used as a stage. Success maps to progress 1.0, while failure
is not reset to zero. Timeout is a termination reason rather than a fabricated
failure stage. Regressions remain representable and no future terminal
outcome is backfilled into earlier labels.

The following integrity checks all passed:

- privileged fields are excluded from SARM inputs;
- repeated annotation is deterministic;
- timeout semantics and success semantics are preserved;
- stage QA thresholds are satisfied;
- the task registry and rollout manifest are content-bound.

## Review limitation

This is a structured predicate/state consistency review, not a human semantic
review of every video. It establishes deterministic agreement with the
declared task adapters. It does not prove that the shared eight-position
ontology is transferable across unseen task structures; the zero-shot result
shows that it is not reliably transferable in its current learned form.

Machine-readable evidence:
`artifacts/lg_r1b/stage_annotation_report.json`.
