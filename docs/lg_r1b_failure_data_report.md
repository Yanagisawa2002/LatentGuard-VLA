# LG-R1b natural failure data report

## Pilot and expansion

The frozen policy produced 7 natural failures in the pre-registered
80-episode pilot. This selected Pilot B, so every primary task was expanded
uniformly in three 80-episode rounds. No task was selected for extra reruns
based on its outcome.

The completed collection contains 320 valid standard-horizon episodes:
293 successes and 27 natural failures across 8 tasks and 2 suites.

| Task | Success | Failure | Total |
|---|---:|---:|---:|
| LIBERO-10 task 2 | 40 | 0 | 40 |
| LIBERO-10 task 4 | 40 | 0 | 40 |
| LIBERO-10 task 6 | 18 | 22 | 40 |
| LIBERO-10 task 7 | 40 | 0 | 40 |
| LIBERO-10 task 8 | 38 | 2 | 40 |
| LIBERO-10 task 9 | 38 | 2 | 40 |
| LIBERO-Goal task 6 | 40 | 0 | 40 |
| LIBERO-Goal task 9 | 39 | 1 | 40 |

All 320 episodes reload successfully. Processor and checkpoint identity
completeness are 100%. There are zero environment/infrastructure errors, zero
action-contract errors, zero non-finite actions, zero policy optimizer or
backward calls, and no action corruption.

## Failure taxonomy

All 27 failures terminate by standard horizon exhaustion, but the structured
state records nontrivial behavior before termination.

Primary episode-level diagnosis:

| Primary category | Episodes |
|---|---:|
| `FAILED_PLACEMENT` | 22 |
| `OBJECT_DROP` | 5 |

Multi-label evidence:

| Category | Episodes |
|---|---:|
| `HORIZON_EXHAUSTION` | 27 |
| `STAGNATION` | 27 |
| `STAGE_REGRESSION` | 27 |
| `FAILED_PLACEMENT` | 22 |
| `OBJECT_DROP` | 5 |

The first abnormal step ranges from 55 to 448. Failures are concentrated:
22 of 27 occur in LIBERO-10 task 6. The frozen adaptation split contains 24
validation-task failures, 2 train-task failures and 1 untouched-test failure.
This satisfies the pre-registered cross-task/test presence gate, but it is not
a balanced failure corpus.

## Failure windows

The 27 failures yield 7,025 failure windows:

| Window | Count |
|---|---:|
| Pre-failure long | 2,326 |
| Pre-failure medium | 2,336 |
| Pre-failure short | 2,336 |
| Terminal failure | 27 |

There are 2,315 matched success windows and 4,710 unmatched failure windows.
Matches are without replacement and require the same task and split with
bounded stage/progress/remaining-horizon differences. Episode leakage and
seed leakage are both zero.

The window distribution is heavily concentrated in task 6 (6,054/7,025) and
stage 4 (4,758/7,025). Any later failure-head experiment must account for this
imbalance rather than treating the nominal window count as independent,
uniform evidence.

Machine-readable evidence:
`artifacts/lg_r1b/rollout_manifest.json`,
`artifacts/lg_r1b/failure_registry.json`,
`artifacts/lg_r1b/failure_window_manifest.json`, and
`artifacts/lg_r1b/dataset_manifest.json`.
