# LG-RB0.1 results

## Disposition

LG-RB0.1 is **Result C**. The single upstream-quality compatibility patch fixes
the recorded-callable overlay defect and allows all 30 replay attempts to run
to completion. The complete faithful-replay gate still fails because replay
state diverges on the second and third repeat of every recording.

This is a hard stop for the RoboLab counterfactual-platform route:
`LG_RB1_AUTHORIZED=false`. No second compatibility patch, prefix replay,
same-suffix branch test, branch-isolation test, policy candidate, ranker,
world model, failure head, intervention, or training run is authorized.

## Bound source and data

- LatentGuard execution revision:
  `fa31525a4edf42376d896a59017ecd771ab38c6d`
- LatentGuard branch: `codex/lg-rb01-robolab-replay-compat`
- RoboLab version: `v0.2.1`
- RoboLab exact base:
  `0aef241fb088ca21bb4ebd24448940ed56620d17`
- Reused data: the unchanged LG-RB0 set of ten recordings, two tasks, seeds
  `810000..810004`, and 40 fixed mechanical actions per recording
- Repeats: three per recording, 30 attempts total
- No recording was generated, replaced, filtered, or reselected.
- Final seeds were not accessed.

## Recorded-config root cause and patch

The real recording does not lack the condition key. The old `_overlay()` takes
its plain-list wholesale-assignment branch and replaces each live callable
container with the lossy JSON list. Serialized callable leaves are at:

- `/subtasks[0]/conditions/banana[0][0]`
- `/subtasks[0]/conditions/banana[1][0]`
- `/subtasks[0]/conditions/rubiks_cube[0][0]`
- `/subtasks[0]/conditions/rubiks_cube[1][0]`

The two Rubiks-cube paths occur only in `RubiksCubeAndBananaTask`. The first
subtask check then calls the recorded string as
`conditional_func(**params_with_env)`, producing
`TypeError: 'str' object is not callable`.

The patch follows RoboLab's documented contract: recorded configuration
restores declarative data; the live source supplies classes, types, callables,
and runtime-owned metadata. It:

- preserves a live list or tuple containing callable runtime code;
- preserves an existing callable after a safe identity check;
- rejects missing runtime-code keys instead of injecting strings;
- admits recorded-only pure data only in explicit declarative namespaces;
- preserves live `/_instruction_variants`;
- restores the recorded resolved `/instruction`;
- classifies every skip as `EXPECTED_RUNTIME_PRESERVATION`,
  `SCHEMA_DRIFT`, or `INVALID_RECORDED_VALUE`;
- uses no `eval`, `exec`, arbitrary string execution, or task-specific repair.

Patch SHA-256:
`5b72f598c8a7fb8c07f1c3c2a35e136a805cb5b87b77e1f3cfbc4398c4939310`.
The patched tree digest is
`050397d38fb9e8ea4b9acb557b5b01f16ac64ca6`.

## State-schema canonicalization

The only allowlisted live-only empty mappings are:

- `/deformable_object`
- `/gripper`

Across 1,230 comparisons, both were strictly empty and were removed
symmetrically. No numeric state was changed. Unknown empty mappings and
non-empty mappings remain errors. The strict tolerance stayed `1e-6`, the
separately reported official RoboLab tolerance stayed `0.01`, and pixel
tolerance stayed zero.

## Faithful replay

All execution-stage denominators are complete:

| Field | Result |
| --- | ---: |
| attempted | 30 |
| env_config_overlay_completed | 30 |
| environment_created | 30 |
| replay_started | 30 |
| replay_completed | 30 |
| per_step_validation_completed | 30 |
| terminal_validation_completed | 30 |
| success_validation_completed | 30 |
| config overlay fatal errors | 0 |
| execution errors | 0 |
| initial restore failures | 0 |
| terminal mismatches | 0 |
| success mismatches | 0 |
| strict per-step state failures | 800 |
| official state-validator failures | 20 |

The repeat pattern is exact and consistent across both tasks and all five
seeds:

| Repeat | Passing recordings | Strict step failures | Official failed recordings | Maximum official error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10/10 | 0 | 0 | 0 |
| 1 | 0/10 | 400 | 10 | 0.14900782704353333 |
| 2 | 0/10 | 400 | 10 | 0.09731712192296982 |

The initial restored state has maximum absolute error zero in all attempts.
Terminal and success values also match in all attempts, but those coarse
outcomes do not establish faithful state replay. Ten first-repeat passes are
partial diagnostic evidence, not a 30/30 faithful result.

The confirmed post-patch blocker is repeat-dependent simulator state drift:
within a process-isolated recording worker, the first replay is exact and both
later replays diverge at every checked step. The repeated, task-dependent
magnitudes strongly indicate state retained outside the restored comparison
tree or incomplete runtime reset. The specific internal component remains
unverified; the milestone stop rule forbids a second patch used to localize or
repair it.

## Overlay skips

There are no fatal, unknown, schema-drift, invalid-recorded-value, or
unexpected skips. Expected runtime-preservation records are:

- `/subtasks[0]/conditions/banana`: 30
- `/subtasks[0]/conditions/rubiks_cube`: 15
- `/_instruction_variants`: 30

## Downstream gates

- Prefix replay: `not_run`, blocked by faithful-replay gate.
- Same-suffix branch determinism: `not_run`, blocked by faithful-replay gate.
- Branch isolation: `not_run`, blocked by faithful-replay gate.

Their metrics are unavailable, not zero.

## Claim boundary

The evidence supports that the compatibility patch fixes the callable overlay
failure and makes replay mechanically executable. It does not support faithful
repeated replay, anchor restoration, deterministic counterfactual branches,
branch isolation, policy takeover, or task improvement. No tolerance, task,
physics, action, recording, or selection rule was changed.
