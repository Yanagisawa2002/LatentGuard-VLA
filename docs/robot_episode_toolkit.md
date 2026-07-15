# Robot Episode Toolkit

RET-1 is LatentGuard's shared, simulator-independent data and evaluation layer.
It adds strict read-only quality auditing, deterministic offline Episode
iteration, and candidate-level dataset metrics on top of the existing M0 data
contract. It does not create a separate package or repository.

## Audit contract

Auditing first runs the existing strict Episode validator. A contract error is
still an error and is never converted into a quality warning. The audit then
reports findings without clipping, filling, interpolating, dropping, or
otherwise changing data. Its stable issue codes and severities are:

| Code | Severity | Meaning |
| --- | --- | --- |
| `REPLAY_HORIZON_MISMATCH` | error | Candidate horizon differs from observation count. |
| `TIMESTAMP_PERIOD_DRIFT` | warning | Observation cadence exceeds the configured tolerance. |
| `CAMERA_SET_CHANGED` | warning | Camera IDs differ from the first observation. |
| `CAMERA_FROZEN` | warning | Identical RGB frames meet the configured run length. |
| `ROBOT_STATE_FROZEN` | warning | State is identical while actions are not all near zero. |
| `ACTION_NEAR_ZERO_DOMINANT` | info | Near-zero rows exceed the configured fraction. |
| `ACTION_DISCONTINUITY` | warning | An explicitly enabled L2 jump threshold is exceeded. |
| `DUPLICATE_CANDIDATE_ACTION` | info | Different IDs contain identical action arrays. |

Frozen-camera comparison is deterministic and exact and uses RGB only. Depth is
not part of this check. The action-discontinuity check is disabled by default
because a universal robot-independent threshold would be misleading.

## Offline replay is not simulator replay

`build_replay_plan` and `iter_replay_steps` pair observation and action row at
the same index. The action horizon must exactly equal the observation count.
There is no interpolation, truncation, terminal repetition, no-op insertion,
normalization, state restoration, `env.step`, rendering, or outcome update.
`nominal_action_timestamp_s` is `step_index * control_period_s`; the original
observation timestamp remains unchanged and audit reports cadence drift.

This deterministic traversal is a debugging/data-access operation. It must not
be described as M2B exact-state replay, physical validation, or
simulator-verified evidence.

## Metrics and denominators

Episode counts describe inventory only. Success, unsafe, progress, label, and
failure metrics use the number of candidates as their denominator. RET-1 does
not define episode success: it does not treat any successful candidate, the
best candidate, or an arbitrary candidate as the action that was executed.
Empty input is rejected, so reports contain neither undefined ratios nor
NaN/Infinity.

## CLI

After creating or receiving a valid M0 Episode bundle:

```text
latentguard audit-data --input-dir .tmp/ret1-sanity --output .tmp/ret1-audit.json
latentguard summarize-data --input-dir .tmp/ret1-sanity --output .tmp/ret1-metrics.json
latentguard replay-episode --input-dir .tmp/ret1-sanity --episode-id synthetic-s00000042-e0000 --candidate-id synthetic-s00000042-e0000-candidate-000 --output .tmp/ret1-replay.jsonl
```

`audit-data` supports `--fail-on error` (default), `--fail-on warning`, and
`--fail-on never`. JSON is sorted, finite-only UTF-8 with stable indentation.
JSONL includes identifiers, timestamps, camera IDs, robot state, and action; it
never includes RGB or depth pixels. Output files must not already exist.

## Future LangMani adapter boundary

RET-1 does not import or integrate LangMani. A future typed adapter may map M3A
or M3B records into the unchanged LatentGuard Episode contract. For M3A, it
must align T actions with exactly T observations selected from T+1 simulator
states and retain the terminal state only as adapter-specific audit evidence,
not as a fabricated action step. It must map scene/task IDs, TaskSpec,
instructions, outcomes, failure codes, and provenance explicitly. Simulator
oracle data must not leak into policy observations, and only evidence produced
by trusted real simulator replay may set `simulator_replay_verified=True`.

The adapter remains responsible for framework-native conversion. LangMani,
ManiSkill, LeRobot, ROS, and simulator-native objects must not enter the core
toolkit models.
