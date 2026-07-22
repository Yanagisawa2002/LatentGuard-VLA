# PickCube ACT P0.2 temporal-alignment audit

## Conclusion

The 500-episode PickCube-v1 demonstration archive is aligned at offset zero.
The collector stores the post-reset/pre-action observation at simulator step
`t`, records the exact action presented to the environment, and executes that
action once in the same transition. ACT therefore receives `observation[t]`
and is supervised with a chunk beginning at `action[t]`. Candidate A was not
run because there is no observation/action or chunk-start defect to repair.

This conclusion comes from the recording interception contract and exact
episode boundaries, not from choosing whichever offset gives the smallest
robot tracking error. A joint-position controller is expected to respond after
the command, so measured `qpos[t+1]` being numerically closest to a nearby
target does not redefine the stored supervision timestamp.

## Frozen archive

- Dataset digest:
  `sha256:234b82709359f1f2fde513ab26c015f4ebfb262a16b1f16206c54eddbde44800`.
- Episodes/frames: 500 / 21,053.
- Split: 400 train, 50 validation, 50 test.
- Control frequency: 20 Hz.
- Observation: pre-action RGB plus the existing 18D Panda state.
- Action: the same absolute eight-dimensional `pd_joint_pos` command executed
  by the environment.
- Chunk length: 16; padding is zero storage with a true `action_is_pad` mask.
- Reset frame: first post-reset, pre-action observation.
- Terminal contract: `T` pre-action observations and `T` actions; no fabricated
  `T+1` observation, terminal no-op, repeated transition, frame skip, or
  cross-episode padding.

The archive has no independent wall-clock timestamp field. This is reported as
an unavailable field rather than reconstructed. Step identity is established
by the recording call order and the one-action-per-step interception evidence.

## Offset audit

| Target offset | Valid pairs | Phase mismatch | Mismatch rate | Controller tracking proxy MAE |
| ---: | ---: | ---: | ---: | ---: |
| -2 | 19,553 | 3,000 | 15.343% | 0.005535 |
| -1 | 20,053 | 1,500 | 7.480% | 0.004258 |
| **0** | **20,553** | **0** | **0.000%** | 0.010894 |
| +1 | 20,553 | 1,500 | 7.298% | 0.018027 |
| +2 | 20,053 | 3,000 | 14.960% | 0.025537 |

Offset `-1` has the smallest tracking proxy because measured joints lag a
position target. Selecting it would instead pair observations with commands
from an earlier semantic phase in 1,500 cases. Offset zero is the only tested
offset that both matches the recorder contract and preserves every recorded
phase boundary.

## Action-chunk execution

Training starts each 16-action target at the same frame as the input
observation. The evaluator predicts 16 actions and uses a receding-horizon
queue with no temporal aggregation. The P0.2 development audit changes only
how many leading actions are executed before the next policy query; it does not
change stored labels or the trained chunk length. Horizons 1, 2, and 4 use the
same checkpoint, seeds, simulator configuration, and action transform.

## Decision

`temporal_alignment_defect_found=false`, `selected_offset=0`, and Candidate A
is skipped. P0.1's zero-grasp result cannot be attributed to a one-step label
shift. The remaining supported causes are transition supervision diluted by
ordinary per-dimension L1, late-checkpoint gripper collapse, and closed-loop
phase/contact instability.

The machine-readable source is
`artifacts/pickcube_act_p02/dataset_audit/temporal_alignment_audit.json`.
