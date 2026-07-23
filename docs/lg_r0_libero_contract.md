# LG-R0 LIBERO observation and action contract

The machine-readable contract is
`artifacts/lg_r0/libero_contract.json`.

## Observation mapping

| Environment field | Policy field | Transform |
|---|---|---|
| `agentview_image` | `observation.images.image` | uint8 HWC → float32 CHW `[0,1]`, then flip H and W |
| `robot0_eye_in_hand_image` | `observation.images.image2` | same |
| `robot0_eef_pos` | state slots 0–2 | float32 position |
| `robot0_eef_quat` | state slots 3–5 | quaternion → axis-angle |
| `robot0_gripper_qpos` | state slots 6–7 | float32, no inferred repair |
| task language | `task` | exact native LIBERO instruction |

The checkpoint declares two 224×224 images and state dimension 8. Static
fixtures and real environment resets both match this contract.

## Action mapping

The policy emits `[batch, 7, 7]`: seven future actions with six relative OSC
pose dimensions and one gripper dimension. Closed-loop execution consumes all
seven actions before replanning. Environment bounds are `[-1,1]`.

The raw normalized policy chunk and executed action are distinct evidence:

1. the LatentGuard read-only adapter leaves the raw chunk unchanged;
2. the official VLA-JEPA postprocessor clips normalized values to `[-1,1]`;
3. it pre-snaps the normalized gripper;
4. it unnormalizes with the checkpoint statistics;
5. it maps the gripper to the official `-1/+1` convention.

No scale was selected by rollout trial and no action was repaired by
LatentGuard. Native inference emits no `action_mask`; LG-R0 records explicit
`null` plus the semantic `not_emitted_by_native_inference`.

## Episode semantics

The controller runs at the configured 30 Hz. The environment settles for ten
no-op steps at reset. Native `check_success()` defines success. Native `done`
or success terminates an episode; reaching the frozen suite horizon is reported
as `horizon_exhausted`, not model success.

The 40-episode schedule covers only task 0 in each of four suites and ten fixed
seeds per suite. It is an interface smoke baseline, not equivalent to the
official ten tasks × ten episodes × four suites protocol.
