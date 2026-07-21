# PickCube-v1 native ACT contract

This contract is bound to the checked-in M2C compatibility identity
`sha256:05a560fd89e0989b6d94017c6456efc535de4a0bf87ededaf468f5e2d9ef71f7`.
The runtime audit must reproduce it before expert evaluation, data collection,
training, or inference.

## Environment

| Field | Exact value |
|---|---|
| `environment_id` | `PickCube-v1` |
| environment versions | ManiSkill 3.0.1, SAPIEN 3.0.3, mplib 0.1.1 |
| `robot_uid` | `panda` |
| `controller_mode` | `pd_joint_pos` |
| `observation_mode` | `none` (policy RGB is rendered separately without advancing physics) |
| control / simulation frequency | 20 Hz / 100 Hz |
| reset seed | explicit Gymnasium uint32 seed; a fresh environment is used per episode |
| episode horizon | 50 control steps |
| success | official `is_obj_placed AND is_robot_static` evaluation |
| failure | terminal success false, timeout, unsafe state, or execution error, reported separately |
| truncation / timeout | Gymnasium time limit at 50 control steps |

## Policy observation

The deployable allowlist contains only:

- `observation.images.front_oblique`: state-preserving RGB, raw `uint8`
  HWC `[224, 224, 3]`, processed float CHW `[3, 224, 224]`;
- `observation.state`: 18 values in fixed order: qpos for the nine active Panda
  joints, followed by qvel for those same joints.

Active-joint order is:

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4,
panda_joint5, panda_joint6, panda_joint7,
panda_finger_joint1, panda_finger_joint2
```

Cube pose, goal pose, contact graph, expert phase, corruption/provenance fields,
success, and any outcome-derived value are forbidden policy inputs.

The selected camera belongs to fixed rig `pickcube_multiview_rig_v1`; its exact
intrinsics, extrinsics, pose, and content digest remain defined by
`configs/vision/m4a/camera-rig-v1.json`.

## Action

The environment consumes one finite float32 vector of dimension 8 every 0.05 s.
Clipping, coercion, repair, or reordering is prohibited.

| Index | Meaning | Unit | Lower | Upper |
|---:|---|---|---:|---:|
| 0 | absolute `panda_joint1` target | rad | -2.8973000049591064 | 2.8973000049591064 |
| 1 | absolute `panda_joint2` target | rad | -1.7627999782562256 | 1.7627999782562256 |
| 2 | absolute `panda_joint3` target | rad | -2.8973000049591064 | 2.8973000049591064 |
| 3 | absolute `panda_joint4` target | rad | -3.0717999935150146 | -0.0697999969124794 |
| 4 | absolute `panda_joint5` target | rad | -2.8973000049591064 | 2.8973000049591064 |
| 5 | absolute `panda_joint6` target | rad | -0.017500000074505806 | 3.752500057220459 |
| 6 | absolute `panda_joint7` target | rad | -2.8973000049591064 | 2.8973000049591064 |
| 7 | normalized Panda mimic-gripper target | unitless | -1.0 | 1.0 |

The arm controller has `normalize_action=False`. The gripper maps normalized
`-1` to its closed/force target (-0.01 m) and `+1` to open (0.04 m). This is a
controller command convention; the policy must not infer or overwrite it.

## Expert boundary

The project expert instruments the installed official ManiSkill solver into the
four physical stages it actually commands: `REACH_PREGRASP`,
`DESCEND_TO_GRASP`, `CLOSE_GRIPPER`, and `TRANSPORT_TO_GOAL`. It may read
privileged geometry only for demonstration generation and may act only through
seeded `reset` and validated environment `step` calls. Static and runtime gates
prohibit teleportation, direct state mutation, fabricated success, non-finite
actions, out-of-bounds actions, or missing phase execution.
