# LG-R2a numeric action contract

The machine-readable source of truth is
`artifacts/lg_r2a/action_contract.json`.

The frozen policy emits seven actions and the controller executes all seven
before replanning. Each action has seven dimensions in this order:

| Index | Name | Semantics |
|---:|---|---|
| 0 | `delta_x` | checkpoint-normalized relative translation |
| 1 | `delta_y` | checkpoint-normalized relative translation |
| 2 | `delta_z` | checkpoint-normalized relative translation |
| 3 | `delta_axis_angle_x` | checkpoint-normalized relative rotation |
| 4 | `delta_axis_angle_y` | checkpoint-normalized relative rotation |
| 5 | `delta_axis_angle_z` | checkpoint-normalized relative rotation |
| 6 | `gripper` | official binary command: -1 open, +1 closed |

The executed values are official VLA-JEPA postprocessor outputs after clipping
the normalized policy output, pre-snapping its gripper, applying the frozen
checkpoint statistics, and converting the gripper to the official convention.
All native environment values must lie in `[-1, 1]`.

Native inference emitted no action mask and LG-R0/R1 recorded `null` with
`action_is_pad=false`. LG-R2a materializes `true` for each real row. The general
padding representation is an exact all-zero action row with `mask=false`; any
non-zero masked value is invalid. Accepted probe samples have seven real rows,
an all-true mask, and effective action horizon 7. Incomplete tail chunks are
excluded rather than exposing actual remaining episode length through padding.

Validation checks exact `[N, 7, 7]` shape, boolean `[N, 7]` mask, finiteness,
bounds, exact zero padding, exact -1/+1 gripper values, and reproducibility from
the frozen action/postprocessor identity.
