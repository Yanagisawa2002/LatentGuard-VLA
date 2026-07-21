# PickCube-v1 Native ACT bounded-action audit

## Scope and conclusion

This is the pre-training root-cause audit for **WM-v0 P0.1 End-to-End
Bounded ACT**. It binds the unchanged 500-episode, 21,053-frame P0 dataset to
the exact PickCube-v1 Panda `pd_joint_pos` controller contract.

The dataset is not the source of the P0 action-contract failure. Every recorded
native target is finite and inside the exact environment Box. The P0 failure is
caused by an unbounded final `Linear` ACT regression head followed by the old
action mean/standard-deviation inverse transform. All audited P0 anchor
violations occur in action dimension 7, the continuous normalized gripper mimic
target. No audited arm dimension exceeds its native bound.

P0.1 changes only the action output parameterization. It does not widen a
bound, clip an action, alter the controller, change the data split, change ACT,
or add an expert fallback.

## Exact native action contract

The action is the absolute controller target accepted by ManiSkill 3.0.1's
Panda `pd_joint_pos` controller at 20 Hz. It is not a delta and is not an
increment to a queued command.

| Index | Stable name | Lower | Upper | Unit | Physical semantic |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | `panda_joint1_target_rad` | -2.8973000049591064 | 2.8973000049591064 | radian | Panda joint 1 absolute position target |
| 1 | `panda_joint2_target_rad` | -1.7627999782562256 | 1.7627999782562256 | radian | Panda joint 2 absolute position target |
| 2 | `panda_joint3_target_rad` | -2.8973000049591064 | 2.8973000049591064 | radian | Panda joint 3 absolute position target |
| 3 | `panda_joint4_target_rad` | -3.0717999935150146 | -0.0697999969124794 | radian | Panda joint 4 absolute position target |
| 4 | `panda_joint5_target_rad` | -2.8973000049591064 | 2.8973000049591064 | radian | Panda joint 5 absolute position target |
| 5 | `panda_joint6_target_rad` | -0.017500000074505806 | 3.752500057220459 | radian | Panda joint 6 absolute position target |
| 6 | `panda_joint7_target_rad` | -2.8973000049591064 | 2.8973000049591064 | radian | Panda joint 7 absolute position target |
| 7 | `panda_gripper_normalized_target` | -1.0 | 1.0 | normalized unitless | Continuous normalized mimic-joint target; -1 is closed and +1 is open, mapping to controller physical targets [-0.01, 0.04] m |

The gripper controller contract is a continuous Box dimension. The expert data
happens to use its two endpoints, but that observation does not make the
environment action discrete. P0.1 therefore uses the same differentiable affine
`tanh` parameterization for all eight dimensions. A categorical gripper head
would change the controller-output contract and is not justified by the
installed API.

## End-to-end lifecycle

1. The installed official motion-planning solver returns a native 8D
   `pd_joint_pos` action.
2. The collection proxy validates it against the native Box, captures its exact
   pre-step bytes, passes an unchanged copy to `env.step`, and checks that the
   caller-visible value was not mutated.
3. `save_demo_episode` writes that native floating action to `actions.npy`.
   It is neither policy-normalized nor an intermediate planner representation.
4. The P0 loader returned the native chunk and the LeRobot preprocessor applied
   train-only action mean/std normalization.
5. P0 ACT trained against those mean/std-normalized targets. Its final
   `action_head` was a plain unbounded `torch.nn.Linear`.
6. P0 inference applied the standard LeRobot postprocessor, which inverted the
   action mean/std transform. A small normalized overshoot was therefore mapped
   to a native overshoot.
7. Temporal ensembling was disabled (`temporal_ensemble_coeff=None`). No
   non-convex aggregation ran.
8. The project did not call LeRobot `select_action` and did not maintain a
   cross-query action queue. It requested one 16-action chunk, executed the
   first four already-native actions, and replanned. There was no residual
   addition, duplicate queue accumulation, or second inverse normalization.
9. The existing native boundary validator checked every action immediately
   before execution. It rejected P0 output without clipping, repair, zero-action
   substitution, expert fallback, or `env.step`.

P0.1 instead maps each legal native dataset target once to canonical `[-1, 1]`,
uses LeRobot `IDENTITY` normalization for the action feature, squashes the
decoder output before ACT computes its reconstruction loss, and maps that
bounded canonical value once to the native Box. Observation normalization stays
unchanged.

## Dataset target audit

The full 21,053-frame audit found zero nonfinite values and zero native-bound
violations. Observed native minima and maxima were:

| Dimension | Observed minimum | Observed maximum | Exact native-bound samples |
| ---: | ---: | ---: | ---: |
| 0 | -0.1659023 | 0.1703211 | 0 |
| 1 | -0.1044329 | 0.9477294 | 0 |
| 2 | -0.0836286 | 0.0879274 | 0 |
| 3 | -2.3596661 | -1.3685579 | 0 |
| 4 | -0.0799839 | 0.0646659 | 0 |
| 5 | 1.8842217 | 2.8188231 | 0 |
| 6 | 0.5668733 | 1.0010501 | 0 |
| 7 | -1.0 | 1.0 | 21,053 |

The exact lower/upper gripper counts are 10,735 and 10,318 respectively. This
is deliberate expert open/close saturation, not an illegal target. The
canonical target is exactly -1 or +1 for that dimension. P0.1 reports endpoint
frequency and saturation explicitly and does not shrink the legal Box.

The complete min/max, p0.1, p1, p5, p50, p95, p99, p99.9, bound-distance,
phase, and split statistics are generated from the immutable dataset by
`scripts/audit_pickcube_action_pipeline.py` into
`artifacts/pickcube_act_bounded/action_target_distribution.json`.

## P0 decoder and normalization failure

The P0 train-only gripper mean was approximately `-0.02037664` and its standard
deviation `0.99979237`. Consequently, legal native endpoints became about
`-0.9798268` and `1.0205885` in the P0 regression space. That space was not the
native Box and the final Linear layer had no output constraint.

Observed P0 decoder-normalized gripper ranges and native maximum exceedances
were:

| Role | Decoder-normalized gripper range | Maximum native exceedance |
| --- | --- | ---: |
| early | [-1.51751, 0.69348] | 0.5375752 |
| mid | [-1.07593, 1.13469] | 0.1140766 |
| best/final | [-1.02447, 1.09467] | 0.0740651 |

Across the 20 diversity anchors, all 20 violations were gripper violations: 16
were at `CLOSE_GRIPPER` and four at `DESCEND_TO_GRASP`. The arm remained legal.
The reduction from early to best correlates with training progress, but never
establishes a mathematical action guarantee.

## Why validation loss was insufficient

The P0 best validation loss of `0.1042463` is an average reconstruction
objective. It does not encode a hard per-dimension Box constraint. A low average
loss can coexist with one unsafe dimension, and inverse mean/std normalization
can turn a modest normalized overshoot into an illegal native target. Neither
the loss nor the unbounded Linear layer proves that every finite output is
legal.

P0.1 makes legality an architectural invariant:

`raw logits -> tanh(raw) * (1-eps) -> exact affine native Box map`.

The boundary validator remains an assertion and fail-closed safety gate. It is
not used as a runtime repair.

## Aggregation, queue, and checkpoint compatibility

Temporal aggregation is disabled for the frozen P0/P0.1 comparison. The shared
library also exposes only an explicit convex aggregation helper and tests that
weights are nonnegative and sum to one. P0.1 queues no actions across policy
queries; the four executed actions are already legal native actions from the
current chunk.

P0.1 checkpoints use schema `pickcube_act_bounded_v1` and contain
`pretrained_model/action_transform.json`. Their training identity binds exact
bounds, names, units, continuous/discrete inventory, gripper convention,
transform version, temporal mode, dataset digest, source commit, chunk horizon,
and execution horizon. The bounded loader rejects an old P0 checkpoint rather
than silently assigning it new semantics. Old P0 checkpoints remain immutable
and available only through their historical unbounded experiment identity.
