# PickCube ACT P0.2 phase and gripper audit

## Phase evidence

The immutable demonstrations contain an official expert phase tracker. P0.2
maps those labels to diagnostic phases without adding phase information to the
policy observation. All 500 episodes contain the four evidenced phases below.

| Phase | Frames | Episodes | Train-only loss weight |
| --- | ---: | ---: | ---: |
| APPROACH | 6,395 | 500 | 0.923228 |
| PREGRASP | 3,923 | 500 | 1.177783 |
| GRIPPER_CLOSING | 3,000 | 500 | 1.346748 |
| TRANSPORT_OR_COMPLETION | 7,735 | 500 | 0.838719 |

The largest-to-smallest phase imbalance is 2.5783. Every episode has the three
recorded transitions `APPROACH -> PREGRASP -> GRIPPER_CLOSING ->
TRANSPORT_OR_COMPLETION`. Of 21,053 possible 16-step chunks, 13,273 (63.0456%)
cross at least one phase transition. Ordinary mean action loss therefore lets
long stable segments dominate the few timesteps that decide when to close.

The archive does not retain independent contact and lift labels, so
`CONTACT_OR_GRASP` and `LIFT` are not fabricated offline. Those events are
measured only in real closed-loop rollouts through public contact, pose, and
Panda grasp APIs. Phase labels remain privileged diagnostics and train-only
weights; they are never policy inputs.

## Gripper contract

- Action dimension 7 is the continuous normalized Panda mimic-joint target.
- `+1` means open; `-1` means closed.
- The demonstrations store commanded targets, not measured finger positions.
- The 18D policy state independently includes measured Panda joint positions
  and velocities, including the two finger joints.
- `affine_tanh_v1` preserves the two endpoint targets exactly within its
  declared epsilon and never clips or projects an action.
- The archive has 500 close transitions, one per episode. Their mean frame is
  20.636 (range 17--25, standard deviation 1.5998).
- Open and closed command frames are nearly balanced: 49.0096% and 50.9904%.

The defect is not sign, scale, or class-frequency inversion. In the original
eight-dimensional standard L1 objective, the gripper contributes only 1/8 of
the per-dimension loss while close timing occupies a narrow transition. The
full P0.1 validation audit also found that all 15 checkpoints predicted a
closed endpoint for every validation frame; the earlier sparse screen had
missed this global always-closed collapse.

The supported P0.1 failure classes are therefore:

- `close_transition_underweighted`;
- `always_closed_collapse`.

There is no evidence for `gripper_sign_mismatch`, `gripper_scale_mismatch`,
`always_open_collapse`, or a dataset chunk-alignment error.

## Candidate B repair

Candidate B keeps ACT, the observations, continuous native action, train split,
optimizer family, and intrinsic action bounds unchanged. Its loss separates
seven arm dimensions from the gripper and adds a bounded close-transition
term:

```text
0.875 * arm_loss
+ 0.500 * gripper_loss
+ 0.125 * transition_loss
```

Train-only inverse-frequency phase weights use exponent 0.5 and are capped at
3.0. Transition samples receive a bounded boost of 3.0. Open, closing, and
closed targets all remain supervised, preventing a constant-close output from
being rewarded as a complete solution.

At 5,000 screening steps, all five checkpoints were finite, nonconstant, and
native-action legal. Phase-aware ranking selected step 4,000 rather than the
minimum scalar-loss step 5,000. The selected screen checkpoint had arm MAE
0.040231, gripper MAE 0.022613, close-event precision 0.4310, recall 0.9800,
F1 0.5987, and mean close timing error +0.112 steps. Its predicted open and
closed endpoint rates were 0.3729 and 0.6079, so the P0.1 all-closed collapse
was removed.

## Closed-loop screening diagnosis

On the same 10 development seeds, execution horizon 2 was best:

- 8/10 entered pre-grasp;
- 7/10 issued a valid close near the cube;
- 1/10 produced a verified grasp;
- that grasp dropped before lift;
- 0/10 lifted and 0/10 completed the task;
- 0 action, simulator, or workspace integrity failures.

This is real phase progress compared with P0.1's 0/30 grasp result, but it is
not grasp competence or policy acceptance. The remaining failures center on
contact-to-grasp stability and grasp-to-lift transition. Candidate C is not
justified: the offline archive lacks non-fabricated contact/lift auxiliary
labels, and Candidate B already produced meaningful pre-grasp, close, and one
verified grasp. Adding an auxiliary phase head would be speculative rather
than a diagnosis-backed minimal repair.

Machine-readable sources are under
`artifacts/pickcube_act_p02/dataset_audit/` and the three
`candidate_b_horizon_h*.json` summaries.
