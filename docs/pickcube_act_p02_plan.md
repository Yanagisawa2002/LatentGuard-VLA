# PickCube ACT P0.2 plan

## Question

P0.2 tests whether the frozen P0.1 zero-grasp result comes from temporal
misalignment, receding-horizon execution, weak grasp-transition supervision,
gripper collapse, or a more fundamental limitation of the current ACT/data
formulation. Lower scalar validation loss is not evidence of improvement; real
closed-loop approach, close, contact, grasp, lift, and success events decide the
result.

## Frozen inputs

- Starting source: `f2224d9bf3570a5bb1e51221019a98cb7997ef1a`.
- Dataset: the immutable 500-episode PickCube-v1 archive identified in
  `artifacts/pickcube_act_p02/input_manifest.json`.
- Observation: one 224x224 front-oblique RGB image plus the existing 18D Panda
  state. Phase and simulator progress are never policy inputs.
- Action: absolute eight-dimensional `pd_joint_pos` with the unchanged
  `affine_tanh_v1` bounds and no clipping, projection, fallback, or post-hoc
  repair.
- ACT chunk length: 16. Predetermined execution horizons: 1, 2, and 4.
- Development: seeds 800000 through 800029. Final seeds 900000 through 900099
  remain sealed unless development success is at least 75%.

## Ordered protocol

1. Reproduce the P0.1 action audit and fixed 30-seed step-1000 result.
2. Audit all demonstrations at offsets -2, -1, 0, +1, and +2.
3. Derive diagnostic-only phases from the recorded expert tracker and audit the
   gripper command/state contract.
4. Add per-step closed-loop progress evidence and staged 30-seed gates.
5. Rank every P0.1 checkpoint by action integrity, transition-event quality,
   phase-conditioned errors, and arm error instead of scalar loss alone.
6. Run at most three bounded candidates. Candidate A is allowed only for a
   proven alignment defect. Candidate B reweights train-only phases and
   separates arm, gripper, and close-transition losses. Candidate C is allowed
   only if B reaches pre-grasp but diagnostics prove phase representation is
   deficient.
7. Screen candidates for 5,000 steps, then run 10 fixed development rollouts.
   Promote at most one frozen configuration to 20,000 steps.
8. Evaluate horizons 1, 2, and 4 on development only, then run the fixed
   30-seed gate once for the selected checkpoint/horizon.
9. Classify exactly Result A, B, or C. Only Result A may open the sealed set,
   build a PolicyPackage, or authorize D2.

## Audit decision before training

The complete dataset audit found no alignment defect: the collector records a
pre-action observation and pairs it with the same action executed once at that
step. Offset 0 has zero phase mismatch, while nonzero offsets cross phase
boundaries. Candidate A is therefore not justified.

The expert gripper command is `+1` open and `-1` closed; endpoints are preserved
by the action transform. Open and closed frames are nearly balanced, but the
gripper contributes only one of eight dimensions to the original standard L1
loss and 63.05% of 16-step chunks cross a recorded phase transition. Candidate
B is the first and currently only justified substantive training candidate.

## Stop conditions

Stop without expanding scope if action integrity regresses, gradients become
non-finite, the bounded screen produces zero pre-grasp entries, or the allowed
candidates end in Result B/C. Diffusion Policy, VLA, RL, expert fallback, task
simplification, final-seed tuning, and a rejected-policy package are outside
P0.2.

## Completed outcome

The bounded protocol completed with Candidate B as the only substantive
candidate. Its 20,000-step full run produced 14 complete checkpoints. The
frozen offline shortlist was steps 6,000, 7,000, and 18,000; the identical
10-seed H=2 behavioral screen selected step 18,000 because it was the only
shortlisted checkpoint with a verified grasp. On the fixed 30 development
seeds it achieved 28 pre-grasp entries, 20 valid near-object closes, one grasp,
zero lifts, and zero successes, with zero action, simulator, or workspace
integrity failures. The grasp dropped before lift.

P0.2 is therefore frozen as `Result C`: the approach gate passed, but the
gripper, grasp, lift, and unchanged 75% promotion gates failed. Final seeds
900000--900099 were not accessed, no `PolicyPackage` was built, and D2 remains
blocked with zero accepted compatible policies.
