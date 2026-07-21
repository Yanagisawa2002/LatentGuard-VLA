# WM-v0 P0 — PickCube-v1 Native ACT Policy

## Objective

Produce a native PickCube-v1 ACT policy from real successful expert execution,
then package at least one independently evaluated checkpoint through the
existing D2 fail-closed `PolicyPackage` and `PolicyRegistry` contracts.

## Fixed baseline and branch

- Baseline branch: `codex/wm-v0-d2-policy-binding`
- Baseline commit: `64d4d8713c44f9fee3275e40e1cc1fc9706d7e75`
- Milestone branch: `codex/wm-v0-p0-pickcube-act`
- Local tracked source is authoritative. The remote host runs exact pushed
  commits only and must not receive tracked-file hotfixes.

## Gates and execution order

1. Bind the installed PickCube task, Panda controller, action bounds, camera,
   and named proprioception to the accepted M2C compatibility identity.
2. Revalidate the privileged official ManiSkill motion-planning expert on 100
   independent seeds. Require at least 95 successes, zero simulator errors,
   zero non-finite/out-of-bounds actions, zero workspace violations, and real
   execution in all declared phases.
3. Only after gate 2 passes, collect at least 500 complete successful episodes.
4. Split by episode/scene seed at 80/10/10 before deriving training windows;
   validate reload, alignment, duplication, finite values, and leakage.
5. Pass configuration, CPU loader, GPU forward, forward/backward, short
   overfit, checkpoint save/load/resume, and one-episode closed-loop gates.
6. Train one standard ACT configuration and preserve genuine early, mid,
   best-validation, and final checkpoints.
7. Evaluate every checkpoint on at least 30 development seeds and promoted
   checkpoints on at least 100 disjoint final seeds. Classification thresholds
   are frozen before outcomes are read.
8. Measure checkpoint action diversity on at least 20 real anchors; package and
   validate accepted compatible policies through D2.

## Assumptions

- The accepted M2C runtime identity remains ManiSkill 3.0.1, SAPIEN 3.0.3,
  mplib 0.1.1, PickCube-v1, Panda, `pd_joint_pos`, `obs_mode=none`, and GPU
  simulation.
- Repository source, CPU validation, and public interfaces remain Python 3.11.
  Upstream LeRobot 0.6.0 declares Python >=3.12 and uses PEP 695 syntax, so the
  isolated remote ACT training/inference process uses Python 3.12.3. A direct
  Python 3.11 import probe failed at LeRobot's
  `deserialize_json_into_object[T: JsonLike]` syntax before any training. This
  narrowly documented tool exception does not change the simulator, task,
  action, observation, data, or policy contract; the combined Python 3.12
  runtime must independently pass the pinned ManiSkill/SAPIEN/MPLib versions,
  full dataset reload, GPU, checkpoint/resume, and closed-loop gates.
- Policy RGB comes from the already accepted, state-preserving M4A camera rig;
  simulator observations remain `none`.
- The expert may read cube and goal poses for demonstration generation only.
  Those fields and expert phase are reporting metadata, never model inputs.
- The official planner's public joint velocity and acceleration limit scales
  are fixed to 3.0 for the next candidate gate. The 2.0 candidate completed
  only 54/100 seeds within the unchanged native 50-step horizon; raising these
  planner limits changes waypoint density only and does not change the task,
  controller, action contract, control frequency, horizon, or executed-action
  validation.
- The final transport command may repeat its last planned joint target for up
  to 50 refinement calls, but the recording wrapper stops at the native task
  termination or 50-step Gymnasium truncation. This provides real controller
  settling time without extending the task horizon or mutating state.
- If and only if the official screw planner reports failure before executing a
  transport action, the expert invokes the official RRTConnect transport
  planner under the same target, state, controller, action checks, refinement
  policy, and native horizon. It does not choose a fallback from task outcome.

## Exclusions

- No D1 synthetic future is policy training data.
- No WM-v0 counterfactual collection, world-model/verifier training, ranking,
  selective intervention, repeated shield, Diffusion Policy, VLA, or LangMani
  checkpoint conversion.
- No threshold relaxation or post-final-outcome tuning.
- Raw images, demonstrations, checkpoints, and caches stay outside Git; only
  compact sanitized evidence and package manifests may be retrieved.

## Failure behavior

If the expert gate fails, demonstration collection and training remain blocked.
If no checkpoint reaches the predeclared acceptance criteria, the milestone is
reported as Result B without relabeling a weak policy as accepted.
