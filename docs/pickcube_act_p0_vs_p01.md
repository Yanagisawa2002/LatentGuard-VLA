# PickCube native ACT: P0 versus P0.1

P0.1 is a controlled repair, not a model sweep. It changes only the ACT output
parameterization and the contracts needed to train and execute that
parameterization end to end.

| Dimension | P0 unbounded baseline | P0.1 bounded repair |
| --- | --- | --- |
| Dataset | 500 episodes, 21,053 frames | identical |
| Split | 400/50/50 episodes | identical |
| Inputs | RGB 224x224 + 18D Panda state | identical |
| ACT architecture | 16x8 chunk, execute 4 | identical |
| Seed/optimizer/budget | seed 0, AdamW, 20,000 steps | identical |
| Decoder-to-action semantics | unbounded decoded native target | `affine_tanh_v1`, strict native interior |
| Runtime clipping/repair | prohibited | prohibited |
| Valid checkpoints entering rollout | 0/16 | 4/15 |
| Development execution | 480/480 rejected before `env.step` | 70 legal rollout executions |
| Action-contract violations | 480 | 0 |
| Selected 30-seed success | not executable | 0/30 |
| Selected 30-seed grasp success | not executable | 0/30 |
| Dominant failure | action boundary rejection | timeout before grasp |
| Final 100 seeds | sealed | sealed |
| PolicyPackage | none | none |
| Accepted D2 registry policies | 0 | 0 |
| Classification | Result B | Result B |

## What P0.1 resolved

P0 could not answer whether ACT control quality was poor because every policy
chunk failed the native action contract before execution. P0.1 removes that
confounder. Its transform is shared by training, validation, offline inference,
closed-loop inference, queueing, screening, and the package contract; property
tests and real rollout evidence show zero violations without clipping.

## What P0.1 did not resolve

P0.1 did not turn the policy into a compatible controller. Only four early
checkpoints survived the static gripper-collapse screen, none grasped the cube,
and the strongest development candidate timed out on all 30 frozen seeds.
Consequently the 75% promotion gate failed by 75 percentage points.

The comparison therefore isolates a useful negative result: bounded output
parameterization fixes the safety/compatibility defect but is not sufficient
for closed-loop skill acquisition. P0.1 provides no basis for opening final
seeds, accepting a weak checkpoint, building a package, or restarting the D2
policy-generated pilot.

## Evidence index

- Root-cause and action-chain audit:
  [pickcube_act_bounded_audit.md](pickcube_act_bounded_audit.md)
- Formal training:
  [pickcube_act_bounded_training_report.md](pickcube_act_bounded_training_report.md)
- Static and physical evaluation:
  [pickcube_act_bounded_evaluation_report.md](pickcube_act_bounded_evaluation_report.md)
- Compact immutable JSON:
  [`artifacts/pickcube_act_bounded/`](../artifacts/pickcube_act_bounded/)
