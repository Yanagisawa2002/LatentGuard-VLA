# PickCube-v1 bounded ACT evaluation report

## Result

WM-v0 P0.1 closes as **Result B: actions are legal, policy quality is
insufficient**. The end-to-end action-contract repair succeeded, but the frozen
75% development promotion gate failed. No final seed was accessed, no
`PolicyPackage` was built, the registry has zero accepted compatible policies,
and the D2 policy-generated pilot remains blocked.

This separates two claims:

- **Action legality:** passed. Every executed development action was finite and
  in the exact native interval, with no clipping or fallback.
- **Closed-loop control quality:** failed. The evaluated policies never grasped
  the cube and every episode timed out at the native 50-step horizon.

## Static checkpoint screen

All 15 retained checkpoints were queried on the same 2,000 real validation
anchors: 602 `REACH_PREGRASP`, 380 `DESCEND_TO_GRASP`, 288 `CLOSE_GRIPPER`, and
730 `TRANSPORT_TO_GOAL` frames.

- all 15 checkpoints had zero canonical/native boundary violations;
- all 15 had zero non-finite actions and were not constant-output collapsed;
- checkpoints at steps 1,000, 2,000, 3,000, and 4,000 passed the complete
  development-entry screen;
- the remaining 11 checkpoints were rejected solely because their gripper
  output collapsed near the closed endpoint;
- the screen itself passed the action contract and has digest
  `sha256:40f2ce2c3b2f1bbfaee11c4b3e175143b6d5c6e2c805adf13e7ec6939918dc9c`.

The progressive gripper collapse explains why the best validation-loss
checkpoint at step 11,000 was not a viable physical candidate. The static gate
prevented spending rollout budget on those later checkpoints without changing
the model or threshold after outcomes were seen.

## Development rollouts

Every one of the four statically valid checkpoints first ran on seeds
`800000..800009`:

| Checkpoint | Episodes | Success | Grasp success | Timeout | Action rejection | Simulator/workspace error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step 1,000 | 10 | 0 | 0 | 10 | 0 | 0 |
| step 2,000 | 10 | 0 | 0 | 10 | 0 | 0 |
| step 3,000 | 10 | 0 | 0 | 10 | 0 | 0 |
| step 4,000 | 10 | 0 | 0 | 10 | 0 | 0 |

The predeclared ranking selected step 1,000 as the top initial candidate. Its
fixed 30-seed extension on `800000..800029` yielded:

- 0/30 successes, Wilson 95% interval `[0.0, 0.11351339317396876]`;
- 0 grasp successes, drops, or release failures;
- 30 timeouts and mean episode length 50;
- 0 action-contract violations;
- 0 simulator errors and 0 workspace violations;
- 390 real ACT queries;
- p95 policy-query latency `0.00895929` seconds;
- fixed-seed action and outcome reproduction passed.

The report retains 40 initial and 30 extended executions (70 total execution
records). The extension intentionally includes the same first ten seeds for the
selected checkpoint; there are 60 unique checkpoint/seed pairs.

The failure has therefore moved from P0's pre-execution boundary rejection to
pre-grasp/grasp control: legal joint targets are produced and executed, but the
cube is never grasped before the horizon expires. There is no evidence of a
transport, drop, or release problem because those stages are never reached.

## Diversity and promotion

The early, mid, best-validation, and final roles were compared on 20 real
anchors. All four roles were action-contract valid, all six role pairs produced
different chunks on all 20 anchors, and no exact checkpoint duplicate existed.
Gripper disagreement was zero while arm chunks differed, so the diversity is
real but does not provide a second compatible policy. The report digest is
`sha256:5991a04b8524f0811afbf5a6d10b257bc3d91c7a5475445f5658c7abc29faf38`.

The frozen promotion gate required at least 75% success plus zero action
rejections, simulator errors, and workspace violations. The selected checkpoint
met the safety/contract clauses but scored 0%, so:

```text
development_gate_passed = false
promoted_checkpoint_id = null
final_seed_schedule_accessed = false
final_evaluation.status = NOT_RUN_PROMOTION_GATE_FAILED
```

No threshold was relaxed and no final episode was fabricated. The rejected D2
registry entry is structurally valid only with the explicit diagnostic
`--allow-blocked` mode and contains a null package.

## Interpretation and next bounded question

P0.1 proves that legality can be made intrinsic across training, validation,
offline inference, queueing, closed-loop execution, and future package loading.
It also proves that a lower validation loss is insufficient: the policy can
minimize imitation loss while saturating the gripper closed and failing to
reach a grasp-compatible arm trajectory.

The next investigation should remain on ACT fundamentals before introducing a
different policy family: inspect time/phase weighting, horizon alignment,
gripper supervision, and whether the observation/action representation makes
the pre-grasp transition learnable. That is a new authorization; P0.1 itself
does not start it.
