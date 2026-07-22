# PickCube ACT P0.2 evaluation report

## Closed-loop result

The selected step-18,000 Candidate B checkpoint was evaluated once on the
fixed seeds 800000--800029 with execution horizon 2. All 30 episodes were real
ManiSkill `PickCube-v1` rollouts using native absolute `pd_joint_pos` actions.
There was no clipping, projection, expert fallback, zero-action replacement,
task simplification, or privileged policy input.

| Gate | Requirement | Observed | Result |
| --- | ---: | ---: | --- |
| Action integrity | 0 failures | 0 | pass |
| Enter pre-grasp | at least 24/30 | 28/30 | pass |
| Valid close near cube | at least 21/30 | 20/30 | fail |
| Verified grasp | at least 18/30 | 1/30 | fail |
| Lift | at least 15/30 | 0/30 | fail |
| Task success | at least 23/30 (75%) | 0/30 | fail |

The one verified grasp dropped before lift. All 30 episodes timed out. The
earliest progress taxonomy was 2 never-enter-pregrasp, 8 close-too-early, 12
contact-without-grasp, 1 grasp-without-lift, and 7 timeout-after-progress.
There were zero non-finite or out-of-bound actions, simulator errors, workspace
violations, and release failures. Fixed-seed action and outcome reproduction
for seed 800000 matched exactly.

## Execution horizon

The predetermined horizon audit used the same 5,000-step screen checkpoint and
the same ten seeds:

| H | Pre-grasp | Valid close | Grasp | Lift | Success | Queries | Smoothness L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7/10 | 4/10 | 0/10 | 0/10 | 0/10 | 500 | 0.5344 |
| 2 | 8/10 | 7/10 | 1/10 | 0/10 | 0/10 | 250 | 0.2671 |
| 4 | 7/10 | 6/10 | 0/10 | 0/10 | 0/10 | 130 | 0.0957 |

H=2 was frozen before the full run because it was the only horizon with a
verified grasp and had the strongest close/pre-grasp evidence. H=1 replanned
more often but did not improve contact; H=4 was smoother but lost the grasp
event, consistent with more open-loop error around a short phase transition.

## Required questions

1. **Was the dataset temporally aligned?** Yes. The canonical contract is
   `observation[t] -> action[t]`; offset zero alone preserved every phase
   boundary.
2. **Was the gripper action contract correct?** Yes. `+1` is open, `-1` is
   closed, demonstrations store commands, and forward/inverse bounded transforms
   preserve endpoints.
3. **Why did P0.1 produce zero grasps?** The original ordinary L1 objective
   diluted one gripper dimension and narrow close transitions; full checkpoint
   audit also exposed global always-closed endpoint collapse. It was not a sign,
   scale, or timestamp error.
4. **Which repair was accepted for evaluation?** Candidate B's bounded
   phase/gripper-aware loss, with no architecture or deployed-action change.
5. **Did approach improve?** Yes, from no grasp progress in P0.1 to 28/30
   pre-grasp entries; this is not policy acceptance.
6. **Did close behavior improve?** Yes, the endpoint collapse disappeared and
   20/30 closes occurred near the cube, but the 21/30 gate still failed.
7. **Did grasp and lift emerge?** One verified grasp emerged and then dropped;
   no lift emerged. Grasp competence did not emerge.
8. **Did execution horizon matter?** Yes. H=2 was the only tested horizon with
   a grasp and was frozen for full development.
9. **Was the 75% promotion gate reached?** No; success was 0/30.
10. **Was the sealed set accessed?** No; 0/100 final seeds were opened.
11. **Was a PolicyPackage built?** No.
12. **Is D2 authorized?** No; accepted compatible policy count remains zero.

## Classification

P0.2 is **Result C: ACT still fails to acquire grasp**. Actions are valid and
the bounded temporal/gripper/phase investigation is complete, but 1/30 grasps
is below the 18/30 gate and no lift or task success occurred. The next policy
family experiment, if separately authorized, must start as a new phase; it is
not part of this result.

Machine-readable sources are `development_rollout_summary.json`,
`staged_gate_decision.json`, `final_result.json`,
`sealed_set_access_decision.json`, and `policy_registry.json` under
`artifacts/pickcube_act_p02/`.
