# PickCube ACT P0.2 training report

## Outcome

P0.2 completed one diagnosis-backed ACT repair rather than a hyperparameter
sweep. Candidate B preserved the P0.1 architecture, observations, dataset
split, optimizer family, continuous native action contract, and
`affine_tanh_v1` output. It separated seven-dimensional arm loss from the
gripper loss, added bounded close-transition supervision, and applied capped
train-only phase weights. No phase or simulator state became a policy input.

Candidate A was not run because the recorder and five-offset audit proved that
`observation[t]` is paired with `action[t]`. Candidate C was not run because
Candidate B already recovered approach and close behavior, while the archive
does not contain independent contact/lift labels that could justify a small
training-only auxiliary head.

## Frozen configuration

| Field | Value |
| --- | --- |
| Training producer | `486c0cfce975b2afcd9eec403d29016221b7dbda` |
| Dataset digest | `sha256:234b82709359f1f2fde513ab26c015f4ebfb262a16b1f16206c54eddbde44800` |
| Training identity | `sha256:d9ba1aec545ca306371da03fe333fbd2225fd9d3e8d2ba7eb8ad4840a9ca405c` |
| Seed / data | seed 0 / all 400 train episodes |
| ACT chunk | 16 continuous actions |
| Output | `affine_tanh_v1`, no clipping or projection |
| Arm / gripper / transition weights | 0.875 / 0.500 / 0.125 |
| Transition boost | 3.0, bounded |
| Phase weight exponent / cap | 0.5 / 3.0 |
| Steps / batch | 20,000 / 32 |
| Checkpoint interval | 2,000 plus validation-improvement checkpoints |
| Mixed precision | BF16 |

The four train-only phase weights were APPROACH 0.923228, PREGRASP 1.177783,
GRIPPER_CLOSING 1.346748, and TRANSPORT_OR_COMPLETION 0.838719. Their weighted
mean is one and no weight exceeds the cap.

## Screening and full training

The 5,000-step screen produced five finite, nonconstant, action-legal
checkpoints. Phase-conditioned ranking selected step 4,000 with arm MAE
0.040231, gripper MAE 0.022613, close-event F1 0.5987, and close timing error
+0.112 steps. Unlike P0.1, its predictions contained both open and closed
endpoints. H=2 then produced 8/10 pre-grasp entries, 7/10 valid closes, and one
verified but dropped grasp, so Candidate B was frozen and promoted to the one
allowed full run.

The full run completed 20,000/20,000 steps with 14 complete checkpoints and no
non-finite gradient or action-bound event. The minimum scalar validation loss
was 0.028784 at step 8,000, but scalar loss was not the selection rule. The
phase-aware offline top three were:

| Rank | Checkpoint | Arm MAE | Gripper MAE | Close F1 | Mean timing error |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | step 6,000 | 0.035385 | 0.018915 | 0.60114 | -0.1749 |
| 2 | step 7,000 | 0.032518 | 0.019814 | 0.60107 | -0.1942 |
| 3 | step 18,000 | 0.025171 | 0.018727 | 0.60091 | +0.0127 |

All 14 checkpoints passed a 2,000-anchor native and canonical action screen:
zero non-finite values, zero boundary violations, no constant output, and no
gripper endpoint collapse. The identical 10-seed behavioral screen selected
step 18,000 under the previously committed lexicographic rule because it was
the only shortlisted checkpoint with a verified grasp.

## Infrastructure recoveries

Invalid launch attempts are retained but excluded from quality evidence. The
full training itself needed no corrected rerun. One post-training launcher
passed a training config to the static-screen CLI, and the first fixed-30
launcher declared 30 in the base config instead of using the explicit bounded
override. Both failed before consuming relevant outcomes; the latter was fixed
locally with a regression test, committed, pushed, and pulled remotely before
a new evidence root was opened.

Machine-readable sources are under `artifacts/pickcube_act_p02/`, especially
`candidate_b_full_training_summary.json`,
`candidate_b_full_checkpoint_phase_metrics.json`, and
`full_checkpoint_selection_result.json`.
