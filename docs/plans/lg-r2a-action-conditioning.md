# LG-R2a numeric action-conditioning identifiability probe

## Decision and scope

LG-R2a asks one bounded question: on the frozen real on-policy LIBERO
episodes, does an explicit numeric executed action chunk add reproducible
short-horizon predictive information beyond the current frozen VLA-JEPA
representation?

This is a diagnostic identifiability probe. It is not candidate generation,
same-state counterfactual evaluation, candidate ranking, planning, a safety
detector, or an intervention study. Even a positive result can only authorize a
separately reviewed LG-R2b pilot; it cannot start that pilot.

The exact local baseline is
`a2cdd476fe4c45e453f6d9535fd6c3ae6b2a83a5` on
`codex/lg-r1c-task-agnostic-reward`. All tracked changes are made locally.
Remote execution must use a clean checkout of the exact pushed LG-R2a revision.

## Frozen evidence and exclusions

The probe binds the accepted LG-R1b/LG-R1c evidence:

- 320 episodes, 8 tasks, 2 suites, and 79,326 frames;
- 293 successes and 27 natural failures;
- 8,470 LG-R1c evaluation windows;
- 22 `FAILED_PLACEMENT` and 5 `OBJECT_DROP` episodes;
- zero episode leakage, zero seed leakage, and no sealed-final seed overlap;
- VLA-JEPA checkpoint and processor revision
  `735d9f692981e286ade093b5046627eda876e5d0`;
- execution horizon 7, unchanged policy, task registry, and episode outcomes.

No rollout, synthetic action corruption, manufactured failure, weak
checkpoint, future observation, privileged simulator input, foundation-model
training, candidate generation, ranking, uncertainty calibration, or
intervention is allowed. ROBOMETER, TOPReward, and SARM outputs are not model
inputs.

## Pre-registered data path

The frozen LG-R1c cache has 5,000 current pooled Qwen/VLA representations with
shape `[5000, 2048]`. Each cache identity is joined to:

1. seven consecutive frozen real executed actions, shape `[7, 7]`;
2. an all-true action mask and effective action horizon 7;
3. current pre-execution 8-dimensional `observation.state` for Model D only;
4. future progress, regression, event, and terminal labels used only as targets.

Cache samples without seven real action rows are excluded. They are not padded
for the accepted probe, because a tail mask could reveal actual remaining
episode length. The general action contract still defines and validates exact
zero padding.

The target horizons are frozen at 7, 21, and 49 environment steps. Progress is
the post-action progress after the horizon minus current pre-action progress.
Stagnation is a valid-horizon delta below 0.02. Regression is a progress drop
below -0.02 or a frozen stage-regression event. The two local events use the
LG-R1b frozen `first_abnormal_step`. Terminal failure is secondary and denotes
an unsuccessful natural episode whose terminal boundary falls within 49 steps.
Time-to-failure buckets are diagnostic only.

## Evaluation design

A deterministic five-fold assignment groups by episode and seed. It assigns
natural-failure seed groups first and then greedily balances taxonomy, task,
and cached-window count. For outer fold `i`, fold `i` is test, fold
`(i+1) mod 5` is validation, and the remaining folds are training. Every test
fold must contain a natural failure. Each sample is tested exactly once after
validation-only checkpoint and threshold selection.

Reports include pooled grouped-CV results, mean/std across folds, episode-level
2,000-resample bootstrap intervals, task macro, the original split as a
compatibility view, failure-task macro, and leave-LIBERO-10-task-6-out. A fully
held-out failure task is exploratory only and is omitted if its support is too
sparse for an interpretable result.

## Models and controls

The only architecture family is a small multi-task MLP:

- A: frozen current representation only, widened for parameter matching;
- B: numeric action chunk and mask only;
- C: frozen current representation plus numeric action and mask;
- D: C plus current pre-execution proprioception;
- E: optional internal-token diagnostic only when an already frozen cache
  exists.

C uses one pre-registered concatenation-plus-action-gate fusion. Its action
encoder is capped at 500,000 trainable parameters. No architecture search is
performed. A and C must differ by less than 5% in trainable parameters. All
foundation models and the cached representation remain frozen.

The shared heads predict three progress deltas, three stagnation logits, three
regression logits, two local-event logits, and one terminal-failure logit.
Progress uses Huber loss; binary targets use BCE-with-logits. Class weights are
fit on each train fold only. Fixed loss weights are recorded in the configs.

## Sensitivity and truthfulness controls

At test time only, C is evaluated with:

- real executed actions permuted within the same task, nearest progress bin,
  same effective horizon, and another episode;
- a tighter real-action swap surrogate using the same task and, when
  available, the same stage plus nearest progress bin;
- first-action-only, endpoint-summary, and fully masked action ablations;
- absolute input-gradient attribution for the short progress output.

The swap is explicitly not a real same-state counterfactual. No permuted or
ablated action is used in training.

## Promotion and stop rules

All mandatory integrity checks must pass. C must beat A by at least 10% short
progress MAE or 0.10 short progress Spearman, and by at least 0.05 AUPRC on one
action-sensitive classification target. Real-action permutation must worsen
short progress MAE by at least 10% or reduce a classification AUPRC by at least
0.03. The direction must hold in at least four of five folds, task-macro gain
must be non-negative, and the leave-task-6-out classification result must beat
its prevalence baseline.

Result A authorizes design review for LG-R2b but no execution. Result B records
local or unstable signal and stops ranking work. Result C stops the current
numeric-action failure-head route; it does not authorize a more complex MLP.

## Remote gates, cost, and ETA

Remote execution order is:

1. exact revision and clean-checkout verification;
2. source/input validation and grouped-fold construction;
3. target materialization and CPU reload smoke;
4. one-batch GPU forward;
5. one-batch forward/backward;
6. short max-step run;
7. checkpoint save/resume;
8. four complete five-fold probe runs;
9. sensitivity and compact evaluation;
10. retrieval of compact JSON/Markdown evidence only.

Expected RTX 5090 GPU use is 0.5-1.5 GPU-hours for all lightweight heads,
including smoke/resume gates. Data preparation and statistics are expected to
take 20-45 minutes of CPU time. End-to-end ETA after remote synchronization is
approximately 2-4 hours, with a stop at the first failed safety gate. No
foundation inference, new rollout, or simulator is included.

## Release and isolation audit

The plan adds only a new `lg_r2a` namespace and does not overwrite LG-R0,
LG-R1, LG-R1b, or LG-R1c artifacts. Large caches, checkpoints, and OOF
predictions remain under `outputs/lg_r2a/<run_id>/` outside Git. Only compact
reviewed artifacts are retrieved.

The plan does not access sealed final seeds, change the frozen policy or
processor, modify LangMani, use a LangMani worktree, run a simulator, or weaken
the existing Result B. The remote server remains powered on and SSH-ready
after completion unless the user separately authorizes shutdown.

Target materialization must load each compressed frozen feature array once.
Repeated per-sample decompression is a failed resource-safety gate and must be
stopped before GPU work.

## Completion record

The accepted remote training revision is
`1f03f484c9f29e0a3dbfee767b56dbfd34795e3d`; the final no-retraining
evaluation revision is `f867f97c09e8785b91c1c94c18cae1b316795f59`. The accepted run ID is
`20260724T070334Z_lg-r2a-action-conditioning_1f03f48_seed0`.

Formal head training consumed 91.2 measured wall-seconds across four models
(0.025 GPU-hours), substantially below the 0.5-1.5 GPU-hour budget. The
successful exact-revision gates, training, sensitivity, and evaluation
completed in minutes. Earlier gate attempts were stopped before formal
training: one exposed repeated compressed-cache decompression and one exposed
secret-scanner self-matches. Both fixes were made locally, validated, committed,
pushed, and synchronized before continuation.

The final result is Result B and `LG_R2B_AUTHORIZED=false`. No rollout,
simulator, foundation training, candidate generation, ranking, intervention,
LangMani change, or final-seed access occurred. The server was left powered on
and idle.
