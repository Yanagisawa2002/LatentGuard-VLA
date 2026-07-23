# LG-R1 next-step decision

## Decision

Do not start LG-R2.

LG-R1 establishes a useful stage-aware progress baseline on held-out seeds,
but the failure-data gate is not met: one natural failed episode was observed
where at least ten are required. No failure head, intervention policy, or
failure-sensitive threshold can be justified from this evidence.

## What is promoted

The SARM-style small baseline is the reference progress baseline for the eight
covered LIBERO tasks. Future progress models should, at minimum, compare
against its validation-only checkpoint selection and report:

- progress MAE/RMSE and rank correlation;
- stage accuracy/macro F1 with unsupported classes visible;
- same-stage and cross-stage pairwise accuracy;
- stagnation and regression recall using validation-frozen thresholds;
- held-out-seed performance;
- frozen-backbone, split, and identity checks.

This promotion is limited to progress representation. It is not a task-success
or safety promotion.

## What remains unresolved

- Failure behavior is represented by one horizon-exhaustion episode.
- Negative-progress and regression recall is weak (`0.25` on test).
- No held-out-task or held-out-suite split was statistically valid.
- Stage 7 has no current pre-action samples in the bounded feature sets.
- VLA-JEPA predictor statistics were descriptive correlations only.
- No claim can be made that progress explains most policy failures.

## Required authorization boundary

The next permissible research action is a separately authorized, pre-registered
natural-rollout expansion on untouched standard tasks and new development
seeds. It must retain the policy, action contract, horizon, and no-synthetic-
failure rules. Only after at least ten independent natural failed episodes,
plus all other LG-R2 checks, may a new milestone propose failure-head training.

The server remains online, but this milestone schedules no additional remote
work.
