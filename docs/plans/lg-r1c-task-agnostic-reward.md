# LG-R1c Task-Agnostic Reward Model Benchmark

## Objective

Determine whether frozen video-language reward models solve the cross-task
progress and natural-failure weaknesses established by LG-R1b, and whether
their outputs are usable before an unexecuted action candidate is chosen.
This milestone does not train a LatentGuard failure head and does not execute
intervention or new rollouts.

## Frozen evidence

The benchmark is bound to LG-R1b commit
`61e6ab8b5fb55eb93a7bd76c1c7bbd8957150fdd`: 320 episodes, 8 tasks,
2 LIBERO suites, 79,326 frames, 293 successes, 27 natural failures,
7,025 failure windows, and 2,315 matched success windows. The original
task-level train/validation/test split and all stage, progress, failure, seed,
and task identities remain unchanged. Final seeds `900000..900099` remain
sealed.

## Pre-registered windows

Each episode contributes endpoints at 25%, 50%, 75%, and 100% of its recorded
frames. Short, medium, and long contexts span 8, 32, and 64 control steps.
Every common comparison samples eight frames with inclusive uniform integer
spacing, using the same camera, instruction, endpoint, and temporal span.
TOPReward additionally receives a 16-frame native diagnostic only for the
long anchor context.

All 2,315 existing matched success windows and their 2,315 referenced failure
windows are included. A strict match is defined before inference as progress
distance at most 0.025, stage distance 0, and remaining-horizon distance at
most 32. No model output can add, remove, or reweight a window.

## Evaluation order

1. Validate repository isolation, frozen dataset hashes, split identity, and
   the deterministic window manifest.
2. Freeze exact LeRobot, ROBOMETER, TOPReward, Qwen, processor, tokenizer, and
   checkpoint revisions and hashes.
3. Run the online time baseline and frozen SARM on the common endpoints.
4. Run the frozen VLA-JEPA representation probe; only its linear head trains.
5. Run ROBOMETER and TOPReward strict zero-shot inference and freeze prediction
   file hashes before loading or computing any test metric.
6. Fit scalar calibration and the pre-declared linear/logistic ensemble from
   the content-bound validation-only label projection. The full labeled window
   file is read once after selection is fixed.
7. Report original split, equal task macro, leave-LIBERO-10-task-6-out,
   failure-task macro, taxonomy, stage-balanced, matched, and strictly matched
   results.
8. Evaluate the pre-registered LG-R2 promotion gate and publish the
   candidate-time limitations and feature contract.

## Claim and stop rules

Progress improvement requires held-out Spearman at least 0.60 and pairwise
accuracy at least 0.70. Outcome transfer requires task-macro success AUROC at
least 0.75. Failure promotion requires AUPRC at least
`max(2 * prevalence, 0.20)`, precision at least 0.25 and FPR at most 0.35 at
recall at least 0.60, with leave-task-6-out AUPRC dropping by at most 0.15.
Failure to meet any check stops formal reward promotion. The nearest baseline
may be retained only as a negative or auxiliary comparison.

Neither a progress correlation, scalar calibration improvement, high recall
with high false-alarm rate, representation-probe result, nor post-execution
reward can be described as task success, safety, intervention efficacy, or
pre-execution candidate selection.

## Execution and cost estimate

Local work is CPU-only schema, protocol, validation, documentation, and
artifact review. Remote work uses one RTX 5090 and an isolated overlay:

- environment/model identity and load gates: 0.5-1.5 GPU-hours;
- frozen SARM common-window inference: 0.5-1.5 GPU-hours;
- VLA-JEPA feature extraction and linear probe: 1-3 GPU-hours;
- ROBOMETER zero-shot: 2-6 GPU-hours;
- TOPReward zero-shot plus native 16-frame diagnostic: 4-10 GPU-hours;
- calibration, evaluation, and artifact checks: under 1 GPU-hour.

The initial total estimate was 8-22 GPU-hours and 12-30 elapsed hours,
including model downloads and conservative resume checks. The completed run
was materially cheaper:

- ROBOMETER common-window forward time: 1,545.6 seconds;
- TOPReward common plus native-16 forward time: 1,395.1 seconds;
- directly timed large-model forward total: 0.817 GPU-hours;
- remote phase from run creation to final compact evidence: approximately
  1 hour 47 minutes.

Small load gates, SARM, and the VLA-JEPA probe were not all separately
wall-timed, so exact total GPU occupancy is unavailable. The report preserves
that limitation rather than treating elapsed wall time as GPU time.

## Completed decision

The milestone ended as **Result B: useful partial signal, not promoted**.
ROBOMETER zero-shot improved natural-failure AUPRC over frozen SARM and met the
FPR bound at the recall operating point, but failed the pre-registered AUPRC,
precision, progress, pairwise, and task-macro success checks. TOPReward,
validation calibration, and the ensemble did not close those gaps.
`LG_R2_REWARD_BASELINE_AUTHORIZED=false`, with no authorized model.

## Exclusions and freeze audit

No new rollout, simulator, synthetic failure, foundation-model fine-tuning,
SARM adaptation, stage-ontology change, failure-head training, candidate
ranking, online intervention, RoboLab, PointWorld, LingBot-VA, ROS 2, or
LangMani work is authorized. Tracked source is changed locally, committed, and
pushed before the remote checkout executes it. Large weights, raw frames,
feature caches, predictions, and checkpoints stay outside Git; only compact
hash-bound evidence returns to the repository. The server remains online after
completion unless the user explicitly authorizes shutdown.
