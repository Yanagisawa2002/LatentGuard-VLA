# LG-R1c task-agnostic reward report

## Decision

LG-R1c is **Result B: useful partial signal, not promoted**.
`LG_R2_REWARD_BASELINE_AUTHORIZED=false`, and the authorized-model list is
empty. The closest task-agnostic baseline is strict-zero-shot ROBOMETER, but it
does not pass the complete pre-registered gate.

This result does not establish policy task success, safety, intervention
efficacy, or the ability to rank unexecuted action candidates.

## Frozen evaluation scope

The benchmark reused the exact LG-R1b dataset: 320 episodes, 8 tasks,
79,326 frames, 293 successes, and 27 natural failures. It created no rollout,
synthetic failure, or new split. Evaluation covered 8,470 content-bound common
windows, including 4,630 matched windows, 2,315 matched pairs, and 2,279
strictly matched pairs. A validation-only projection of 4,988 rows was the
only label source available during calibration.

Zero-shot prediction identities were frozen before labels or metrics were
loaded. Validation-fitted candidates were selected without test labels and
were evaluated once on the original task-held-out test scope.

## Promotion metrics

| Candidate | Scope | Progress Spearman | Pairwise accuracy | Task-macro success AUROC | Natural-failure AUPRC | Precision at recall gate | Recall | FPR | Leave-task6 AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Online time | all | 0.918 | 0.845 | 0.206 | 0.045 | 0.084 | 1.000 | 1.000 | 0.011 |
| Frozen SARM | all | 0.217 | 0.363 | 0.465 | 0.062 | 0.074 | 0.630 | 0.730 | 0.014 |
| ROBOMETER zero-shot | all | 0.268 | 0.478 | 0.566 | 0.164 | 0.160 | 0.630 | 0.304 | 0.222 |
| TOPReward zero-shot | all | -0.101 | 0.151 | 0.515 | 0.084 | 0.109 | 0.741 | 0.556 | 0.023 |
| ROBOMETER calibrated | test | 0.130 | 0.396 | 0.719 | 0.048 | 0.048 | 1.000 | 0.253 | 0.048 |
| TOPReward calibrated | test | 0.250 | 0.117 | 0.500 | 0.024 | 0.013 | 1.000 | 1.000 | 0.024 |
| Task-agnostic ensemble | test | 0.141 | 0.396 | 0.490 | 0.067 | 0.067 | 1.000 | 0.177 | 0.067 |

The required thresholds were progress Spearman >= 0.60, pairwise accuracy
>= 0.70, task-macro success AUROC >= 0.75, natural-failure AUPRC
>= `max(2 * prevalence, 0.20)`, precision >= 0.25 and FPR <= 0.35 at
recall >= 0.60, and leave-task6 AUPRC drop <= 0.15.

ROBOMETER zero-shot is the only task-agnostic method with a clear partial
improvement over frozen SARM on natural-failure AUPRC and the FPR operating
point. It still fails AUPRC, precision, progress, pairwise ranking, and
task-macro success transfer. Validation calibration does not rescue those
failures. TOPReward is near chance or worse for the decisive transfer
metrics, and the ensemble inherits ROBOMETER's weak precision rather than
creating a promotable signal.

## Representation probe

The frozen VLA-JEPA probe trained only two scalar linear projections
(4,098 parameters). The backbone digest was identical before and after
feature extraction, with zero foundation-model backward calls. Validation
progress Spearman was 0.588, but task-held-out test Spearman fell to 0.054.
Test success AUROC was 0.761 under 97.7% positive prevalence; AUPRC was 0.993,
so that number is dominated by class prevalence and is not evidence of robust
failure detection. This is a representation diagnostic, not a trained failure
head.

## Cost and execution

The remote run was
`20260724T034646Z_lg-r1c-task-agnostic-reward_7ab788d` on one NVIDIA GeForce
RTX 5090. ROBOMETER used 1,545.6 seconds for 8,470 windows
(182.5 ms/window, 9.70 GB peak allocated). TOPReward used 1,142.0 seconds for
the common windows plus 253.1 seconds for 1,280 native 16-frame diagnostics
(17.72 GB peak allocated). These directly timed forward passes total
0.817 GPU-hours. The full remote phase from run creation to final compact
evidence was approximately 1 hour 47 minutes; not every load gate and small
baseline was separately timed, so a more precise total GPU-occupancy claim is
unavailable.

No foundation model, reward model, policy, or failure head was trained.
Large weights, features, predictions, and caches stayed outside Git. The
server was left online and GPU-idle.

## Evidence map

The decision is machine-readable in
`artifacts/lg_r1c/lg_r2_gate.json`. Full aggregate metrics are in
`task_macro_results.json`, `failure_signal_results.json`,
`leave_task6_out_results.json`, and `evaluation_summary.json`. Exact execution,
revision, download, and isolation facts are in
`remote_execution_audit.json` and `environment_validation.json`.
