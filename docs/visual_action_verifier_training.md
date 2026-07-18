# M4B Visual Action-Verifier Training Contract

## Scope

M4B trains compact PickCube visual/action failure verifiers from the accepted
M4A M3A development dataset. It neither renders nor replays a simulator. The
M3C external images and accepted outcomes are used only after training,
promotion, checkpoint selection, calibration, thresholds, and ensemble
semantics are frozen.

The deployable input allowlist is exactly:

- one or three fixed-order RGB views, or their content-bound ResNet-18
  features;
- one candidate action chunk with shape `[16, 8]`;
- one action mask with shape `[16]`.

All provenance, split, candidate, corruption, camera, domain, trajectory,
packet, evidence, state, and outcome fields are reporting-only or privileged.
They cannot enter the student forward signature.

## Backbone and caches

The only pretrained backbone is torchvision ResNet-18 with
`ResNet18_Weights.IMAGENET1K_V1`. Its manifest binds the torchvision version,
weight enum and bytes, RGB224 normalization, and 512-dimensional output.

Frozen-feature caches are safe NPY bundles with exact file inventories. Each
row is keyed by packet plus ordered camera slot so candidate samples sharing a
packet never duplicate extraction. Cache identity binds the exact Git SHA,
visual and source dataset identities, split/source-set identity, image digest,
backbone/weight digests, preprocessing semantic, dtype, shape, and complete
array bytes. M3A and M3C caches are distinct. M3C cache creation requires the
evaluation-only CLI gate and occurs only after model freeze.

The teacher is the accepted five-seed M3B temporal structured-state ensemble.
Its cache contains per-seed raw logits, per-seed calibrated probabilities, and
their deterministic ensemble probability. It contains no raw state, actions,
paths, simulator data, or credentials. Only training-split indices may use the
teacher probability in the fixed distillation loss; the student never needs
the teacher at inference.

## Models and cost policy

Seed-zero screening trains exactly:

1. `random_resnet18_multiview_action`;
2. `frozen_resnet18_singleview_action`;
3. `frozen_resnet18_multiview_action`;
4. `frozen_resnet18_multiview_action_distilled`.

Every model uses the same masked temporal action encoder and fusion design.
The frozen multi-view students share projection, learned slot positions, and
attention pooling. The distilled model changes only the training loss.

The primary promotion metric is mean corrupted-only failure AUPRC across the
three validation domains. Tie breakers are worst-domain AUPRC, pairwise
concordance, Brier score, trainable parameter count, and stable model ID. The
validation winner, strongest direct non-distilled baseline, and at most one
essential ablation may advance. Seed zero is reused; promoted families add
only seeds one and two. M4B refuses seeds three and four even when its explicit
authorization-shaped CLI flag is supplied.

## Evaluation order

For each promoted seed, temperature calibration and maximum-balanced-accuracy
and target-recall thresholds are fitted from combined M3A validation logits
only. No per-domain calibration is permitted. The frozen three-seed ensemble
then opens canonical, strong-camera, and strong-light internal test domains
once and reports candidate classification, calibration, group ranking,
selection, coverage-risk, seed statistics, robustness drops, and deterministic
trajectory-level paired intervals.

Only afterward may M3C external images be opened. Stage A scores the fixed
eight-candidate groups independently in three visual domains and publishes
immutable manifests with `outcomes_available_during_selection=false` and an
empty outcome-path inventory. Stage B joins those candidate IDs to the already
accepted M3C outcomes. It performs no simulator call and cannot change
rankings, calibration, thresholds, or seed count.

## Persistence and lifecycle

Feature/teacher caches, checkpoints, optimizer/scaler state, raw predictions,
and visual datasets remain outside Git. Compact self-digesting scalar reports
may return to Git after strict reload and sanitization. Complete outputs are
immutable; resume accepts only exact identities and performs zero duplicate
work for completed caches, training reports, and benchmark reports.

The remote server is an execution environment only. Tracked fixes are made,
validated, committed, and pushed locally before exact-SHA synchronization.
The server, environment, datasets, caches, checkpoints, reports, and run roots
remain available after M4B. Shutdown requires a separate explicit user request
in the current task.

## M4C online inference

The accepted M4B direct and distilled three-seed ensembles are loaded frozen for
M4C. At each boundary, three state-preserving RGB224 renders occupy slots 0..2
of an exact batch of 128, with round-robin repeats filling the rest. Only the
first three feature rows are consumed. No external image, outcome, checkpoint,
calibration, threshold, or rendering configuration is used for tuning.
