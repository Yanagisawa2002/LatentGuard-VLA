# LG-R2b0 real same-state counterfactual identifiability pilot

## Decision question

LG-R2b0 asks one bounded question: can the frozen VLA-JEPA policy produce
multiple real action chunks from an identical, completely restored LIBERO
state, and do those native candidates create enough within-anchor outcome
variation to justify a later LG-R2b1 ranking study?

This milestone is data and identifiability work only. It does not train a
predictor, rank candidates, select an action online, intervene in an episode,
modify the frozen policy, generate a checkpoint, or access sealed final seeds.
Result A can authorize a separately reviewed LG-R2b1 design. Result B records
real candidates with insufficient outcome variation. Result C records that the
real sampling source or complete restoration contract is not viable and stops.

The baseline is commit
`6ae64db453f835e27bba700bc1f58b227fd16939` on
`codex/lg-r2a-action-conditioning`. LG-R2a Result B remains frozen and is not
reinterpreted as action conditioning success.

## Frozen evidence and scope

The pilot reuses the accepted LG-R1b native-policy rollout line: 320 episodes,
8 tasks, 2 suites, 79,326 frames, 293 successes, and 27 natural failures. Of
the failures, 22 occurred on LIBERO-10 task 6; non-task-6 failures occurred on
LIBERO-10 tasks 8 and 9 and LIBERO-goal task 9. The existing action contract is
seven native OSC actions with seven dimensions and official VLA-JEPA
postprocessing.

The pilot selects 60 independent primary-pilot source episodes across six
pre-registered task/suite pairs, ten anchors each:

- LIBERO-10 tasks 2, 6, 8, and 9;
- LIBERO-goal tasks 6 and 9.

This gives two suites, three non-task-6 failure-bearing tasks, and two
high-success controls. Task-id 6 contributes 20 of 60 anchors (33.3%), below
the 35% cap. Each source episode contributes at most one anchor. Stage priority
is frozen in `configs/lg_r2b0/anchors.yaml`; candidate values and branch
outcomes are unavailable during anchor selection.

## Hard gate 1: real candidate source

Source A is the frozen policy's own flow-matching sampling. Every candidate
uses the same checkpoint, processor, current observation, instruction, action
API, and policy-reset boundary. Only the explicit recorded PyTorch inference
seed changes. The source audit requires:

1. repeated generation with one fixed seed is byte-identical;
2. observation preprocessing is byte-identical;
3. at least two of four different seeds produce different native chunks;
4. every candidate is finite, shape `[7, 7]`, within native bounds, and carries
   complete policy/checkpoint/processor/sampling identity.

Failure yields Result C immediately. There is no fallback to synthetic
perturbation, action optimization, another checkpoint, another policy, or a
larger model.

## Hard gate 2: complete state restoration

Ten pre-registered states spanning six task/suite pairs are each restored and
executed five times. A snapshot binds flattened MuJoCo state, exposed simulator
data arrays, controller numeric state, wrapper/task latches, Python/NumPy/CPU
Torch/CUDA Torch/environment RNG state, exact structure, and exact content.
The registered numeric tolerance is `1e-6`; rendered observation bytes,
predicate values, stage labels, termination, and success are exact.

Each repeat executes the same native seven-step candidate and the same frozen
continuation seed for a 49-step trace. The gate permits zero restoration
failure, zero terminal-result mismatch, and zero predicate mismatch. A failure
yields Result C and stops before the 60-anchor pilot. The tolerance may not be
relaxed after seeing these results.

## Candidate and branch protocol

The target is four real native candidates per accepted anchor, with at least
three pairwise meaningfully distinct candidates required. Generation attempts
are capped at eight. Exact duplicates, near duplicates, and accepted distinct
candidates remain in the audit; only the pre-registered distinct set is
executed. A candidate chunk is followed by the same frozen primary-policy
continuation for every candidate at that anchor. Continuation seed and
checkpoint are common within anchor.

Branch evidence records exact state and observation identity, executed action
alignment, the 7-, 21-, and 49-step boundaries, progress delta, object motion,
stage transition/regression, grasp/drop/stagnation-style local events,
termination, explicit success/failure/unresolved outcome, and continuation
recoverability. Environment errors are not task failures. An unresolved
horizon is not counted as failure.

All comparisons are within anchor. No cross-anchor predictor, learned score,
calibration, or outcome-conditioned selection is allowed.

## Pre-registered gates

Infrastructure and authenticity require:

- zero restoration mismatch and zero branch contamination;
- complete candidate and checkpoint metadata;
- 100% frozen-policy-generated candidates and 0% synthetic candidates;
- at least 50 valid anchors and 150 valid branches;
- at least 70% of anchors with three distinct candidates;
- exact duplicate rate no greater than 20%;
- task-id 6 branch ratio no greater than 35%.

Outcome identifiability requires non-task-6 divergence and at least one of:

- at least 30% of anchors with meaningful 49-step progress spread;
- at least 15 anchors with local-event disagreement;
- at least 10 anchors with mixed success/failure terminal outcomes.

The aggregate within-anchor ranking tie rate must be no greater than 70%.
Result A requires every gate. Real candidates with a failed outcome or ranking
gate are Result B. A failed source/restoration gate is Result C.

## Execution order and stop policy

Tracked implementation is local-only. After local validation, the exact pushed
commit is synchronized to a clean remote checkout. The server execution order
is:

1. clean exact-revision and frozen-dependency audit;
2. candidate-source audit;
3. ten-state, five-repeat restoration audit;
4. outcome-independent 60-anchor registry freeze;
5. candidate generation and diversity gate;
6. one-branch mechanics smoke in a separate runtime root;
7. 150-240 branch collection with resume markers;
8. dataset validation, within-anchor analysis, and promotion gate;
9. compact source/remote audit and retrieval.

Every failed hard gate stops downstream work. A code defect is fixed locally,
committed, pushed, and synchronized before a fresh run. Remote tracked files
are never edited. Raw snapshots, branch traces, observations, and caches stay
outside Git. The server remains powered on and SSH-ready.

## Cost and ETA

No training or backward pass is budgeted. The work is dominated by frozen
VLA-JEPA inference and LIBERO simulation on one RTX 5090:

- source plus restoration gates: 0.25-0.75 GPU-hours;
- 60 source episodes and anchor capture: 1.5-3.5 GPU-hours;
- candidate generation: 0.25-0.75 GPU-hours;
- 150-240 counterfactual branches: 2.5-6.0 GPU-hours;
- validation and analysis: under 1 CPU-hour.

The full upper-bound budget is 4.5-11 GPU-hours and approximately 7-16 hours
wall-clock, with an immediate stop at the first failed gate. These are planning
estimates, not measured results.

## Isolation and release audit

LG-R0, LG-R1, LG-R1b, LG-R1c, and LG-R2a evidence remains immutable. The plan
does not use or modify LangMani, RoboLab, PointWorld, synthetic actions, final
seeds `900000..900099`, another VLA, another checkpoint, or external outcome
data. It does not create a public success claim: even Result A means only that
this bounded real same-state dataset is sufficiently identifiable to review a
later ranking experiment.

## Completion record

Pending exact-revision remote execution. This section will be updated only
with retrieved, content-bound compact evidence.
