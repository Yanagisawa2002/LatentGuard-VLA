# M3B — Direct Action Verifier Baselines

## Objective

Train and evaluate small structured-state baselines that predict
`failure = not final_task_success` for a 16-step candidate action chunk from the
accepted M3A `PickCubeVerifierStateV1` state. M3B compares four learned models
with three non-learned baselines, selects an architecture using aggregated
validation results across five fixed seeds, freezes that selection, and then
evaluates each checkpoint once on the untouched test split for the required
same-sample comparisons.

## Authoritative starting point and data gate

- Starting revision: `e960023a4710b8dbd3693fd763c3126115e35e81`.
- Branch: `codex/m3b-direct-verifier`.
- Required dataset digest:
  `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`.
- Training is blocked unless an independent safe reload validates the exact
  dataset digest, full-target acceptance report, 60 trajectory assignments,
  360 candidate groups, 3,240 samples, 38-dimensional state, `[16, 8]` actions,
  all-true masks, exact strong simulator evidence, 48/6/6 trajectory splits,
  2,592/324/324 sample splits, and zero split leakage.

The persistent M3A full dataset is checked before any paid GPU work. If it is
unavailable, the accepted full M3A pipeline must be rerun; the smoke dataset or
manually reconstructed samples are not substitutes.

## Input and leakage boundary

Learned models receive only the state vector, candidate action chunk, action
mask, and an internal row index used outside the model to join reports. The
binary failure target is used only by loss and metrics. Identifiers, provenance,
corruption metadata, candidate type, source trajectory, sample order, and every
post-execution outcome field are excluded from model inputs. Reporting metadata
is kept in a separate immutable table.

State and action normalization, class weights, and action-magnitude baseline
statistics are fitted on the training split only. Validation is used only for
early stopping, checkpoint and architecture selection, temperature scaling, and
threshold fitting. Test data is not loaded by selection or fitting APIs and is
evaluated only after the selection record is frozen.

## Baselines and protocol

Non-learned baselines are training-majority, training failure prevalence, and a
fixed action-magnitude score fitted without corruption metadata. Learned models
are a state-only MLP, action-only MLP, joint state-action MLP, and compact
two-layer temporal Transformer. All configurations are strict, serializable,
content-digested, and dimension checked; no pretrained model is used.

Each learned model uses the same five seeds, AdamW, finite-gradient checks,
gradient clipping, per-epoch validation, early stopping, and best-checkpoint
selection by validation corrupted-only failure AUPRC. Architecture selection is
based on the five-seed validation aggregate with Brier score, group concordance,
and parameter count as deterministic tie breakers. The selected architecture's
five frozen checkpoints are the primary result. After selection freezes, the
other learned checkpoints are also evaluated once on the same untouched test
samples solely for the required paired baseline comparisons.

The selection inventory is exactly the four configured architectures in the
canonical order and exactly the five configured seeds for each architecture.
Every seed entry binds the best-checkpoint run identity, checkpoint content
digest, checkpoint kind, selected epoch, resolved model/training digests, and
validation-prediction digest. Loading a record recomputes the canonical winner;
a missing architecture, missing seed, substituted checkpoint, or forged winner
is rejected.

## Calibration and test-isolation contract

Temperature is fitted separately for each learned checkpoint authorized for
test comparison, using that checkpoint's validation logits only. The calibration
record binds the exact checkpoint identity and bytes, resolved model
configuration, preprocessing state, and validation-prediction digest. Both
frozen threshold policies bind the same validation-prediction digest.

Before any test inference, evaluation must parse and validate the complete
frozen selection, calibration, and threshold records, verify the selected best
checkpoint kind/epoch/content digest, and recompute its validation predictions.
Only an exact digest match unlocks the untouched test split. A dry run performs
these authorization checks without running test inference. Test predictions
cannot be used to refit preprocessing, class weights, temperature, thresholds,
checkpoints, or architecture selection.

## Resume and immutable-run contract

An interrupted run resumes only from its matching periodic checkpoint and
matching history cursor. Resume restores the model, optimizer, scheduler,
Python/NumPy/PyTorch RNG state, epoch, global step, best checkpoint state, and
early-stopping patience count. Missing history, a non-periodic resume
checkpoint, an inconsistent cursor, or any identity/configuration change fails
closed.

A valid completed run is immutable. Reuse validates the exact directory
inventory, manifest bytes, complete scalar history, training summary,
completion marker, best/final/periodic checkpoint bytes, validation evaluation
and predictions, resolved configuration, class weight, and environment bindings
before it is skipped. Test artifacts are written under a separate
`evaluations/` tree and do not modify completed training directories. Explicit
force-reruns use a new numbered rerun root instead of overwriting prior evidence.

## Acceptance semantics

Acceptance targets are evaluated per architecture from the arithmetic mean of
its five calibrated test reports; no single favorable seed can satisfy the
gate. The failure-prevalence comparison is the fixed training-fitted constant
baseline, and all required candidate and group metrics must be finite. The
joint-learning claim is a separate validation-only check: the selected
architecture's aggregate validation failure AUPRC must strictly exceed both
state-only and action-only aggregates. Failed targets remain reportable results
and do not authorize test tuning or a successful joint-learning claim.

## Ordered validation and execution gates

1. Strict configuration and accepted-dataset validation.
2. CPU loader smoke with the input allowlist checked.
3. CPU finite forward/backward for every architecture.
4. Deterministic tiny overfit, checkpoint reload, and exact resume for every
   architecture.
5. Local complete CPU suite, CLI smoke paths, Ruff, mypy, and diff checks.
6. Commit `feat(m3b): add direct action verifier baselines`, push, and synchronize
   the remote checkout to the exact full SHA.
7. On one RTX 5090: dataset reload, CPU loader, one-batch GPU forward/backward,
   tiny overfit, checkpoint reload/resume, and a bounded 50-step smoke for every
   architecture.
8. Run 20 full jobs (four architectures by five seeds), select using validation,
   then perform frozen test evaluation, calibration reporting, group ranking,
   coverage-risk analysis, slice analysis, and trajectory-level comparisons.
9. Retrieve only compact sanitized reports, validate them locally, record the
   outcome in a result commit, push it, and shut down the paid GPU server.

No failed gate may be skipped. Benchmark orchestration binds the acceptance
report, benchmark/training configuration, dataset/split/preprocessing, exact Git
SHA, execution mode, and complete planned-run inventory. A stale summary or a
completed seed whose content no longer matches is rejected instead of reused.
The remote bounded smoke and full benchmark use separate immutable output roots
because execution mode participates in this identity.

## Required local validation

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
python -m latentguard train-action-verifier --help
python -m latentguard evaluate-action-verifier --help
python -m latentguard benchmark-action-verifier --help
git diff --check
```

Model-specific CPU forward/backward, tiny-overfit, checkpoint/reload/resume,
metrics, calibration, threshold, selection, bootstrap, CLI dry-run, and benchmark
resume tests are also mandatory.

## Artifacts and automatic push

Training runs, checkpoints, optimizer state, the full dataset, vectors, actions,
and raw simulator artifacts remain outside Git. Compact resolved configs,
manifests, scalar histories, metrics, selection/calibration/threshold records,
coverage and ranking tables, sanitized predictions, statistical comparisons, and
the final Markdown summary may be retrieved under
`reports/m3b/<benchmark-run-id>/` after review.

Retrieved envelopes and their content digests are reloaded before the result
commit. Reports may contain the sanitized environment and semantic identities,
but not private absolute paths, host credentials, state/action tensors, or
checkpoint payloads. The full run also emits a content-bound `summary.md`; its
five-seed aggregates retain explicit undefined statuses rather than dropping
weak or single-class metrics.

The implementation commit is pushed automatically after local validation. The
result commit `docs(m3b): record direct verifier baseline results` is pushed only
after remote acceptance artifacts are safely retrieved and independently
validated.

## Exclusions

M3B adds no policy or dynamics model, image/video input, VLM/LLM, LangMani,
additional simulator, task, robot, controller, pretrained backbone,
hyperparameter sweep, multi-GPU training, online data collection, real-robot
execution, or general safety claim.

## Current state

The implementation and local regression hardening are complete. The local CPU
suite passes with 1,098 tests and three pre-existing Windows-only symbolic-link
tests skipped because this account lacks link-creation privilege; Ruff, format,
mypy, all three CLI help paths, and diff checks pass. A four-model bounded CPU
smoke and a controlled 20-run end-to-end fixture exercise both completed and
validated immutable-result reuse. No authoritative M3A full-dataset GPU
benchmark or frozen remote test evaluation has yet been accepted from this
branch.
