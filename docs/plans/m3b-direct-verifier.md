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
raw per-sample predictions, and raw simulator artifacts remain outside Git.
Per-sample predictions may be retrieved only into ignored audit staging so their
digests and aggregate metrics can be checked. Compact resolved configs,
sanitized manifests, scalar histories, metrics, selection/calibration/threshold
records, coverage and ranking tables, statistical comparisons, and the final
Markdown summary may be committed under `reports/m3b/<benchmark-run-id>/` after
review.

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

## Accepted execution and result

The implementation commit is
`87280afc294c6c9eda038ad4c5d7b1972249fa37`. The authoritative M3A archive
exposed its exact action dtype as little-endian float64, so the localized
archive/batch-boundary correction was committed as
`91a8910c572d843ac739be35ffd93d9b821d3395`. A Linux gate then exposed a
Windows-specific interpreter path in one test; the portable test fix and exact
remote execution revision is
`46f15d8cf50ac45027b62b0d73a4f2f49ac30224`. All three revisions were pushed
before remote use.

The local suite passes with 1,099 tests and three Windows-only symbolic-link
privilege skips. The exact remote revision passes all 1,102 Linux tests with no
skips, plus Ruff, format, and mypy. The accepted dataset dry-run revalidated
digest `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`,
the fixed inventory and split, and created no output.

Remote run `20260716T071853Z_m3b-smoke_46f15d8` passed all four RTX 5090 GPU
forward/backward, tiny-overfit, checkpoint reload, exact resume, bounded
50-step, and memory gates without test evaluation. Full run
`20260716T072051Z_m3b-full_46f15d8` completed four architectures by seeds
`[0, 1, 2, 3, 4]`, froze validation-only selection, and only then evaluated the
test set. A second `--resume` invocation strictly reloaded the complete output
without retraining and reproduced report digest
`sha256:9e569db6cbca9626380373bea51af31bf66b3e73bc6fa2af22fb9d1dc9396bce`.
The environment was one NVIDIA GeForce RTX 5090, driver 580.76.05, CUDA 12.8,
PyTorch 2.8.0+cu128, Python 3.11.15, and NumPy 1.26.4.

Validation selected `temporal_state_action_verifier` at mean corrupted-only
failure AUPRC 0.924928, ahead of joint MLP 0.908771, action-only 0.730439, and
state-only 0.386841. Its untouched candidate-level test means were failure AUPRC
0.894784, ROC AUC 0.971208, calibrated Brier 0.060384, and calibrated ECE
0.061362. Corrupted-only pairwise concordance was 0.986349 and top-1 success was
1.0. Three of four learned architectures passed all fixed acceptance targets;
state-only did not. The selected temporal model's paired test interval versus
the joint MLP included zero, and the joint MLP's calibrated test AUPRC was
slightly higher, so no clear test superiority claim is made.

Reviewed compact evidence is under
`reports/m3b/20260716T072051Z_m3b-full_46f15d8/`. It contains no checkpoints,
optimizer state, per-sample predictions, raw runtime manifests, private absolute
paths, state vectors, or action chunks. The retrieval manifest digest is
`sha256:0c16dd8f7ace4b7e58c36d778df01033f7c29fb6195613bf8167a3dab59765b0`;
the detailed human review is `review.md`. The result commit uses the required
subject `docs(m3b): record direct verifier baseline results`. The paid server is
shut down only after that commit is pushed and local/upstream/GitHub SHA
equality is confirmed.
