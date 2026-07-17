# Direct Action Verifier baselines

M3B predicts whether one candidate action chunk will ultimately fail the fixed
PickCube task when execution continues with the content-bound source policy. The
binary target is `1 - final_task_success`. The verifier scores actions; it is not
a policy, planner, dynamics model, or general safety monitor.

## Accepted input boundary

The training loader first performs the complete safe M3A dataset reload and
requires the accepted full-target report and exact dataset digest. It then
projects every sample into an immutable model table containing only:

- the 38-dimensional `PickCubeVerifierStateV1` vector;
- the `[16, 8]` candidate action chunk;
- the 16-element action mask;
- the binary failure target for loss and metrics; and
- an internal integer row index used only to join reports.

Only the state, action, and mask tensors are accepted by model `forward`
methods. Sample, group, trajectory, evidence, adapter, corruption, severity,
candidate-type, split, filesystem, and outcome-derived fields remain in a
separate reporting table. They cannot be supplied as learned features.

The accepted archive contract is float32 for state and float64 for actions,
matching the authoritative M3A serialization exactly. After dtype, shape, and
finiteness validation, batching performs the sole controlled conversion of
actions to float32 for the configured models; malformed arrays are never cast
into compliance.

## Training-only preprocessing

State mean and standard deviation are fitted per state component using training
rows. Action statistics are fitted per action component across valid training
steps; masked padding never participates. A configured standard-deviation floor
is applied deterministically. The frozen preprocessing record binds the dataset
digest, training-membership digest, counts, statistics, and floor. Validation
and test data use that record without adaptation.

Class weights and the action-magnitude baseline are also fitted from training
only. No validation or test target contributes to them.

## Baselines

Three non-learned references are reported: training-majority, training failure
prevalence, and an action-magnitude score that does not inspect corruption
metadata. Four small randomly initialized neural models are compared:

1. state-only MLP;
2. action-only MLP;
3. state-and-action MLP;
4. temporal state-and-action Transformer with two encoder layers.

The temporal model uses mask-aware action aggregation. Every architecture is
strictly configured and dimension checked, and the default parameter count is
below two million. No pretrained weights, images, text embeddings, simulator
objects, or metadata inputs are used.

## Optimization, selection, and test freeze

Training uses BCE-with-logits (unweighted or a training-only positive class
weight), AdamW, deterministic explicit seeds, bounded batches, gradient
clipping, per-epoch validation, and early stopping. The best checkpoint for each
run is selected by validation corrupted-only failure AUPRC.

Each architecture runs the same five seeds. Architecture selection uses the
five-seed aggregated validation corrupted-only failure AUPRC, then lower Brier
score, higher pairwise concordance, and lower parameter count as deterministic
tie breakers. Selection produces an immutable record before test data is
evaluated. All five checkpoints of the selected architecture are then evaluated
on the untouched test split as the primary result. Only after selection freezes,
the other learned checkpoints are evaluated once on the same test samples for
paired baseline comparisons; their test results cannot change the selection.

The selection record requires the canonical four-architecture inventory and
exact five-seed inventory. Each seed result identifies one best checkpoint by
run identity, byte-content digest, checkpoint kind and epoch, resolved
model/training configuration digests, and the digest of its ordered validation
labels and logits. Deserialization recomputes the aggregate winner and rejects a
missing seed/model, changed training protocol, substituted checkpoint, or
declared winner that does not follow the metric and tie breakers.

## Calibration, thresholds, and metrics

A single positive finite temperature is fitted on validation logits and frozen.
It preserves score ranking. The calibration state binds the exact checkpoint
identity and content digest, resolved model configuration, preprocessing state,
and ordered validation-prediction digest. Two validation-only threshold policies
are frozen: maximum balanced accuracy and the highest threshold satisfying
target failure recall when possible. The threshold record must bind that same
validation-prediction digest.

Test evaluation validates all frozen records before test inference, verifies
the best-checkpoint bytes/kind/epoch, and reruns validation inference to prove
that the current checkpoint still produces the authorized validation digest.
Only then may the test split be inferred. A dry run stops after authorization,
so it cannot accidentally perform test inference. Calibration and threshold
records from another seed or checkpoint cannot be mixed into an evaluation.

Reports include failure prevalence, failure ROC AUC and AUPRC, success AUPRC,
NLL, Brier score, balanced accuracy, precision, recall, F1, specificity, MCC,
confusion counts, reliability bins, and ECE. Single-class slices are explicitly
marked undefined rather than represented by NaN. Reporting-only slices cover
candidate type, corruption family and severity, anchor reason when a validated
sidecar is available, source trajectory, and split.

Candidate-group reports exclude source candidates where specified and include
corrupted-only top-1 success, random-choice expectation, pairwise
success-over-failure concordance, top-1 failure, mean reciprocal rank, eligible
group counts, and a source-margin diagnostic. Coverage-risk reports use fixed,
deterministic coverage levels and state their actual retained coverage.
Trajectory-level paired bootstrap comparisons preserve all samples and groups
from each resampled source trajectory.

M3B acceptance is assessed for each architecture from the arithmetic mean of
its five calibrated test reports, never by selecting a favorable test seed.
Candidate-level AUPRC, ROC AUC, Brier score, and calibration constraints and the
group-level concordance constraint retain their distinct semantics. The
failure-prevalence Brier reference is computed from the training-fitted constant
predictor. Separately, a positive joint state-action claim requires the selected
validation aggregate to exceed both state-only and action-only validation
aggregates. A failed target is preserved as a negative result without test-set
retuning.

## Checkpoints and artifacts

Trusted locally generated PyTorch checkpoints bind the dataset, split,
preprocessing, model, training configuration, seed, and run-manifest identities;
incompatible reloads fail before state is applied. Checkpoints also store
optimizer, scheduler, epoch/step, best validation score, and random-number state
plus the early-stopping patience cursor for exact interruption resume. Resume is
allowed only from a matching periodic checkpoint with a consistent persisted
history and best-checkpoint cursor. PyTorch checkpoints are pickle-based and
must never be loaded from an untrusted source.

Completed training directories are immutable. Their manifest, scalar summary,
completion marker, checkpoint content digests, validation prediction, class
weight, resolved configurations, and environment bindings are revalidated
before reuse. Post-selection calibration and test artifacts live in a separate
evaluation tree. An explicit rerun receives a new numbered root rather than
mutating the completed evidence it supersedes.

Checkpoints, optimizer states, datasets, vectors, actions, and raw predictions
remain outside Git. Compact sanitized metrics, manifests, prediction tables,
selection records, and summaries may be reviewed and committed.

## Scope limitation

M3B is a single-task experiment using privileged structured state previously
collected and verified by M3A and one fixed continuation policy. Training and
evaluation perform no simulator rollout. Results do not establish visual or
language understanding, cross-task transfer, open-world safety, causal failure
diagnosis, or real-robot validity. LangMani and VLM integration are intentionally
deferred until this small controlled baseline and its leakage-resistant
evaluation are understood.

## M3C deployment-evaluation boundary

M3C loads only the fixed action-only, state+action MLP, and temporal five-seed
families. Each selector averages the five validation-calibrated failure
probabilities. Checkpoint bytes, model and training configuration,
preprocessing, calibration, accepted M3A dataset, and split are all verified by
content digest before inference. Runtime checkpoint paths do not participate in
bundle identity.

The M3C model call receives one verified 38-component state, candidate actions
with shape `[K, 16, 8]`, and masks with shape `[K, 16]`. It cannot receive
candidate distribution, corruption family, severity, source/corrupted marker,
trajectory identity, anchor reason, evidence, or outcome. Stable candidate IDs
break score ties outside the model.

Ensemble abstention thresholds are newly derived from M3B validation ensemble
scores because the existing thresholds are per seed. This is a frozen
validation-only deployment transformation, not model or architecture
reselection. M3C full outcomes neither fit nor change calibration, thresholds,
preprocessing, checkpoints, candidate definitions, or primary-selector status.
Balanced-accuracy and failure-recall operating points use every corrupted M3B
validation candidate; coverage operating points use the validation per-group
minimum ensemble scores that match selection-time application. This distinction
is required when all validation group minima happen to share one outcome class.

The deployment inventory is exactly three learned bundle families, each with
seeds 0 through 4. Stage A evaluates those action-only, state+action MLP, and
temporal ensembles alongside deterministic random, frozen action magnitude, and
six frozen temporal abstention/coverage variants, yielding exactly eleven blind
selectors. Oracle is intentionally excluded and can only be constructed after
complete Stage C outcomes exist.

Runtime artifact trust is anchored to the committed M3B compact benchmark
summary. For every architecture/seed, the calibration and threshold file must
match the outer strict-report digest recorded there; changing a temperature or
threshold and recomputing a locally self-consistent report is rejected. The
complete ordered validation logits/labels must also reproduce the checkpoint's
validation-prediction digest before corrupted rows are used for the five-seed
ensemble abstention policies. A preparation summary binds the resulting policy
report and three bundle reports before Stage A.

Inference profiling is evidence about deployment cost, not a model input or
selection feature. CPU and GPU runs record per-candidate and per-group
p50/p95/p99 latency, throughput, peak allocated device memory, bundle/model
loading time, and an explicit joint-MLP versus temporal comparison. Raw
predictions remain outside compact reports.

Checkpoint preparation also performs a fixed pre-collection forward smoke on
the selected device. It uses only a synthetic 38-vector and eight synthetic
`[16, 8]` chunks, records content digests rather than raw outputs, and explicitly
proves that no M3C source trajectory or candidate outcome was loaded. CPU and
GPU gates can thus finish before simulator collection begins.

M3C preserves the M3B limitations: the verifier sees privileged structured
PickCube state and ranks one fixed 16-step pool once. The archived continuation
is held constant to isolate selection effects, but it prevents closed-loop
replanning and may dominate late-trajectory outcomes. This is not
receding-horizon control, visual/language verification, or evidence for
cross-task or real-robot deployment. The blind full experiment met its
predeclared structured-state intervention targets, but that result does not
authorize visual/language scope automatically; VLM and LangMani integration
remains deferred to a separately reviewed milestone.

M4A preserves these structured-state outcomes while adding separately
content-bound RGB packets. The 38D verifier state remains privileged teacher
metadata and is not a visual-student input. Visual candidate records reference
the accepted action chunks, masks, labels, and evidence without duplicating or
relabeling them. No visual model is trained in M4A.
