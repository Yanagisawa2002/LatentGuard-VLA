# Data contract

## Purpose

The M0 contract represents action-conditioned robot-policy evidence without
choosing a robot, simulator, task suite, camera layout, action dimension, or
image resolution. Every top-level entity carries explicit schema-version
metadata. The current supported version is documented by the package constant;
unknown versions are rejected rather than guessed or migrated silently.

M1 adds a separate contract for proposed action corruptions. These values are
inputs to later evaluation, not evidence that an action succeeds, fails, is
unsafe, makes no progress, or has been replayed in a simulator.

M2A adds evaluation evidence, run-ledger, and resume contracts. Evidence is a
record of what an evaluator established; it remains distinct from the M0
`OutcomeLabel` that may be derived only from complete conclusive evidence.

M2B-Core adds content-bound replay references, cases, restoration/execution/task
evidence, adapter trust, paired results, and a replay bundle. These are generic
contracts; no simulator-native state or framework object is serializable here.

M3A-Data adds compact, model-ready action-verifier samples and candidate groups.
Their labels are foreign keys into strong, simulator-verified paired-replay
evidence; full simulator state trees remain in an external adapter archive.

RET-1 adds no schema entity and does not change the serialization version. Its
audit reports, offline replay plans/steps, and metrics are derived read-only
views of already validated Episodes. Candidate outcome aggregates always use a
candidate denominator; no Episode-level success is inferred.

## Entities

- `CameraFrame` contains RGB data and optional aligned depth, intrinsics, and
  extrinsics. An observation may contain zero, one, or many named cameras.
- `ObservationFrame` contains a non-negative timestamp, an arbitrary-length
  finite rank-1 robot-state vector, and camera frames.
- `ObservationHistory` is a strictly ordered sequence of observations.
- `ActionChunk` contains a finite `[horizon, action_dim]` sequence, a positive
  control period, and coordinate-frame metadata.
- `CandidateAction` assigns a unique candidate identifier to an action chunk
  and its provenance.
- `OutcomeLabel` records success, normalized progress, unsafe status, optional
  failure type, label source and strength, and verification state.
- `FailureEvent` records a typed failure at a timestamp with optional details.
- `SampleProvenance` records source episode, policy, task, split group,
  transformation or corruption metadata, seed, label provenance, and simulator
  replay status.
- `Episode` combines instruction/task/policy metadata, observation history,
  candidates, labels, failures, and source identity.

M1 adds the following entities without changing the meaning of M0 labels:

- `ActionLayout` declares a positive action dimension and named action fields.
  Each field has one or more explicit, in-range, non-overlapping indices, a
  semantic type (`translation`, `rotation`, `gripper`, `auxiliary`, or
  `unspecified`), and optional units and descriptive metadata. Unused
  dimensions are allowed; semantics are never inferred from dimension count.
- `CorruptedActionProposal` stores a deterministic proposal identifier, its
  single source episode and candidate, source policy, source task, split group,
  detached transformed `ActionChunk`, corruption name, fully resolved
  parameters, seed, generation ordinal, schema version, and optional notes.
  It intentionally contains no `OutcomeLabel`.
- The corruption dataset references an M0 source dataset and combines one
  action layout with ordered proposal records and their transformed arrays. It
  does not copy observations or RGB data.

M2A adds:

- `EvaluationEvidence`, with stable proposal/source/evaluator identity,
  configuration digest, seed, attempt ordinal, explicit status, optional task
  values, failure events, termination information, scalar diagnostics, relative
  artifact references, label provenance, and schema version.
- `LedgerEntry`, which records pending, running, completed, indeterminate,
  invalid, skipped, or execution-error attempt state plus sanitized operational
  errors, retry eligibility, and audit timestamps.
- An evaluation dataset, which binds the complete source corruption-bundle
  digest, exact ordered proposal selection, evaluator and resolved
  configuration, evidence, ledger, summary, and run environment metadata.

M2B-Core adds:

- `ReplayStateReference`, an opaque adapter/version/source reference plus one
  stable state key or index, expected state digest, comparison semantic, and
  canonical JSON metadata. Adapter-specific comparison semantic and tolerance
  bindings live in this canonical identity metadata when required. It never
  contains raw state, arrays, or paths.
- `ReplayTaskReference`, a task ID and contract version with uninterpreted
  canonical JSON metadata.
- `ReplayCase`, which binds one M1 proposal to the matching M0 original action,
  transformed action, source identities, both dataset content digests, state
  and task references, adapter identity, and progress/unsafe semantics.
- `StateRestorationEvidence`, `ActionExecutionEvidence`, and
  `TerminalTaskEvidence`, which retain explicit completeness and missingness
  without storing raw state or inferring task values.
- `ReplayTrustDescriptor`, which caps label source, label strength, and
  simulator-verification permission for an adapter.
- `PairedReplayResult`, the validated result of the baseline and corrupted
  sessions, and `ReplayBundle`, an ordered JSON-only case-reference manifest.

M3A-Data adds:

- `ActionVerifierSampleV1`, containing one verified compact state vector, one
  fixed `[16, action_dim]` candidate chunk, an all-true action mask, task and
  continuation identity, raw strong outcome fields, and content-bound
  provenance. The continuation and raw simulator state are not model inputs.
- `ActionVerifierCandidateGroupV1`, grouping one successful source chunk and
  all accepted conclusive corruptions for one exact state, task, and
  continuation. It does not invent pairwise preferences.
- `TrajectorySplitAssignmentV1`, recording leakage-sensitive state, anchor,
  proposal, source-seed, and split-group inventories for one source trajectory.
- `ActionVerifierDatasetV1`, binding deterministic samples, groups, split
  assignments, state-vector schema, dimensions, and a content digest.

Evidence status has exact semantics: `conclusive` is projectable when complete;
`indeterminate` is a completed but insufficient evaluation; `invalid` is an
unevaluable proposal/context; `skipped` is an explicit applicability/policy
decision; and `execution_error` is evaluator infrastructure failure. Invalid,
skipped, and execution-error records are not failed robot actions.

Heuristic labels are weak evidence. A simulator-sourced strong label is valid
only when simulator replay is explicitly verified. M0 synthetic labels remain
clearly identified as synthetic evidence; they do not claim simulator replay.
Likewise, the fact that M1 applied a heuristic corruption is never represented
as a weak or strong outcome label and never as simulator verification. M2 may
attach an outcome only when an evaluator supplies the corresponding evidence.

## Validation contract

Validation is strict and non-repairing. It rejects unsupported schema versions;
negative, repeated, inconsistent, or non-monotonic timestamps; malformed RGB
or depth arrays; RGB/depth resolution mismatches; non-finite numeric values;
invalid camera matrices; malformed or empty action sequences; non-positive
control periods; candidate horizon inconsistencies; invalid normalized ranges
or probabilities; duplicate candidate IDs; missing source/split provenance;
unsupported label source/strength combinations; invalid strong simulator
labels; and inconsistent episode identities.

Errors name the entity, field, offending episode/sample/candidate when known,
and a concise reason. Loaders never clip, normalize, reshape, drop, or otherwise
silently repair invalid data.

Corruption configuration is strict as well: unknown types or fields, ambiguous
field/index targeting, empty targets, duplicate or out-of-range indices,
invalid ranges, and non-finite values are errors. If a valid transformation
cannot apply to a particular source action/layout pair, generation returns a
descriptive skip without changing its dimensions or parameters; strict
applicability mode promotes any such skip to an error.

M2A evidence validation additionally rejects empty identities, unsupported
versions, forged deterministic IDs, non-finite metrics, normalized progress
outside `[0,1]`, delta outside `[-1,1]`, inconsistent before/after/delta,
negative control-step counts, unsafe artifact paths or URLs, and inconsistent
status/task fields. A verified replay flag is valid only for simulator-sourced
evidence, strong simulator evidence requires verified exact replay, and
heuristic evidence cannot be strong. Missing values are never inferred,
normalized, or filled with defaults.

## Provenance and splits

Every evaluated derived candidate identifies the source episode, source
policy, source task, transformation type and parameters, seed, label source
and strength, simulator-replay status, split-group identifier, and schema
version. An M1 proposal carries the corresponding single-source identity,
transformation, parameter, seed, ordering, split, and schema data but carries no
label provenance because it is unlabeled. A source episode and every derivative
share one split group. Dataset code must split source groups before generating
derivatives.

M3A assigns splits only at original source-trajectory granularity. Source
trajectory IDs, source seeds, state digests, anchor IDs, proposal IDs, and split
group IDs are all checked for cross-split reuse. The full policy is 48 train, 6
validation, and 6 test trajectories; bounded smoke datasets use explicit quotas
that exactly cover their trajectories.

M1 prohibits cross-source transformations. A proposal has exactly one source
episode and candidate until a later schema version can represent complete
multi-parent provenance. Proposal ordering is deterministic. Its identifier has
the format `cap-sha256-<64 lowercase hexadecimal characters>`, where the digest
covers canonical UTF-8 JSON containing the source episode ID, source candidate
ID, corruption name, fully resolved parameters, derived seed, and generation
ordinal. Canonical JSON uses sorted keys, compact separators, and no NaN. The
identifier never uses Python `hash()`.

An M2A evidence ID has the form
`evd-sha256-<64 lowercase hexadecimal characters>`. Its canonical JSON payload
contains only proposal ID, evaluator ID and version, evaluator-configuration
digest, evaluation seed, and attempt ordinal. Absolute paths, output directory,
host, timestamp, process ID, and status/result values are excluded. The run ID
separately binds the path-independent digest of the complete validated M1
bundle, the M0 source ID, evaluator/configuration, base seed, and exact ordered
proposal selection.

`evidence_to_outcome_label` is the only M2A projection boundary. It accepts only
validated conclusive evidence with success, `progress_after`, unsafe status,
label source and strength, and valid failure events. The canonical mapping is
`OutcomeLabel.progress = EvaluationEvidence.progress_after`; delta is never
substituted and missing values are never derived. Projection does not mutate the
evidence.

An M2B replay-case ID is canonical SHA-256 over the stable source/proposal,
action-contract, state/task, adapter, semantic, schema, and complete M0/M1
content-digest inputs. A replay-bundle digest additionally binds the ordered
case identities and canonical metadata. Absolute paths, state blobs, hostnames,
timestamps, process IDs, and Python `hash()` are excluded.

A stochastic proposal seed is derived by hashing canonical UTF-8 JSON
containing the base seed, source episode ID, source candidate ID, corruption
name, and configuration ordinal. The first eight SHA-256 digest bytes are read
as a big-endian integer and masked to the non-negative 63-bit range. This keeps
one proposal's randomness independent of skip behavior and mutable RNG state.

## Serialization

The M0 local format is versioned and round-trip safe. Its readable manifest
stores schema versions, scalar metadata, enum values, array references,
identifiers, and provenance. Arrays use non-pickle binary storage that
preserves dtype and shape. Saving uses a fresh staging directory and publishes
only to an absent or empty destination, so partial writes cannot corrupt an
existing bundle. Loading checks exact manifest fields, versions, array
inventory, dtype and shape, reconstructs typed entities, and validates them
before returning data. This is an interchange foundation, not a
storage-performance design.

The M1 corruption format follows the same safety boundary under a distinct
`latentguard-corruption-dataset` format marker, serialization version `1`, and
dataset schema version `1.0`. `manifest.json` records a path-independent digest
of the validated source M0 bundle contents, complete layout, ordered proposal
provenance, resolved parameters, array dtype and shape, and counts. Transformed
actions are stored as
`arrays/000000.npy` and subsequent non-pickle NPY files; source observations
are referenced rather than duplicated.

Saving accepts only an absent or empty real directory. CLI generation also
requires resolved input and output paths that neither equal nor contain one
another, preserving the validated M0 source bundle. Saving writes exclusive
files in a sibling staging directory and publishes only a complete dataset.
Loading rejects unsupported versions, unknown or duplicate fields, unsafe
links or paths, missing or extra arrays, and any proposal that fails validation.
Zero-proposal datasets are represented explicitly rather than mistaken for a
partial write.

The M2A format is a human-readable, versioned JSON manifest and never duplicates
M1 action arrays. Initial creation stages a complete pending ledger and publishes
only to an absent or empty real directory. Each subsequent ledger transition
writes and flushes a sibling temporary manifest before atomically replacing the
live manifest. A run marked `in_progress` or `interrupted` is loadable but cannot
be mistaken for `complete`.

M2A artifact references are safety-validated relative POSIX metadata only. The
runner does not materialize referenced artifacts, and referenced files are not
part of the evaluation bundle inventory, which contains only `manifest.json`.
Physical artifact creation and inventory validation require a later adapter
contract.

Evidence and ledger entries are ordered by source proposal generation ordinal
and attempt ordinal. Terminal attempts are not rerun on resume. An interrupted
`running` entry is recovered with the same seed and evidence identity. An
`execution_error` is retried only by explicit policy, which appends the next
attempt and preserves the earlier error. Any source digest, proposal selection,
base-seed, evaluator-version, or resolved-configuration conflict rejects resume.

The M2B replay bundle is a strict JSON manifest that references case IDs and
the already validated M0/M1 action arrays rather than duplicating them. Initial
creation is transactional and requires an absent or empty real destination.
Loading rejects links, traversal, unsupported versions, unknown or duplicate
fields and IDs, digest or identity tampering, missing source/proposal references,
and any changed source or corruption content. The caller must supply the current
content binding when reloading; a stored digest is never trusted by itself.

The M3A training dataset is a separate strict JSON/NPY bundle. It stores stacked
non-pickle arrays for state vectors, candidate chunks, and masks, plus compact
labels, identifiers, groups, and split assignments. Saving is transactional and
deterministic. Loading checks exact inventory, safe paths, links, dtype, shape,
content digests, evidence foreign keys, candidate-group coverage, and split
leakage. It never duplicates T+1 state trees or continuation actions.

## M3B learned-input projection

An M3B accepted-dataset loader revalidates the complete M3A bundle, its exact
content digest, split assignments, evidence properties, required full-target
report, and fixed 60-trajectory acceptance facts before exposing training rows.
The resulting model-input schema is an allowlist, not a view of the original
sample dataclass:

```text
state_vector: float32[38]
candidate_action_chunk: float32[16, 8]
action_mask: bool[16]
failure_target: binary scalar (loss and metrics only)
sample_index: internal reporting join key
```

Model `forward` receives only the first three tensors. Every identifier,
provenance field, corruption descriptor, candidate type, split label, evidence
field, source ordering value, and post-execution outcome remains in a separate
reporting record. Missing, additional, non-finite, wrong-dtype, or wrong-shape
model fields fail validation; no silent casting or repair is permitted before
the explicit NumPy-to-PyTorch conversion boundary.

The preprocessing identity binds the accepted dataset digest, exact training
membership digest, state/action component counts, means, standard deviations,
and fixed floor. A run additionally binds the complete split digest, resolved
model and training configuration digests, seed, and code semantic. Filesystem
location and host identity never contribute to semantic identity.

The run manifest also binds the accepted full-target report digest, exact
resolved configurations, training-only positive class weight, source revision,
and sanitized deterministic environment. Checkpoint content has an independent
byte digest. A validation seed result binds that digest, checkpoint kind/epoch,
and the ordered validation label/logit digest. Calibration binds the same
checkpoint, model and preprocessing identities; thresholds bind the same
validation-prediction digest. These links are checked before test inference, so
artifacts from different seeds, epochs, or runs cannot be silently combined.

A completed training-run marker binds its scalar summary, seed result, and best
checkpoint. Benchmark orchestration separately binds the acceptance report,
dataset/split/preprocessing, benchmark/training configuration, Git revision,
mode, and full planned-run inventory. Runtime paths, hostnames, process IDs, and
timestamps remain non-semantic. Completion is immutable; resume may extend only
an incomplete matching run, while an explicit rerun receives a new identity
location.

## M2C integration archive boundary

The ManiSkill PickCube runtime archive is not a new core replay format. It is an
adapter-owned, versioned store that supplies opaque state references to M2B.
Its semantic identity binds compatibility, official solver source, environment
contract, source actions, state-tree structure and bytes, seed, and task
evidence while excluding runtime paths.

State dictionaries use canonical structural paths, preserve mapping/list/tuple
structure, and store numeric leaves as non-pickle NPY files. The loader validates
the complete inventory, dtype, shape, finite data, structure, and digest before
reconstructing a value. Object arrays, traversal, links, arbitrary Python
objects, and silent conversions are rejected. Runtime tensor/device conversion
happens only inside the adapter.

An M2C M0 source episode exists only after an independent fresh-session replay
has restored the recorded state, replayed every official-source action, and
established official task success. Its robot-state vector is bound to recorded
Panda active-joint names and a versioned qpos-then-qvel semantic: N real joint
names bind a length-2N vector of ordered qpos followed by ordered qvel. Subsequent
M1 corruptions remain unlabeled until the standard M2A/M2B pipeline evaluates
them.
