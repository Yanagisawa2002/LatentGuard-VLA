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

## Provenance and splits

Every evaluated derived candidate identifies the source episode, source
policy, source task, transformation type and parameters, seed, label source
and strength, simulator-replay status, split-group identifier, and schema
version. An M1 proposal carries the corresponding single-source identity,
transformation, parameter, seed, ordering, split, and schema data but carries no
label provenance because it is unlabeled. A source episode and every derivative
share one split group. Dataset code must split source groups before generating
derivatives.

M1 prohibits cross-source transformations. A proposal has exactly one source
episode and candidate until a later schema version can represent complete
multi-parent provenance. Proposal ordering is deterministic. Its identifier has
the format `cap-sha256-<64 lowercase hexadecimal characters>`, where the digest
covers canonical UTF-8 JSON containing the source episode ID, source candidate
ID, corruption name, fully resolved parameters, derived seed, and generation
ordinal. Canonical JSON uses sorted keys, compact separators, and no NaN. The
identifier never uses Python `hash()`.

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
