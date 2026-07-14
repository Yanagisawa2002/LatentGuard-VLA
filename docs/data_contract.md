# Data contract

## Purpose

The M0 contract represents action-conditioned robot-policy evidence without
choosing a robot, simulator, task suite, camera layout, action dimension, or
image resolution. Every top-level entity carries explicit schema-version
metadata. The current supported version is documented by the package constant;
unknown versions are rejected rather than guessed or migrated silently.

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

Heuristic labels are weak evidence. A simulator-sourced strong label is valid
only when simulator replay is explicitly verified. M0 synthetic labels remain
clearly identified as synthetic evidence; they do not claim simulator replay.

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

## Provenance and splits

Every derived candidate identifies the source episode, source policy, source
task, transformation type and parameters, seed, label source and strength,
simulator-replay status, split-group identifier, and schema version. A source
episode and every derivative share one split group. Dataset code must split
source groups before generating derivatives.

## Serialization

The local format is versioned and round-trip safe. Its readable manifest stores
schema versions, scalar metadata, enum values, array references, identifiers,
and provenance. Arrays use non-pickle binary storage that preserves dtype and
shape. Saving uses a fresh staging directory and publishes only to an absent or
empty destination, so partial writes cannot corrupt an existing bundle. Loading
checks exact manifest fields, versions, array inventory, dtype and shape,
reconstructs typed entities, and validates them before returning data. This is
an interchange foundation, not a storage-performance design.
