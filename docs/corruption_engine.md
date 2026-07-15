# Corruption engine

## Semantic boundary

Milestone M1 converts an existing M0 `ActionChunk` into one or more action
proposals for later evaluation. A transformation can make an action different,
but difference is not evidence of failure. M1 therefore does not assign
success, safety, progress, failure events, label strength, or simulator-replay
status, and `CorruptedActionProposal` contains no `OutcomeLabel`.

M2 may attach an outcome only after simulator replay, deterministic evaluation,
a trusted-oracle evaluation, or explicit human review supplies evidence. Until
then, every M1 result is explicitly unlabeled.

## Explicit action semantics

An `ActionLayout` declares a positive `action_dim` and zero or more immutable
`ActionField` values. Each field has:

- a unique name;
- one or more explicit indices in `[0, action_dim)`;
- an `ActionSemantic`: `translation`, `rotation`, `gripper`, `auxiliary`, or
  `unspecified`;
- optional units and descriptive metadata.

Indices cannot overlap across fields, while dimensions not assigned to a field
are valid. A configuration can target named fields or explicit indices, but not
both ambiguously. The engine never assumes that actions are seven-dimensional,
that translation is at indices 0-2, that rotation uses Euler angles, or that a
gripper is the final dimension.

The checked-in [M1 smoke configuration](../configs/corruptions/m1-smoke.json)
shows the complete serializable layout used by the synthetic workflow. Layout
and configuration validation is strict: unknown fields, unknown corruption
names, duplicate or out-of-range indices, empty target selections, invalid
ranges, and NaN or infinity are errors. Inputs are never silently clipped,
normalized, reshaped, repaired, or retargeted.

## Transformations

M1 registers six single-source transformations under stable names:

| Name | Operation and required behavior |
| --- | --- |
| `additive_gaussian_noise` | Adds seeded Gaussian noise with finite mean and positive finite standard deviation to named fields or explicit indices. It applies only to floating-point actions, preserves shape and dtype, and does not clip. |
| `constant_bias` | Adds a finite scalar or dimension-compatible bias to the selected dimensions. Integer outputs are accepted only when every result is exactly representable; the transform never rounds or clips. |
| `temporal_field_shift` | Shifts one selected field by a nonzero signed number of control steps using explicit `edge` or `zero` fill. A positive shift delays values and fills the beginning; a negative shift advances values and fills the end. A magnitude at least as large as the horizon is not applicable. |
| `segment_hold` | On the half-open interval `[start_step, end_step)`, replaces selected dimensions with their value immediately before `start_step`. `end_step` may default to the horizon, and `start_step` must allow a preceding value. |
| `segment_zeroing` | Sets selected dimensions to zero on a validated, non-empty half-open interval `[start_step, end_step)`. |
| `local_temporal_permutation` | Uses an explicit seed to permute selected values within a validated half-open interval while preserving their multiset. A segment shorter than two steps is not applicable. |

Segment transformations may explicitly select fields or all dimensions. In
every operation, unselected dimensions remain unchanged, the horizon and
action dimension remain unchanged, the output dtype is preserved, and the
returned array is detached from the source.

The registry rejects unknown names and duplicate registrations. Each
transformation validates its serializable configuration independently, checks
applicability against the concrete action and layout, and raises a descriptive
`CorruptionApplicabilityError` when it cannot apply.

## Generation, ordering, and identity

The generation engine consumes one validated M0 `Episode`, selects candidates
without modifying it, and visits candidates and configured corruptions in a
stable order. For each applicable pair it records:

- source episode, candidate, policy, task, and split-group identifiers;
- corruption name and complete resolved parameters;
- explicit resolved seed and generation ordinal;
- schema version and optional notes;
- a detached transformed `ActionChunk`.

Each proposal has exactly one source candidate and retains its source split
group. Cross-episode swaps, donor splices, and all other cross-source
transformations are prohibited in M1 because the schema has no complete
multi-parent provenance representation.

For a stochastic transform, the generator derives its seed from canonical
UTF-8 JSON containing the base seed, source episode ID, source candidate ID,
corruption name, and corruption-configuration ordinal. It takes the first eight
bytes of the SHA-256 digest as a big-endian integer and masks it to the
non-negative 63-bit range. A skip therefore cannot consume shared RNG state or
change later proposal randomness.

Proposal identity never uses Python's process-randomized `hash()`. The
generator canonicalizes the source episode ID, source candidate ID, corruption
name, fully resolved parameters, derived seed, and generation ordinal as JSON
with sorted keys, compact separators, and no NaN, encodes it as UTF-8, and
hashes those bytes with SHA-256. The proposal-ID format is
`cap-sha256-<64 lowercase hexadecimal characters>`. Identical stable inputs
produce identical IDs and ordering across runs and supported Python versions;
any identity-bearing input change produces a different digest.

## Applicability and skips

Invalid configuration is always an error. Non-applicability is different: a
valid transformation may not fit a particular candidate horizon or declared
layout. The engine reports a structured deterministic skip containing the
corruption name, source candidate ID, reason, and relevant action/layout
information. It never chooses substitute dimensions or changes parameters.

Normal CLI mode counts and reports skips while continuing in deterministic
order. `--strict-applicability` converts the first skip into a command failure.
In both modes, already examined sources and output ordering are unambiguous.

## Corruption-dataset format

The output uses format marker `latentguard-corruption-dataset`, serialization
version `1`, and dataset schema version `1.0`. `manifest.json` is human-readable
and stores a path-independent digest of the validated source M0 bundle contents,
complete action layout, ordered proposal provenance and resolved parameters,
array references, dtype, shape, and counts. Only transformed action arrays are stored as
non-pickle `arrays/000000.npy` files; M0 observations, RGB frames, and source
episodes are not duplicated.

The destination must be absent or an empty real directory, and the resolved
input and output paths must not equal or contain one another. Saving writes
exclusive files in a sibling staging directory and publishes only a complete
dataset. Loading checks exact fields, versions, paths, inventory, dtype, shape,
and every reconstructed proposal. It recomputes each proposal ID and validates
the corruption name and fully resolved parameters against the six M1 built-ins.
Unsupported versions, unsafe links, missing or extra arrays, malformed
proposals, and incomplete outputs fail clearly. Zero proposals are represented
explicitly.

## Local workflow

From an installed development environment:

```text
latentguard sanity-data --seed 42 --output-dir .tmp/m1-source --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 2
latentguard corrupt-data --input-dir .tmp/m1-source --output-dir .tmp/m1-corrupted --config configs/corruptions/m1-smoke.json --seed 314159
```

`corrupt-data` validates the M0 source and corruption configuration, generates
ordered unlabeled proposals, saves transactionally, reloads the result, checks
identifiers and round-trip properties, and prints source, proposal, skip, and
per-corruption counts. A non-applicable combination can be reported as a skip
or made fatal with `--strict-applicability`.

This M1 workflow is CPU-only and local-authoritative. It performs no GPU
training, simulator execution, SSH synchronization, or AutoDL access.

## M3A fixed-window extension

Schema-1.1 corruption definitions may declare `window_start` and `window_end`.
The transformation applies only inside that half-open step interval and records
the resolved window in proposal identity. M3A fixes the interval to `[0,16)` and
checks the complete source suffix from step 16 byte-for-byte after every
transformation. Gaussian noise, constant bias, temporal gripper shift, segment
hold, segment zeroing, and local temporal permutation support this contract.
Schema-1.0 configurations retain their original whole-chunk behavior.

The checked-in PickCube M3A schedule contains eight definitions across those
families with explicit mild, moderate, and severe identifiers. Values are never
clipped into the action contract: an out-of-bounds result is invalid context at
replay, not a task failure and not a silently repaired sample.
