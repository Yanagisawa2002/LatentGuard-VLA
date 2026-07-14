# Architecture

## Design boundary

LatentGuard-VLA's core describes observations, proposed action chunks,
outcomes, failures, and provenance without depending on a robot, simulator,
policy framework, visual encoder, or neural-network library. Future LangMani,
ManiSkill, LeRobot, simulator, storage, and policy integrations belong behind
explicit typed adapters. Integration-native objects must not leak into the
core contract.

## Foundation components (M0)

The package follows a `src` layout and separates four responsibilities:

- **Models and validation** define immutable or mutation-safe, versioned
  entities and reject malformed input with contextual errors.
- **Synthetic fixtures** deterministically construct small CPU-only episodes
  from an explicit seed and configuration.
- **Serialization** writes a human-readable manifest and safe binary arrays,
  then validates every loaded episode. It does not deserialize executable
  Python objects.
- **CLI and remote synchronization** expose local sanity checks and construct a
  non-destructive exact-revision SSH operation. SSH execution is isolated so
  tests can replace it with a mock.

Dependencies flow inward toward the data contract. The models do not import
the CLI, remote execution, a simulator, or a training system. The synthetic and
serialization modules consume models; the CLI orchestrates these modules.

## Corruption pipeline (M1)

M1 adds a simulator-independent corruption package around the M0
`ActionChunk` contract. Its dependencies flow in one direction:

```text
source Episode + explicit ActionLayout + typed corruption configuration
    -> deterministic generation and applicability handling
    -> ordered CorruptedActionProposal values
    -> transactional corruption-dataset serialization
```

The action layout declares the total dimension and named fields with explicit,
disjoint indices and semantic types. Unused dimensions are valid. Nothing
infers translation, rotation, gripper position, units, or representation from
the action dimension. This keeps arbitrary positive dimensions and non-Euler
or robot-specific conventions behind explicit data rather than assumptions.

Each transform validates its configuration and applicability before applying
it. A non-applicable transform cannot choose another field, index set, or
parameter value: generation records a structured skip, or strict mode turns
the skip into an error. Transformations preserve shape and dtype and never
silently clip, normalize, reshape, or repair their input.

Generation is single-source. An ordered proposal identifies exactly one source
episode and candidate, the source policy, task and split group, the corruption
name, fully resolved parameters, seed, and generation ordinal. A canonical
encoding of those stable inputs is hashed with SHA-256; Python's randomized
`hash()` is never part of identity or ordering. Cross-episode swaps, donor
splices, and every other multi-source operation remain outside M1 because the
schema cannot yet describe multiple parents completely.

Stochastic transforms do not share mutable random state. Each proposal seed is
derived with SHA-256 from the base seed and stable source/configuration
identity, so it does not depend on process hash randomization or skipped items.

Corruption produces hypotheses for evaluation, not outcome evidence.
`CorruptedActionProposal` therefore has no `OutcomeLabel`, success, safety,
progress, failure-event, or simulator-verification field. M2 may attach an
outcome only after simulator replay, deterministic evaluation, a trusted
oracle, or explicit human review provides evidence.

The corruption serializer references the M0 source dataset and writes only the
layout, proposal metadata, and transformed action arrays. It deliberately does
not duplicate RGB observations or source episodes. As with M0, a readable
manifest and non-pickle arrays are staged and published transactionally, then
validated on reload.

## Safety and reproducibility

All random generation takes an explicit seed. Derived examples retain their
source episode, split group, and transformation metadata. Evaluated derivatives
also retain label strength, label source, and simulator-replay state; unlabeled
M1 proposals do not fabricate those fields. Future dataset splitting must
operate at the source-episode group boundary so a source episode and all
derivatives cannot cross splits; M0 records the grouping metadata but does not
split data.

M0 deliberately includes no model training. Later training entry points must
add dry-run, bounded-step and bounded-sample modes, explicit output and seed,
checkpoint save/resume, periodic evaluation, interruption handling, resolved
configuration and run manifests, and peak GPU-memory reporting.

## Authority boundary

All tracked changes are made, checked, committed, and pushed locally. A remote
machine is an execution target only. It pulls an exact pushed commit, writes
large outputs outside the checkout, and never edits tracked source. Remote
failures are preserved as evidence and fixed through a new local revision.
M1 is developed and validated entirely in the authoritative local checkout; it
requires neither AutoDL access nor any other SSH connection.
