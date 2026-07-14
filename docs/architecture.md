# Architecture

## Design boundary

LatentGuard-VLA's core describes observations, proposed action chunks,
outcomes, failures, and provenance without depending on a robot, simulator,
policy framework, visual encoder, or neural-network library. Future LangMani,
ManiSkill, LeRobot, simulator, storage, and policy integrations belong behind
explicit typed adapters. Integration-native objects must not leak into the
core contract.

## M0 components

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

## Safety and reproducibility

All random generation takes an explicit seed. Derived examples retain their
source episode and split group, label strength and source, transformation
metadata, and simulator-replay state. Future dataset splitting must operate at
the source-episode group boundary so a source episode and all derivatives
cannot cross splits; M0 records the grouping metadata but does not split data.

M0 deliberately includes no model training. Later training entry points must
add dry-run, bounded-step and bounded-sample modes, explicit output and seed,
checkpoint save/resume, periodic evaluation, interruption handling, resolved
configuration and run manifests, and peak GPU-memory reporting.

## Authority boundary

All tracked changes are made, checked, committed, and pushed locally. A remote
machine is an execution target only. It pulls an exact pushed commit, writes
large outputs outside the checkout, and never edits tracked source. Remote
failures are preserved as evidence and fixed through a new local revision.
