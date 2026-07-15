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

## Evidence pipeline (M2A)

M2A adds an evaluation package without adding a simulator implementation:

```text
validated CorruptionDataset
    -> explicit evaluator registry and typed applicability boundary
    -> ordered single-process runner
    -> incrementally persisted ledger + EvaluationEvidence
    -> optional explicit projection of complete conclusive evidence
    -> existing OutcomeLabel
```

The evaluator receives only a `CorruptedActionProposal`, stable source identity,
an explicit derived seed, and an attempt ordinal. LangMani, ManiSkill, simulator,
and policy-framework native objects cannot enter this core interface. Unknown
registry names fail; CLI strings cannot dynamically import Python modules.

Evidence is not an outcome label. `conclusive` means enough task evidence exists
for an explicit projection. `indeterminate` means evaluation completed without
enough evidence. `invalid` and `skipped` describe evaluation applicability, and
`execution_error` describes infrastructure failure. None of the latter four is
silently converted to `success=False`.

The runner derives seeds and evidence IDs from canonical JSON and SHA-256,
orders proposals by M1 generation ordinal, and persists `running` before calling
an evaluator. Terminal evidence and ledger state are then published together by
an atomic manifest replacement. Resume preserves terminal attempts, recovers an
interrupted `running` attempt with its original identity, and adds a new ordinal
only when execution-error retry is explicitly requested. The run binds a digest
of the complete corruption bundle, not only its M0 source ID.

## Generic exact-state paired replay (M2B-Core)

M2B-Core adds a simulator-independent replay layer and retains M2A as the only
runner and ledger:

```text
validated M0 source + validated M1 corruption bundle
    -> complete path-independent content binding
    -> deterministic ReplayCase and JSON-only ReplayBundle
    -> independent baseline and corrupted sessions from one state reference
    -> validated PairedReplayResult
    -> M2A EvaluationEvidence, persistence, resume, and retry
```

The original source action is a mandatory validity gate. A returned state
mismatch or incomplete/unsuccessful baseline task evidence is invalid context;
a step or other runtime exception is an execution error. Only after a complete
successful baseline does the core run the corrupted action from the same state.
A complete corrupted task failure is therefore valid conclusive evidence, not
an infrastructure error.

`ReplayStateReference` stores only an adapter-owned state key or index and its
expected digest. Raw simulator state and framework-native objects stay behind
the narrow adapter/session protocols. Replay-case and bundle identities bind
source and corruption content digests plus semantic metadata, never runtime
paths, host information, or timestamps.

The built-in deterministic replay fixture is weak, non-physical infrastructure
evidence only. M2C may add an explicitly trusted ManiSkill reference adapter
without changing the core contracts. LangMani integration is deferred until a
later typed adapter can satisfy the same exact-state, baseline, and trust gates.

## Robot Episode Toolkit (RET-1)

RET-1 adds a read-only layer over validated M0 Episodes. Audit findings remain
separate from schema validation, exact-index offline iteration remains separate
from M2B state-restoring replay, and all outcome aggregates use candidate-level
denominators. Stable JSON/JSONL reports reuse the repository's protected-output
and deterministic-serialization principles without changing the Episode schema
or bundle format. Future LangMani conversion belongs behind a typed adapter.

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
failures are preserved as operational logs and fixed through a new local
revision. M1 is developed and validated entirely in the authoritative local
checkout; it requires neither AutoDL access nor any other SSH connection. M2A
is likewise developed and validated locally, on CPU and without network, SSH,
simulator, LangMani, ManiSkill, or GPU access. M2B-Core follows the same local
boundary: its fixture executes only a deterministic numeric state machine and
does not contact a remote machine or claim physical validation.

## ManiSkill PickCube reference adapter (M2C)

M2C adds an optional outer integration package; no ManiSkill-native value enters
M0, M1, M2A, or M2B-Core. Availability checks and runtime imports are lazy, so
the core dependency graph remains unchanged and the default suite stays
CPU-only and network-free. The integration maps strict compatibility and state
archives into standard
`ReplayStateReference`, `ReplayCase`, `ReplayBundle`,
`ReplayEnvironmentSession`, and `TerminalTaskEvidence` values.

The package separates semantic configuration from runtime paths. Compatibility,
solver/task source, controller/action layout, state semantic, task semantic,
runtime state-comparison semantic and tolerance, and narrow progress/unsafe
semantics participate in deterministic identity.
Archive roots, dataset directories, hostnames, device allocation, and timestamps
do not. A project-owned recording proxy is the only source-action capture
boundary; a separate fresh environment performs mandatory source baseline
validation before M0 import.

The remote simulator is used only after an exact local revision has been pushed
and synchronized. Its state archives and trajectories stay outside the checkout;
only sanitized compact reports return to the local authority.

The post-discovery expected contract locally binds mplib, SAPIEN, observation
mode, the controller/action contract, action dimension, source digests, state
structure, and compatibility identity to the reviewed schema-1.1 report. The
adapter still fails closed for trusted collection or replay until that local
binding is committed, pushed, and matched by a second trusted remote probe; the
local fake tests remain structural infrastructure checks, not remote simulator
acceptance.
