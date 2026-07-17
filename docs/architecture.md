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

## State-indexed action-verifier data (M3A-Data)

M3A extends only the trusted PickCube outer integration and introduces a compact
simulator-independent training-data package:

```text
official successful action sequence + complete s[0:T+1] archive
    -> deterministic public-evidence anchors with H=16
    -> independent successful source-remainder baselines
    -> window-only M1 corruptions + byte-identical source continuation
    -> independent exact-state paired replay through the M2A ledger
    -> strong conclusive evidence foreign keys
    -> trajectory-disjoint compact ActionVerifierDatasetV1
```

The T+1 archive preserves complete state-tree bytes, structure, and component
inventories outside Git. It separately binds the uninterrupted-trajectory task
snapshot used for event anchors and the fresh-restored public task projection
used by replay. `PickCubeVerifierStateV1` is extracted from public named
robot/task interfaces only after a complete fresh-session set/get restoration
has verified, before any action. Its explicit float32 schema includes the
extraction boundary and every declared task component; it never substitutes a
source-time contact flag or includes the future outcome. Each anchor M0 episode
contains the full remaining source action sequence for replay, while the
exported model input contains only the first 16 actions.

The source and corrupted sessions restore the same indexed state independently,
execute the same remaining horizon, and differ only inside the declared
candidate window. Binary official task completion is captured before the
candidate, immediately after it, and at the terminal state. Fractional progress
is omitted because the accepted compatibility probe does not bind ManiSkill's
normalized dense-reward implementation. Training remains prohibited until the
final dataset reload, evidence-reference, class-count, and leakage gates pass.

## Direct Action Verifier training (M3B)

The optional `latentguard.training` layer consumes only independently validated
M3A exports. Its dataset projection forms a hard boundary between deployable
numeric inputs and reporting metadata. Training modules do not import ManiSkill,
and core package and CLI imports remain usable without PyTorch.

The layer separates strict configuration, accepted-dataset projection,
training-only preprocessing, tensor batching, four small model families,
non-learned baselines, losses, metrics, calibration and thresholds,
checkpoint/resume, evaluation, multi-seed selection, statistical comparison,
run identity, and sanitized reporting. Run identity is content-based and omits
runtime paths, hostnames, process identifiers, and timestamps.

Validation artifacts flow in one direction: training data fits preprocessing and
class weights; validation data controls early stopping, checkpoint/architecture
selection, temperature, and thresholds; an immutable selection record unlocks
test evaluation. Reporting metadata can be joined by internal row index only
after inference and cannot enter a model call.

The unlock is content-bound rather than nominal. Every validation seed record
identifies the exact best-checkpoint bytes, epoch, resolved configurations, and
validation-prediction digest. Calibration and thresholds bind that same digest;
test evaluation first validates the complete frozen selection and recomputes the
authorized validation predictions. Training-run completion and benchmark
orchestration are independently digested, completed directories are immutable,
and post-selection evaluation artifacts live outside them.

## Blind candidate selection (M3C)

M3C adds a simulator-independent selection layer above the accepted M3B models
and below the existing replay boundary:

```text
verified state + immutable eight-candidate pool + frozen verifier bundles
    -> label-free five-seed inference and deterministic rankings
    -> finalized blind selection manifest
    -> selected-union replay through the existing M2A/M2B/M2C ledger
    -> complementary full-pool replay
    -> content-bound metrics, oracle analysis, and trajectory bootstrap
```

Candidate generation, selection, and outcome evaluation are different
capabilities and output roots. Selection cannot receive replay or evidence
paths. Reporting-only candidate metadata does not cross the inference boundary.
The source action is absent from the candidate inventory, while the unchanged
source continuation and baseline gate retain their exact M3A/M2C identities.

Selected and remainder phases wrap the existing evaluator with complementary
content-bound executable inventories; they do not introduce a second ledger or
change proposal identity. Only the wrapper evidence identity changes to bind
the phase, pool, and finalized manifest. Complete-pool labels are joined after
selection and cannot mutate the earlier ranking.

Stage A is a capability-restricted, outcome-declassified process. It can load
the immutable candidate pool, but its API has no evidence, outcome, or replay
path and its selector projection contains only state vectors, action chunks,
masks, and opaque candidate IDs. The guarantee is about capabilities and
content-bound inputs; it does not assert that every other filesystem location
on the host is empty. Full-pool replay, evidence, and outcome artifacts enter
the workflow only through Stage B and Stage C.

The Stage A selector inventory is closed and versioned: deterministic random,
frozen action magnitude, action-only ensemble, state+action MLP ensemble,
temporal ensemble, and six temporal abstention/coverage variants. The learned
selectors reference exactly three five-seed bundles. The post-Stage-C oracle is
deliberately outside this eleven-selector inventory.

The trust chain begins at the committed M3B compact benchmark summary. Its
recorded outer strict-report digests authorize each runtime calibration and
threshold report; inner calibration/threshold bindings alone are insufficient.
Ordered validation logits and labels are rehashed against every seed's frozen
validation-prediction identity before corrupted-candidate rows are used to fit
ensemble policies. A preparation summary then binds the bundle, baseline,
policy-report, selector-configuration, and source-summary identities.

Smoke and full source sets use distinct fixed seed windows and are checked for
disjoint reset seeds, trajectory IDs, split-group IDs, and complete state-tree
digests. Full construction loads the smoke pool as an exclusion artifact. A
post-smoke configuration change therefore requires a new pushed revision and a
new untouched full source range.

Operational time is audit evidence, not semantic identity. Nevertheless, final
binding requires the persisted order
`selection < selected start <= selected finish < remainder start <= remainder
finish`; the final outcome timestamp is the recorded remainder finish. CPU and
GPU profiling produces compact p50/p95/p99, throughput, peak-memory, and model
loading summaries while keeping raw predictions outside Git.

The joined replay therefore has two digests. Its semantic evidence digest
excludes envelope timestamps, host/platform data, launch paths, evidence IDs,
and other ledger mechanics while retaining complete task and strong-restoration
content. Its archive-audit digest separately binds the timestamped manifest
envelope and byte-complete selected/remainder dataset digests. Stage A itself is
published by an atomic directory rename only after the manifest, selector
configuration, and internal latency report all reload successfully.

This architecture chooses one 16-step prefix once and then executes the
byte-identical archived source continuation. It has no observation/update loop,
policy recall, or replanning boundary, so it is not receding-horizon control.
Visual/language adapters remain deferred until the structured-state selector
shows outcome-level intervention value under the blind protocol.

## M4A visual-data boundary

The simulator-independent `latentguard.vision_data` package owns camera/domain
configuration, immutable packet and dataset models, canonical identities, safe
NPY publication, source bindings, split/cross-dataset leakage checks, and the
render ledger. It never imports ManiSkill, SAPIEN, CUDA, or Vulkan.

ManiSkill-specific environment creation, exact restoration, public verifier
capture, world-camera configuration, sensor updates, RGB extraction, and visual
compatibility probing live under
`latentguard.integrations.maniskill_pickcube`. Simulator-native objects do not
cross that boundary. A render session exposes images and project-owned audit
records only after the 70-component state, 38-component verifier, task
projection, and no-step gates pass.

The compatibility report is hardware- and source-evidence-bound: it records the
exact archived episode/state used for the probe together with observed renderer
and calibration dtypes. Core packet records retain enough exact verifier,
elapsed-step, close, and calibration evidence to build compact integrity reports
without reopening simulator-native objects.

Visual packets share images across candidate references. The M3A development
binding extends the accepted trajectory split; the independent M3C external
binding is evaluation-only and cannot flow into training loaders.
