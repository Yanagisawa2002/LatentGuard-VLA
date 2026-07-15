# Generic exact-state paired replay

## Core boundary

M2B-Core is a simulator-independent SDK, not a simulator integration. It binds
validated M0 source actions to M1 corrupted proposals, describes opaque
content-bound state and task references, and runs the same paired-execution
algorithm through narrow typed adapter protocols. No LangMani, ManiSkill, CUDA,
GPU, SSH, or simulator-native object is part of the core contract.

M2C may add a real ManiSkill reference adapter behind these protocols. A later
LangMani adapter can use the same contracts without changing replay models,
identities, evidence semantics, or the M2A runner.

## Source binding and identity

Replay source binding validates both datasets and records three independent
values:

- the existing path-independent M0 bundle identifier referenced by M1;
- a canonical logical M0 content digest covering every entity and array value;
- a canonical logical M1 corruption digest covering layout, provenance, and
  every transformed action value.

The resolver does not trust a proposal ID alone because the M1 identifier does
not cover every transformed byte or provenance field. It verifies the bound
proposal, episode, candidate, policy, task, split group, action dtype, shape,
horizon, dimension, control period, coordinate frame, and source/corruption
content snapshots. Original M0 outcome labels are not baseline replay evidence.

A replay case binds the two actions, both content digests, the opaque state and
task references, adapter identity, and explicit progress and unsafe semantics.
Its canonical SHA-256 identity includes action dtype, shape, metadata, and
content hashes. Runtime paths, hosts, clocks, process IDs, and output locations
are excluded.

`ReplayStateReference` stores only stable metadata and an expected digest; raw
simulator state is never embedded. The adapter chooses the comparison semantic
and returns `StateRestorationEvidence`. Core validation requires the expected
digest and semantic to match the case, finite non-negative comparison values, a
positive component count, and a verified match before action execution.

## Paired execution and status mapping

The executor first creates a baseline session, restores and verifies the initial
state, evaluates the initial task state, executes every original action row,
evaluates the terminal task state, and closes the session. The comparison is
valid only when this terminal evidence is complete and successful.

Only then does it create a distinct corrupted session. That session restores
the same frozen state reference, verifies the state again, evaluates the initial
task state, executes every transformed action row, evaluates the terminal task
state, and closes independently. Action rows are detached before entering an
adapter, and both source actions are checked for mutation after execution.

The mapping is deliberately strict:

- static action/context mismatch, state mismatch, or failed/incomplete baseline
  becomes `invalid`;
- an adapter, restoration, step, task-evaluation, or close exception becomes
  `execution_error`;
- complete corrupted evidence with success, progress, and unsafe values becomes
  `conclusive`, including a genuine task failure with `success=false`;
- completed execution with incomplete terminal task evidence becomes
  `indeterminate`, with missing values left missing.

The M2A `replayed_control_steps` value is the sum of baseline and corrupted
steps that returned successfully. `progress_before` comes only from the real
initial corrupted-session task evidence, `progress_after` comes from its
terminal evidence, and no missing value is inferred.

## Trust enforcement

`ReplayTrustDescriptor` separates replay mechanics from adapter authority.
Fixture and deterministic non-simulator tiers cannot emit simulator label
source, strong simulator evidence, or verified simulator replay. An exact
simulator tier merely makes those values eligible: both restorations, the
successful baseline, complete corrupted execution, complete terminal evidence,
and the descriptor's explicit permission must still pass. A verified
restoration uses exact-digest semantics with zero tolerance by default. Only an
exact-simulator descriptor may explicitly bind numeric-tolerance semantics and
a finite positive bound. Both sessions must match that descriptor exactly,
report a complete state comparison, and bind the same expected state digest.
The mode and tolerance are part of the evaluator's resolved configuration and
therefore its identity. An adapter-specific versioned comparison semantic and
tolerance must also be bound into that adapter's semantic configuration and
each replay-case identity; generic numeric-tolerance mode alone is insufficient.

The built-in `deterministic_replay_fixture` is permanently mapped to
`LabelSource.DETERMINISTIC_EVALUATOR`, `LabelStrength.WEAK`, and
`simulator_replay_verified=false`. Its numeric state, target, restoration, and
task checks are synthetic and non-physical. It exists only to test contracts,
serialization, the paired executor, M2A persistence, resume, and retry. It must
not support training, benchmark, or research claims.

## Replay bundle and M2A reuse

`ReplayBundle` is a versioned JSON binding. Its manifest stores ordered case
references and action digests, then reloads original and transformed actions
from separately validated M0/M1 datasets. It does not duplicate action arrays,
raw states, images, videos, checkpoints, or framework objects. Creation is
transactional, loading is strict, and source/corruption/case/bundle tampering is
rejected.

The bundle is constructed in memory for `replay-data`; it is not placed inside
the M2A evaluation output directory, whose inventory remains exactly one
`manifest.json`. The exact-state evaluator implements the existing M2A
`ProposalEvaluator`, so evidence identity, ledger transitions, atomic updates,
interruption recovery, explicit execution-error retry, resume conflict checks,
and outcome projection remain the M2A implementations rather than a second
replay-specific ledger.

## Local fixture workflow

```text
latentguard sanity-data --seed 42 --output-dir .tmp/m2b-source --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 0
latentguard corrupt-data --input-dir .tmp/m2b-source --output-dir .tmp/m2b-corrupted --config configs/corruptions/m1-smoke.json --seed 314159
latentguard replay-data --source-dir .tmp/m2b-source --corruption-dir .tmp/m2b-corrupted --output-dir .tmp/m2b-replayed --adapter deterministic_replay_fixture --config configs/replay/m2b-fixture.json --seed 161803
```

Use `--resume` for an idempotent completed-run check and `--dry-run` with a
separate absent output path to validate source binding, cases, configuration,
bundle identity, and planned M2A evidence IDs without creating a session or
output. This workflow is local, CPU-only, deterministic, and network-free.

## M2C real PickCube adapter

`maniskill_pickcube_v1` implements the same protocols without changing the
generic executor or M2A ledger. Its state reference names one archived official
source trajectory and reset-state key and binds the `maniskill_state_tree_v1`
digest; the archive root itself is runtime-only. The adapter refuses to resolve
a case unless compatibility, source/corruption content, action layout and
bounds, solver identity, task identity, and archive digests agree.

Each session lazily creates a fresh one-environment GPU `PickCube-v1` runtime,
converts safe archive leaves to runtime tensors, calls `set_state_dict`, reads
the complete state back, and emits restoration evidence. The archive reference
remains bound to the exact source-state digest. Live restoration may use the
checked-in numeric tolerance only when the complete structure matches; both
fresh sessions must verify against that same descriptor-bound comparison
contract. Archive, component-inventory, or structural failures leave the typed
complete-comparison flag false. A full finite comparison that exceeds its bound
may record comparison completeness, but restoration remains unverified and no
action may execute. A complete official task failure after a successful
independent source baseline is conclusive strong evidence; a runtime exception
is still an execution error, and an action/bounds or restore mismatch is invalid
context.

For `maniskill_pickcube_v1` only, the authorized runtime comparison semantic is
`tolerance_verified_full_state_v1` with fixed maximum absolute tolerance
`1e-6`. The expected archive manifest, inventory, dtype, shape, leaf bytes, and
digest remain exact. Every baseline and corrupted restoration must independently
cover the same complete leaf and numeric-component inventory, contain only
finite values, and report its observed maximum error and compared component
count. Other adapters retain exact-digest/zero-tolerance behavior unless they
receive a separately versioned and compatibility-bound authorization.

This is the implemented fail-closed adapter path, not by itself a simulator
claim. Discovery observed complete 70-component restoration with maximum error
`1.1920929e-7`; trusted collection and strong PickCube evidence still require a
passing compatibility-bound probe and every subsequent remote gate.
