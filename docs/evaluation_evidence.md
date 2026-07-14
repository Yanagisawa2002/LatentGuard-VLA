# Evaluation evidence and resume contract

## Boundary

M2A evaluates M1 action proposals without implementing or approximating a
simulator. The source corruption dataset has action proposals and provenance but
no generic restorable simulator state. Corruption type is not a task outcome,
and approximate scene reconstruction is not exact replay.

`EvaluationEvidence` records what a named evaluator established during one
deterministic attempt. It is distinct from `OutcomeLabel`. Only complete,
validated conclusive evidence can cross the explicit projection boundary.

## Statuses

- `conclusive`: enough evidence exists to derive success, canonical progress,
  and unsafe status.
- `indeterminate`: evaluation completed but did not establish a complete task
  outcome. Real partial values may be retained; missing values stay missing.
- `invalid`: the proposal or source context cannot be validly evaluated. It is
  not a failed action.
- `skipped`: an explicit applicability or policy decision prevented evaluation.
- `execution_error`: evaluator infrastructure raised or returned a runtime
  failure. It is never represented as `success=False`.

Malformed evidence is a contract error rather than an `invalid` task result.
Runtime exceptions are sanitized into execution-error records with no success,
progress, unsafe, or failure-event values.

## Identity and validation

Evidence carries proposal and source IDs, evaluator ID and semantic version,
the digest of its fully resolved configuration, seed, attempt ordinal, task
fields, diagnostics, artifact references, label metadata, replay verification,
and schema version. Collections are copied and frozen.

The identifier is SHA-256 over canonical UTF-8 JSON containing exactly:

```text
proposal_id
evaluator_id
evaluator_version
evaluator_configuration_digest
evaluation_seed
attempt_ordinal
```

Keys are sorted, separators are compact, NaN is forbidden, and Python
`hash()` is never used. Paths, hostnames, wall-clock time, process ID, output
directory, status, and task-result values do not participate.

Progress before and after are finite normalized values in `[0,1]`.
`progress_delta` is in `[-1,1]` so regress can be represented; when all three
values exist, delta must equal after minus before. Scalar metrics are finite
JSON scalars. Artifact references are unique relative POSIX paths inside the run
directory; absolute paths, traversal, backslashes, URLs, schemes, and userinfo
are rejected. In M2A they are safety-validated metadata only: the runner does
not materialize referenced files, and the serializer does not add them to the
evaluation bundle inventory. Artifact materialization and inventory validation
belong to a later adapter contract.

Verified simulator replay is legal only for simulator-sourced evidence. Strong
simulator evidence additionally requires successful exact replay. Heuristic
evidence is always weak. Skipped, invalid, and execution-error evidence has no
task label metadata; indeterminate evidence is never silently completed.

## Outcome projection

`evidence_to_outcome_label` first revalidates the complete evidence. It rejects
every non-conclusive status and every incomplete or inconsistent conclusive
record. The canonical progress mapping is:

```text
OutcomeLabel.progress = EvaluationEvidence.progress_after
```

The function copies success, unsafe status, label source and strength, replay
verification, and failure events. It does not infer probabilities, use delta as
progress, or mutate evidence.

## Evaluators and fixture limitations

Evaluators expose a stable registry ID, semantic version, resolved serializable
configuration, configuration digest, applicability decision, and evaluation
operation. The core method receives only the M1 proposal plus stable dataset,
seed, and attempt inputs. Arbitrary module imports from CLI strings are not
supported.

`deterministic_fixture` computes action statistics solely to exercise M2A
infrastructure. Its conclusive records use
`LabelSource.DETERMINISTIC_EVALUATOR`, `LabelStrength.WEAK`, and
`simulator_replay_verified=False`. Thresholds can deliberately produce
indeterminate, skipped, or controlled execution-error paths. These values are
synthetic, non-physical, and unsuitable for training, benchmarking, or research
claims.

## Ledger, resume, and retry

The runner processes proposals in M1 generation order. An attempt seed is a
canonical SHA-256 derivation from the base seed, proposal/evaluator/configuration
identity, and attempt ordinal. Before evaluator code runs, the corresponding
ledger entry is atomically persisted as `running`. Terminal evidence and ledger
state are then persisted together.

On resume, completed, indeterminate, invalid, skipped, and execution-error
attempts are preserved. An interrupted `running` attempt is repeated with the
same ordinal, seed, and evidence ID because no terminal result was durable.
Execution errors get a new ordinal and seed only with the explicit retry flag;
the previous error remains in the audit history. Status summary counts are
attempt-level, so a successful retry leaves both the earlier execution error and
the new conclusive attempt visible. Projected-label counts include only complete
conclusive evidence.

Ledger timestamps are audit metadata, not measurements used by evidence. If the
wall clock moves backward, the runner floors a new audit timestamp to the latest
durable run or attempt timestamp. This keeps recovery live and audit order
nondecreasing without changing evidence identity, seeds, status, or task values.

An already complete resume with no eligible work does not rewrite the manifest.
Changing the source corruption content digest, proposal selection, evaluator
identity/version, resolved configuration, or base seed is a run conflict and
requires a new output directory.

## Serialization and run metadata

The evaluation dataset is JSON-only and references proposal IDs rather than
copying action arrays. Initial creation uses a sibling staging directory;
incremental updates flush a sibling temporary manifest and atomically replace
the live one. Strict loading checks exact fields, versions, evidence IDs,
proposal references, ledger/evidence consistency, ordering, summaries, and safe
paths. `in_progress`, `interrupted`, and `complete` are the three run-state
values. The M2A bundle inventory contains only `manifest.json`; artifact
references remain metadata and do not imply that a referenced file exists.
Atomic replacement provides process-crash old-or-new manifest semantics; M2A
does not claim storage-device or sudden-power-loss durability on every filesystem.

The run manifest records available Git branch/SHA, evaluator details, both
source identities, seed, Python/NumPy/platform versions, a sanitized launch
command, timestamps, and final state. Operational timestamps are never identity
inputs. M2B may supply a typed LangMani exact-state adapter later; only successful
exact restoration and replay may set simulator verification.
