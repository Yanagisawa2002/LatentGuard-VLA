# Milestone M2B-Core: Generic Exact-State Paired Replay Core

## Initial repository state

- Inspection date: 2026-07-15 (Asia/Singapore).
- The starting working tree was clean on `codex/m2a-evidence-runner`.
- Local `HEAD`, upstream, and the requested M2A baseline were all
  `b55d5ffbc3f94491b1fc3a61b3549f8a65b45a22`.
- The required M0, M1, and M2A commits were present, and remote metadata was
  fetched before implementation.
- The complete M0/M1/M2A contracts, plans, implementations, and tests were
  reviewed. The baseline suite passed with 538 tests and two Windows
  directory-symbolic-link capability skips.
- No pre-existing user changes were present to discard or overwrite.

## Branch decision

M2B-Core is developed on the required branch
`codex/m2b-generic-exact-replay`, created directly from the pushed M2A
baseline.

## Scope

- Add immutable generic replay state, task, case, restoration, execution,
  terminal-evidence, trust, paired-result, and bundle contracts.
- Bind a validated M0 episode bundle and M1 corruption bundle through both the
  existing source bundle identifier and separate canonical logical content
  digests; never trust the identifier alone.
- Resolve every proposal to its exact original action while checking complete
  episode/candidate provenance and static action contracts without mutating
  either dataset.
- Add narrow replay provider, environment-session, and adapter protocols that
  contain no simulator/framework-native objects.
- Execute an original-action baseline and corrupted action in two independent
  sessions restored from the same content-bound initial-state reference.
- Enforce the baseline-success gate and map restoration mismatch, invalid
  context, indeterminate task evidence, conclusive task failure, and runtime
  failure to their distinct M2A statuses.
- Add a trust gate that fixes the built-in fixture at deterministic weak
  non-simulator evidence while permitting only a future explicitly trusted
  exact-simulator adapter to request verified simulator metadata after every
  replay gate passes.
- Add a deterministic CPU-only numeric fixture adapter, explicit registry,
  strict configuration, replay-bundle serialization, replay reporting, and
  `latentguard replay-data`.
- Reuse the M2A evaluator identity, validation, runner, ledger, atomic
  persistence, resume, retry, and projection mechanisms rather than creating a
  second evaluation-run implementation.

## Assumptions and design decisions

- `CorruptionDataset.source_dataset_id` remains the existing path-independent
  hash of the safe M0 bundle inventory. M2B additionally computes a canonical
  logical digest over all validated M0 entities and array contents, and binds
  both values alongside the existing logical M1 corruption digest.
- Replay-case identities hash canonical action descriptors (dtype, shape,
  metadata, and content hash), both dataset digests, complete state/task
  references, adapter identity, and progress/safety semantics. Paths, hosts,
  clocks, process IDs, and output locations are excluded.
- The serialized replay bundle contains JSON case bindings and action digests,
  then reconstructs actions from separately validated M0/M1 data. It does not
  duplicate source or transformed arrays and is not written inside the strict
  M2A evaluation output directory.
- A well-formed state mismatch and a complete unsuccessful source baseline are
  invalid comparison contexts. Adapter exceptions, malformed adapter evidence,
  and close failures are infrastructure execution errors. A complete failed
  corrupted task remains conclusive task evidence.
- `replayed_control_steps` is the sum of baseline and corrupted steps that
  returned successfully. Initial progress comes only from the corrupted
  session's real initial task evidence; missing values are not inferred.
- The fixture target is deterministically constructed from the original action,
  making the baseline valid except where an explicit selector requests a
  controlled invalid context. Selectors use stable generation ordinals and are
  mutually exclusive.
- CLI dry-run constructs and validates bindings, replay cases, the replay
  bundle, adapter configuration, and M2A attempt identities, but creates no
  session and no output.

## Exclusions

M2B-Core excludes LangMani, ManiSkill, every real simulator adapter, SSH,
AutoDL, CUDA/GPU execution, remote execution, policy checkpoints, neural
networks, training, images/video, subtrajectory state replay, multiprocessing,
distributed replay, real robots, cross-source corruption, and research
performance claims.

## Planned validation

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
git diff --check
latentguard sanity-data --seed 42 --output-dir .tmp/m2b-source --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 0
latentguard corrupt-data --input-dir .tmp/m2b-source --output-dir .tmp/m2b-corrupted --config configs/corruptions/m1-smoke.json --seed 314159
latentguard replay-data --source-dir .tmp/m2b-source --corruption-dir .tmp/m2b-corrupted --output-dir .tmp/m2b-replayed --adapter deterministic_replay_fixture --config configs/replay/m2b-fixture.json --seed 161803
latentguard replay-data --source-dir .tmp/m2b-source --corruption-dir .tmp/m2b-corrupted --output-dir .tmp/m2b-replayed --adapter deterministic_replay_fixture --config configs/replay/m2b-fixture.json --seed 161803 --resume
latentguard replay-data --source-dir .tmp/m2b-source --corruption-dir .tmp/m2b-corrupted --output-dir .tmp/m2b-dry-run --adapter deterministic_replay_fixture --config configs/replay/m2b-fixture.json --seed 161803 --dry-run
```

The smoke output will be reloaded and audited for successful and failed
conclusive outcomes, controlled non-conclusive states, weak non-simulator trust,
separate baseline/corrupted sessions, idempotent resume, and an absent dry-run
destination. All temporary output will be removed before staging.

## Commit and automatic push behavior

After validation, inspect every modified and untracked file, review the staged
diff, and audit for credentials, generated data, caches, checkpoints, and large
artifacts. Commit as
`feat(m2b): add generic exact-state paired replay core` and push with
`git push -u origin HEAD` without an extra confirmation. M2B-Core is complete
only when the tree is clean and local `HEAD` equals upstream and the GitHub
remote branch SHA. Force push and history rewriting are prohibited.

## Completion record

Completed locally on 2026-07-15 (Asia/Singapore).

### Completed scope

- Added the simulator-independent replay models, protocols, canonical
  identities, complete M0/M1 source binding, safe reference-only replay bundle,
  paired executor, exact-replay M2A evaluator, explicit adapter registry,
  reporting, and `replay-data` CLI.
- Added the deterministic CPU-only fixture with strict self-validating
  configuration, independent sessions, configurable invalid/indeterminate/error
  paths, and a fixed weak non-simulator trust ceiling.
- Reused the M2A runner, ledger, atomic persistence, resume, execution-error
  retry, evidence validation, and outcome projection without a second run
  implementation.
- Added source/bundle/configuration/trust drift detection at both evaluator and
  public SDK boundaries. Runtime source mutation is recorded as an execution
  error, while a genuine baseline task failure remains invalid context.
- Added the requested repository rules, architecture/data/evidence/remote
  documentation, exact-replay guide, fixture configuration, and CPU-only test
  coverage.

### Deviations

There were no substantive scope deviations. Execution logic was placed in a
focused `executor.py` module in addition to the suggested package files. Replay
bundle serialization stores JSON references and content identities rather than
duplicating source or transformed action arrays, as allowed by the milestone.
No real adapter, framework integration, remote run, or training was added.

### Validation results

- Python 3.13.5: 697 collected, 694 passed, 3 skipped.
- Python 3.11.14: 697 collected, 694 passed, 3 skipped.
- Replay-specific suite: 156 passed, 1 skipped.
- `ruff check .`: passed.
- `ruff format --check .`: passed (66 files already formatted).
- `mypy src`: passed for 39 source files.
- `git diff --check`: passed.
- Two independent final read-only reviews reported no remaining P1/P2 issue.
- The three skips are narrowly limited to directory-symbolic-link tests because
  the current Windows account lacks link-creation privilege. The corresponding
  rejection logic and non-link serialization paths remain tested.

The final required CLI workflow produced 3 source episodes, 24 observations,
12 candidates, and 72 unlabeled proposals (12 for each of six corruption
types). Paired replay produced 70 valid baselines, 2 invalid baselines, 12
conclusive successes, 57 conclusive task failures, 1 indeterminate result, 2
invalid results, 0 execution errors, 0 skipped results, and 69 projected
outcomes. The fixture emitted no simulator-verified or non-weak claims.

The resume run reused all 72 completed attempts without rerunning them and left
the manifest SHA-256 unchanged. Dry-run planned all 72 attempts with zero
evaluations, zero sessions, and no output directory. Reload audit confirmed one
manifest file, 72 evidence entries, 72 ledger entries, complete run state, and
four sanitized path markers with no workspace path. All `.tmp/m2b-*` output was
removed before staging.

### Remaining limitations and risks

- The fixture is deliberately non-physical and supports infrastructure testing
  only; it cannot support training, benchmark, research, or simulator claims.
- No ManiSkill, LangMani, other simulator, GPU, SSH, AutoDL, or remote execution
  was used. M2C remains responsible for a real typed ManiSkill reference
  adapter.
- Replay is limited to the initial state (`state[0]`); subtrajectory replay from
  later states remains explicitly excluded.
- Path-backed bindings prioritize integrity by revalidating serialized content
  around resolution. A future large-data adapter may need an equivalent
  authenticated snapshot strategy to reduce I/O without weakening this
  guarantee.

The intended files will now be staged, committed with the required message,
pushed automatically, and verified against the upstream and GitHub branch SHA.
