# Milestone M2A: Outcome Evidence Contract and Resume-Safe Evaluation Runner

## Initial repository state

- Inspection date: 2026-07-14 (Asia/Singapore).
- The starting working tree was clean on `codex/m1-corruption-engine`.
- Local `HEAD`, upstream, and the requested M1 baseline were all
  `f7e5eb0fb01a3e2a6546ce163ed762292245b197`.
- History also contains the required M0 commit
  `65dac26c774fa347dd08133eca9449a1afe16424`.
- The M0/M1 architecture, contracts, plans, implementation, and tests were
  reviewed before M2A changes. The baseline suite passed with 297 tests and two
  Windows symbolic-link-capability skips.
- No pre-existing user changes were present to discard or overwrite.

## Branch decision

M2A is developed on the required branch `codex/m2a-evidence-runner`, created
directly from the pushed M1 baseline.

## Scope

- Add a mutation-safe evidence contract with explicit conclusive,
  indeterminate, invalid, skipped, and execution-error states.
- Validate deterministic SHA-256 evidence identity, normalized task values,
  label metadata, finite diagnostics, safe relative artifacts, and complete
  provenance without inferring missing values.
- Add an explicit, non-mutating projection from complete conclusive evidence to
  the existing M0 `OutcomeLabel` contract.
- Add a typed evaluator interface and duplicate-safe registry with no arbitrary
  dynamic imports.
- Add a deterministic CPU-only fixture evaluator for infrastructure tests. Its
  output is weak, non-simulator evidence and is not physically meaningful.
- Add an ordered single-process runner with deterministic per-attempt seeds,
  incremental crash-safe persistence, interrupted-state recovery, idempotent
  resume, explicit execution-error retry, and configuration-conflict checks.
- Add a versioned, human-readable evaluation manifest containing evidence,
  ledger, summary, run metadata, and proposal references without duplicating
  transformed action arrays.
- Add `latentguard evaluate-data`, a checked-in fixture configuration, tests,
  and documentation for the M2A boundary and later exact-state adapter work.

## Assumptions and design decisions

- The M1 corruption bundle has no restorable simulator state. M2A therefore
  cannot claim simulator replay or derive physical outcomes from corruption
  type. Exact-state replay remains an adapter responsibility for M2B.
- Evidence and labels are separate records. Only validated `conclusive`
  evidence with success, canonical progress-after, unsafe status, failure
  events, and consistent label metadata may be projected. The canonical
  `OutcomeLabel.progress` value is `progress_after`.
- Evidence identity is based only on proposal ID, evaluator ID/version,
  evaluator configuration digest, evaluation seed, and attempt ordinal.
  Operational paths, hosts, process IDs, and timestamps are excluded.
- Run identity additionally binds the source corruption bundle identity,
  evaluator/configuration, selected proposal identities, and base seed so an
  incompatible resume fails instead of silently reusing results.
- Ledger timestamps are operational audit metadata. They do not affect evidence
  identity, ordering, or summary counts. New timestamps are floored to the latest
  durable audit event if the wall clock moves backward, preserving resume
  liveness and nondecreasing audit order without altering task evidence.
- A runner-caught evaluator exception becomes `execution_error` evidence and a
  sanitized ledger error. It never supplies success, progress, unsafe, or task
  failure values. Retrying it requires the explicit retry option and creates a
  new attempt ordinal.
- Initial publication is transactional. Subsequent manifest replacements use a
  same-directory temporary file, flush, and atomic replace so the previous
  valid manifest survives a process interruption. M2A does not claim universal
  sudden-power-loss durability across filesystems that do not durably commit
  directory-entry replacement.
- M2A remains single-process, CPU-only, local, and network-free.

## Exclusions

M2A excludes LangMani and ManiSkill integration, simulator reset/replay or state
restoration, strong simulator claims, GPU work, SSH or AutoDL access, remote
execution, neural-network or verifier training, reward models, image/video
processing, multiprocessing, distributed evaluation, cross-source corruption,
and research performance claims.

## Planned validation

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
git diff --check
latentguard sanity-data --seed 42 --output-dir .tmp/m2a-source --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 2
latentguard corrupt-data --input-dir .tmp/m2a-source --output-dir .tmp/m2a-corrupted --config configs/corruptions/m1-smoke.json --seed 314159
latentguard evaluate-data --corruption-dir .tmp/m2a-corrupted --output-dir .tmp/m2a-evaluated --evaluator deterministic_fixture --config configs/evaluation/m2a-fixture.json --seed 271828
latentguard evaluate-data --corruption-dir .tmp/m2a-corrupted --output-dir .tmp/m2a-evaluated --evaluator deterministic_fixture --config configs/evaluation/m2a-fixture.json --seed 271828 --resume
latentguard evaluate-data --corruption-dir .tmp/m2a-corrupted --output-dir .tmp/m2a-dry-run --evaluator deterministic_fixture --config configs/evaluation/m2a-fixture.json --seed 271828 --dry-run
```

The smoke output will be reloaded and audited for deterministic order, status
counts, projected labels, retry counts, and idempotent resume. The dry-run
destination must remain absent. All temporary outputs will be removed before
staging.

## Commit and automatic push behavior

After validation, inspect every modified and untracked file, run the repository
and secret/generated-artifact audits, review the staged diff, commit as
`feat(m2a): add evaluation evidence and resumable runner`, and push with
`git push -u origin HEAD` without an extra confirmation. M2A is complete only
when the tree is clean and local `HEAD` equals both upstream and the remote
branch SHA. No force push or history rewrite is permitted.

## Completion record

M2A completed the scoped evidence contract, typed evaluator boundary,
deterministic fixture evaluator, projection helper, versioned manifest,
resume-safe ordered runner, registry, CLI, configuration, documentation, and
regression coverage described above. No simulator, GPU, network, SSH, AutoDL,
or remote-server work was performed.

Final validation results:

- `python -m pytest` on Python 3.13.5: 538 passed, 2 skipped.
- The same full suite on Python 3.11.14: 538 passed, 2 skipped.
- Both skips are pre-existing Windows directory-symbolic-link capability tests
  skipped because the current account lacks the required privilege; no M2A
  functional test was skipped.
- The seven M2A evaluation test modules contributed 241 passing tests to each
  full run.
- `ruff check .`: passed.
- `ruff format --check .`: passed; all 46 files were already formatted.
- `mypy src`: passed for all 27 source files.
- `git diff --check`: passed.
- An independent focused review passed 106 runner, serialization, and CLI tests
  plus a 43-case adversarial sanitization/tamper matrix without a blocker.

The final clean-directory CLI smoke produced 3 episodes, 24 observations, and
12 source candidates; 72 unlabeled corruption proposals (12 for each of the six
configured transformations); and 72 conclusive fixture evidence records with
72 explicit projections, zero execution errors, and zero retries. Every fixture
record remained `weak` with `simulator_replay_verified=false`. Resuming the
completed run evaluated zero attempts and preserved the manifest byte-for-byte
at SHA-256
`16CE82B3FFA56F4658DFE2312BF5B8300DB82DD032BBB7A3D0BF5F8E574A6EDC`.
Dry-run planned all 72 attempts, performed zero evaluations, and did not create
its output directory. The stored launch command contained redacted path
placeholders rather than machine-specific paths. Temporary smoke outputs were
removed before staging.

Known limitations are deliberate milestone boundaries: the fixture evaluator
is synthetic, single-process, CPU-only, and physically meaningless; artifact
references are validated metadata and are not materialized by M2A; there is no
exact-state simulator replay adapter; and atomic replacement protects against
process interruption but does not claim universal sudden-power-loss durability
on every filesystem.

Remote training was not performed and there are no remote run IDs. This plan is
part of the milestone commit, so it cannot embed that commit's own SHA. The
branch, full commit SHA, push result, clean-tree check, and local/upstream/remote
SHA equality are recorded in the external completion report after publication.
