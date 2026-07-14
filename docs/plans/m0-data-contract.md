# Milestone M0: Repository and Data Contract

## Initial repository state

- Inspection date: 2026-07-14 (Asia/Singapore).
- The workspace initially contained only the user-provided `AGENTS.md` file.
- No Git repository, branches, remotes, or commit history existed.
- Initial Git inspection therefore had no `git status`, branch, remote, or log data to preserve.
- Git was initialized locally with the branch `codex/m0-data-contract`; after initialization, `AGENTS.md` was the only untracked file.
- No pre-existing user changes were discarded or overwritten.

## Branch decision

There was no user-designated existing branch, so M0 uses the required branch
`codex/m0-data-contract` from repository initialization onward.

## Scope

- Establish a Python 3.11-compatible, `src`-layout package and local quality tooling.
- Define typed, simulator-independent data models and descriptive validation.
- Add deterministic, small synthetic fixtures and versioned safe serialization.
- Add `latentguard sanity-data` and a network-free `latentguard remote-sync --dry-run`.
- Add mocked remote synchronization tests and CPU-only data/CLI tests.
- Document the local-authoritative, exact-revision remote execution workflow.
- Create a private remote repository, commit the completed milestone, and push it.

## Assumptions

- NumPy is the sole runtime dependency because typed numerical arrays and exact
  dtype/shape preservation are core M0 requirements.
- JSON manifests plus `.npy` array files are sufficient for the first versioned
  local format; storage optimization is deferred.
- Synthetic outcomes are deliberately varied and carry explicit provenance, but
  they are test fixtures rather than simulator evidence.
- A private repository named `LatentGuard-VLA` is the conservative default for
  the requested new remote repository.
- Python 3.11 compatibility is declared in project metadata and is validated
  with the installed Python 3.11.14 interpreter as well as Python 3.12.10.

## Exclusions

M0 excludes neural networks, pretrained models, simulator or policy-framework
integrations, corruption transforms, simulator replay, dataset/model downloads,
GPU or distributed training, training checkpoints, web interfaces, remote job
management, and any real AutoDL connection or training run.

## Planned validation

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
latentguard sanity-data --seed 42 --output-dir .tmp/m0-sanity --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 2
latentguard remote-sync --dry-run --host example-training-host --repo-dir /example/latentguard-vla --branch codex/m0-data-contract --commit 0000000000000000000000000000000000000000
git status --short
git diff --check
```

Temporary smoke-test output will be removed before staging.

## Commit and push behavior

After all required checks pass, review every modified and untracked file, scan
for secrets and generated artifacts, stage the intended M0 tree, review the
staged diff, commit as `feat(m0): establish repository and data contract`, and
push with upstream tracking. M0 is complete only if the push succeeds, the tree
is clean, and local `HEAD` exactly matches its upstream.

## Completion record

### Completed items

- Initialized Git on `codex/m0-data-contract` and created a private GitHub
  origin for `LatentGuard-VLA`.
- Added the Python 3.11 `src`-layout package, console entry point, NumPy runtime
  dependency, development tooling, strict default CPU test selection, and
  placeholder-only remote configuration example.
- Implemented immutable typed models, complete provenance, descriptive schema
  validation, deterministic synthetic episodes, and transactional JSON/NPY
  serialization with strict file inventory and unsafe-link rejection.
- Implemented and tested `sanity-data` plus sanitized, subprocess-free
  `remote-sync --dry-run`; the real sync path uses a non-shell SSH argument
  vector, local exact-revision guards, and mocked subprocess tests.
- Added the required architecture, data-contract, remote-workflow, installation,
  validation, and lifecycle documentation.

### Deviations

There are no scope deviations. Extra defensive checks were added around
transaction cleanup, malformed manifests and arrays, credential-shaped output,
bare Git repositories, symbolic links, junctions, and hard-linked arrays. These
checks do not add a simulator, network, model, or training dependency.

### Validation results

- Python 3.11.14: `python -m pytest` collected 144 tests with 143 passed and one
  platform-capability skip; `ruff check .`, `ruff format --check .`, and
  `mypy src` passed with no warnings or errors.
- Python 3.12.10: the same 144-test, lint, format, and type-check gates passed
  with the same result.
- Required `sanity-data` smoke test produced three episodes, 24 observations,
  and 12 candidates, then verified the serialization round trip.
- Required `remote-sync --dry-run` printed only the placeholder host, repository,
  branch, full SHA, and sanitized operation summary. Unit tests confirm that
  dry-run invokes no subprocess and the full suite requires no network.
- The skipped test requires directory-symbolic-link privileges unavailable to
  the current Windows identity. A native Windows junction probe independently
  confirmed that a linked bundle root is rejected; hard-link rejection is
  exercised by the normal CPU suite.
- Temporary environments and smoke-test artifacts were removed after use.

### Remaining risks and exclusions

- No real SSH connection, remote smoke test, GPU work, or remote training was
  performed, as explicitly excluded from M0. Non-dry-run SSH behavior is
  isolated behind mocked subprocess tests, so host-specific behavior remains
  for a later explicitly authorized milestone.
- Serialization supports the current schema and format versions only; migration
  and storage optimization remain future work.
- Synthetic labels are fixtures, not simulator evidence. Heuristically
  generated candidates remain weak, and all synthetic candidates are explicitly
  simulator-replay unverified.
- The completion record is included in the milestone commit; final push and
  local/upstream parity are verified after that commit is created.
