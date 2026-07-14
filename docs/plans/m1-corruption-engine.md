# Milestone M1: Deterministic Action Corruption Engine

## Initial repository state

- Inspection date: 2026-07-14 (Asia/Singapore).
- The starting working tree was clean on `codex/m0-data-contract`.
- Local `HEAD`, upstream, and the requested M0 baseline were all
  `65dac26c774fa347dd08133eca9449a1afe16424`.
- The private `origin` contained the pushed M0 branch and no user changes were
  present to preserve or work around.
- The complete M0 tree, required documentation, core models, validation,
  serialization, CLI, synthetic generator, and all existing tests were reviewed.
- The M0 baseline suite passed with 143 tests and one Windows
  symbolic-link-capability skip before M1 changes.

## Branch decision

M1 is developed on the required branch `codex/m1-corruption-engine`, created
directly from the pushed M0 baseline.

## Scope

- Add immutable, validated semantic action layouts with explicit field indices.
- Add an unlabeled, single-source `CorruptedActionProposal` model that cannot
  carry an `OutcomeLabel`.
- Add a typed corruption interface, duplicate-safe registry, strict JSON
  configuration parser, and six required single-source transformations.
- Add deterministic ordered generation, SHA-256 proposal identifiers,
  structured applicability skips, and strict-skip behavior.
- Add a transactional, versioned JSON/NPY corruption-dataset format that stores
  transformed actions and references source data without duplicating images.
- Add `latentguard corrupt-data`, a synthetic-compatible example configuration,
  CPU-only tests, and the required documentation and repository rules.

## Assumptions and design decisions

- M1 uses a corruption schema version independent of the M0 episode schema,
  while transformed `ActionChunk` values retain their M0 schema version.
- JSON configurations and manifests reject unknown and duplicate fields. Fully
  resolved parameters use JSON scalars and scalar sequences only.
- Stable ordering is source-episode order, source-candidate order, then
  configuration order. No Python `hash()` value participates in identifiers.
- Per-proposal random seeds and IDs are derived from canonical UTF-8 JSON and
  SHA-256, making them independent of process hash randomization.
- Gaussian noise is applicable only to floating-point action arrays. Constant
  bias on integer actions is accepted only when every result is exactly
  representable; no transformation clips, rounds, reshapes, or repairs values.
- The source dataset reference is a path-independent digest of the validated M0
  bundle contents (manifest plus arrays), not a machine-specific absolute path.
- Non-strict generation records and reports inapplicable combinations without
  changing dimensions or parameters. Strict mode fails before publishing output.

## Critical semantic boundary

Corruption produces action proposals, not task outcomes. M1 never constructs an
`OutcomeLabel`, never marks a proposal failed or unsafe, and never claims
simulator verification. Every proposal has exactly one source episode and one
source candidate. Cross-source transformations remain prohibited.

## Exclusions

M1 excludes outcome assignment, failure inference, simulator replay, image or
observation corruption, instruction changes, cross-episode/donor transforms,
neural networks, model or dataset downloads, GPU training, remote synchronization
execution, AutoDL access, and background jobs.

## Planned validation

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
latentguard sanity-data --seed 42 --output-dir .tmp/m1-source --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 2
latentguard corrupt-data --input-dir .tmp/m1-source --output-dir .tmp/m1-corrupted --config configs/corruptions/m1-smoke.json --seed 314159
git status --short
git diff --check
```

The smoke corruption dataset will be reloaded and validated, then all temporary
outputs will be removed before staging. Validation will run on Python 3.11 and a
newer compatible local interpreter when available.

## Commit and push procedure

After independent acceptance and security review, inspect every modified and
untracked file, scan for secrets and generated artifacts, stage only intended
M1 files with `git add -A`, review the staged diff, and commit as
`feat(m1): add deterministic action corruption engine`. Push automatically with
`git push -u origin HEAD`. M1 is complete only after the tree is clean and the
full local SHA equals both upstream and the remote branch SHA.

## Completion record

Completed on 2026-07-14. The delivered scope includes the explicit immutable
action layout, unlabeled proposal model, typed interface and registry, all six
required single-source transforms, strict JSON configuration, deterministic
generation and proposal identity, structured applicability skips, transactional
JSON/NPY serialization, `corrupt-data`, the synthetic-compatible configuration,
documentation, and CPU-only regression tests. Public dataset validation
recomputes every proposal identifier, restricts serialized proposals to the six
M1 built-ins with exact fully resolved parameters, and rejects input/output path
overlap before source data can be changed.

There were no scope deviations. The source reference was strengthened from a
manifest-only reference to a path-independent SHA-256 digest of the complete
validated M0 bundle contents. No outcome, failure, safety, progress, label, or
simulator-verification field was added, and no cross-source transform, remote
execution, GPU operation, SSH connection, or AutoDL access occurred.

Validation results:

- Clean M0 baseline: `143 passed, 1 skipped` before implementation.
- Python 3.11.14 with the declared NumPy 1.26.4 dependency:
  `297 passed, 2 skipped`.
- Installed local Python 3.13.5 compatibility run: `297 passed, 2 skipped`.
- `ruff check .`: passed.
- `ruff format --check .`: passed (`29 files already formatted`).
- `mypy src`: passed (`17 source files`).
- `git diff --check`: passed.
- Three independent final acceptance/security reviews: passed after identified
  identifier-integrity, numeric-precision, and path-overlap issues were fixed.

Both skipped tests exercise directory-symbolic-link rejection and were skipped
only because this Windows account lacks the privilege to create directory
symbolic links. All other safe-path, hard-link, inventory, malformed-file, and
transactional tests passed.

The required installed-CLI smoke produced:

```text
sanity-data OK: episodes=3 observations=24 candidates=12 round-trip=verified
corrupt-data OK: source_episodes=3 source_candidates=12 proposals=72 skipped_non_applicable=0 unlabeled=true round-trip=verified
corruption reload OK: proposals=72 ordinals=0..71
```

Each of the six corruption types produced 12 proposals. Reload validation
confirmed source dataset ID
`sha256:52246353c08201993d64817c01066a98f4817dd77a73cc3c0745c058f93cc550`
and first proposal ID
`cap-sha256-e64eb37118075f107d611b9e187c62d80572c5176eb2dd30385236cb298d76a4`.
Temporary smoke and virtual-environment outputs were removed before staging.

Remaining limitations are intentional M1 boundaries: proposals are unlabeled,
only one source parent and the six registered action transforms are supported,
and no evaluator or simulator establishes task outcome. The directory-symlink
test branch remains structurally covered but could not be exercised under this
Windows account's privilege policy.
