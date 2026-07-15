# LatentGuard-VLA

LatentGuard-VLA is a simulator-independent foundation for action-conditioned
robot-policy verification, evaluation, and failure analysis. Milestone M0
provides typed episode data, deterministic synthetic fixtures, safe local
serialization, validation, and exact-revision remote synchronization.
Milestone M1 adds deterministic, action-semantics-aware corruption that turns
existing action chunks into provenance-preserving, unlabeled proposals.
Milestone M2A adds a simulator-independent evidence contract, a typed evaluator
boundary, and an incrementally persisted evaluation runner that can resume
without duplicating completed attempts. Milestone M2B-Core adds a generic,
exact-state paired-replay boundary that content-binds M0 and M1 data and reuses
the M2A runner for baseline-gated original/corrupted comparisons. Milestone M2C
adds a version-locked ManiSkill `3.0.1` `PickCube-v1` reference adapter behind
those same contracts. It probes the installed simulator before trusting it,
records official Panda motion-planning sources, independently replays every
accepted source, and keeps exact runtime state archives outside Git. It does not
use LangMani or train a model.

The current M2C checkout is the locally implemented, CPU-fake-tested first
stage. No real ManiSkill trajectory or simulator evidence has been accepted
yet: the remote mplib version, observation spelling, action layout, and action
semantics remain probe-required rather than guessed.

Tracked source is developed and validated in the local repository, which is
authoritative. Remote servers only pull committed revisions and execute them;
they must never be used to edit tracked source files.

## Install locally

Python 3.11 or newer is required. From the repository root:

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On POSIX systems, activate with `source .venv/bin/activate` instead. The
remaining validation and CLI commands assume the environment is active.

## Validate locally

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
```

Generate, validate, save, reload, and compare a small deterministic dataset:

```text
latentguard sanity-data --seed 42 --output-dir .tmp/m0-sanity --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 2 --depth
```

For transactional safety, the output directory must be absent or empty.

## Generate unlabeled corruption proposals

First create an M0 source dataset, then run the checked-in synthetic-compatible
M1 corruption configuration:

```text
latentguard sanity-data --seed 42 --output-dir .tmp/m1-source --episode-count 3 --episode-length 8 --action-dim 7 --robot-state-dim 10 --camera-count 2
latentguard corrupt-data --input-dir .tmp/m1-source --output-dir .tmp/m1-corrupted --config configs/corruptions/m1-smoke.json --seed 314159
```

The configuration declares the action dimension and the indices and semantics
of each named field; the engine never guesses translation, rotation, or gripper
indices from the action dimension. `corrupt-data` writes transformed actions
and single-source provenance, reloads the output for validation, and reports
source counts, proposal counts, applicability skips, and counts by corruption
type. Every proposal remains unlabeled: corruption alone is not evidence of
failure, unsafe behavior, zero progress, or simulator verification.
Use `--strict-applicability` when any non-applicable source/transformation pair
should fail the command instead of being counted as a deterministic skip.

Both M0 dataset generation and M1 corruption run locally on CPU. M1 does not
require or connect to AutoDL or any other SSH server. See
[the corruption-engine contract](docs/corruption_engine.md) for deterministic
identifiers, applicability behavior, and the M1/M2 outcome boundary.

## Evaluate proposals with the fixture runner

The checked-in M2A evaluator exists only to test evidence, persistence, resume,
and reporting infrastructure. This workflow is local, CPU-only, network-free,
and does not use SSH, a simulator, LangMani, ManiSkill, or a GPU:

```text
latentguard evaluate-data --corruption-dir .tmp/m1-corrupted --output-dir .tmp/m2a-evaluated --evaluator deterministic_fixture --config configs/evaluation/m2a-fixture.json --seed 271828
latentguard evaluate-data --corruption-dir .tmp/m1-corrupted --output-dir .tmp/m2a-evaluated --evaluator deterministic_fixture --config configs/evaluation/m2a-fixture.json --seed 271828 --resume
```

Use `--dry-run` with an absent output path to validate and display the planned
proposal, seed, and evidence identities without evaluating anything or creating
output. `--retry-execution-errors` is the only way to append a new attempt after
an evaluator runtime error; ordinary resume preserves the error without
rerunning it.

`deterministic_fixture` produces weak synthetic evidence from action statistics.
It is not a simulator, is not physically meaningful, never sets simulator replay
verification, and must not be used for training, benchmarking, or research
claims. Only complete `conclusive` evidence can be explicitly projected to an
M0 `OutcomeLabel`; indeterminate, invalid, skipped, and execution-error records
cannot become task failures. See [the M2A evidence contract](docs/evaluation_evidence.md).

## Exercise exact-state paired replay locally

`replay-data` binds the complete M0 source and M1 corruption contents, resolves
each proposal to its original action, restores one opaque state reference in
two independent sessions, and requires the original action to succeed before
the transformed action is interpreted. The built-in adapter is a deterministic
numeric state machine for infrastructure tests only:

```text
latentguard replay-data --source-dir .tmp/m1-source --corruption-dir .tmp/m1-corrupted --output-dir .tmp/m2b-replayed --adapter deterministic_replay_fixture --config configs/replay/m2b-fixture.json --seed 161803
latentguard replay-data --source-dir .tmp/m1-source --corruption-dir .tmp/m1-corrupted --output-dir .tmp/m2b-replayed --adapter deterministic_replay_fixture --config configs/replay/m2b-fixture.json --seed 161803 --resume
```

Use `--dry-run` with a separate absent output path to validate both datasets,
resolve every selected replay case, and plan M2A attempt identities without
creating a session or output. Fixture evidence is weak,
`deterministic_evaluator` evidence: it is not a simulator, is not physically
meaningful, is unsuitable for research or training claims, and can never set
simulator verification. See [the exact-replay contract](docs/exact_replay.md).

## Probe the optional ManiSkill PickCube integration

The ordinary core install does not include or import ManiSkill, SAPIEN, mplib,
Vulkan, or CUDA. The first M2C remote stage is designed to use the separate
probe requirement file, which currently pins only `mani_skill==3.0.1`, and then
runs:

```text
latentguard probe-maniskill-pickcube --help
latentguard collect-maniskill-pickcube --help
latentguard replay-maniskill-pickcube --help
```

The first probe is intentionally required before the exact mplib package and
action layout can be checked in; those values are never guessed. Trusted source
collection and paired replay begin only after a second remote probe agrees with
the locally committed contract. Remote execution and acceptance are currently
pending the key-only authentication prerequisite described in the remote
workflow. See the
[PickCube reference integration](docs/maniskill_pickcube_reference.md).

Preview an exact-revision remote synchronization without network access:

```text
latentguard remote-sync --dry-run --host example-training-host --repo-dir /example/latentguard-vla --branch codex/m0-data-contract --commit 0000000000000000000000000000000000000000
```

The remote-sync example values are placeholders. Store machine-specific values
in environment variables or an ignored local file; never commit credentials
or private paths. See [the remote workflow](docs/remote_training.md),
[architecture](docs/architecture.md), and [data contract](docs/data_contract.md).
