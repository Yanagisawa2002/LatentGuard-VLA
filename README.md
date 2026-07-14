# LatentGuard-VLA

LatentGuard-VLA is a simulator-independent foundation for action-conditioned
robot-policy verification, evaluation, and failure analysis. Milestone M0
provides typed episode data, deterministic synthetic fixtures, safe local
serialization, validation, and exact-revision remote synchronization.
Milestone M1 adds deterministic, action-semantics-aware corruption that turns
existing action chunks into provenance-preserving, unlabeled proposals.

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

Preview an exact-revision remote synchronization without network access:

```text
latentguard remote-sync --dry-run --host example-training-host --repo-dir /example/latentguard-vla --branch codex/m0-data-contract --commit 0000000000000000000000000000000000000000
```

The remote-sync example values are placeholders. Store machine-specific values
in environment variables or an ignored local file; never commit credentials
or private paths. See [the remote workflow](docs/remote_training.md),
[architecture](docs/architecture.md), and [data contract](docs/data_contract.md).
