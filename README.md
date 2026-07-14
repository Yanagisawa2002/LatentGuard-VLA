# LatentGuard-VLA

LatentGuard-VLA is a simulator-independent foundation for action-conditioned
robot-policy verification, evaluation, and failure analysis. Milestone M0
provides typed episode data, deterministic synthetic fixtures, safe local
serialization, validation, and exact-revision remote synchronization.

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

Preview an exact-revision remote synchronization without network access:

```text
latentguard remote-sync --dry-run --host example-training-host --repo-dir /example/latentguard-vla --branch codex/m0-data-contract --commit 0000000000000000000000000000000000000000
```

The example values are placeholders. Store machine-specific values in
environment variables or an ignored local file; never commit credentials or
private paths. See [the remote workflow](docs/remote_training.md),
[architecture](docs/architecture.md), and [data contract](docs/data_contract.md).
