# Remote training and exact-revision synchronization

## Operating rule

The local repository is the sole source of truth. Every milestone is developed
and validated locally, committed, and pushed automatically. A remote server
only pulls and executes the exact pushed revision. Tracked files must never be
edited, committed, or pushed on the remote machine.

If a remote run reveals a bug, preserve its command, resolved configuration,
manifest, and logs; fix and validate the bug locally; commit and push the fix;
then synchronize and restart from that new exact revision. A remote-only
hotfix is not a valid workflow.

The lifecycle is:

```text
local implementation
-> local validation
-> local commit
-> automatic push
-> remote sync to exact commit
-> remote smoke test
-> remote training
-> result retrieval
```

M0 establishes synchronization only. M1, M2A, and M2B-Core are local CPU
milestones. They perform no remote training and must not start a paid GPU
instance.

M2A evaluation must not use this synchronization path at all. Its deterministic
fixture, evidence validation, serialization, resume, and CLI tests are local and
network-free. A runtime exception is evaluation infrastructure state, not task
failure or remote-run evidence.

## Configuration

Prefer an SSH host alias configured outside this repository. Supply values with
CLI flags or environment variables such as `LATENTGUARD_REMOTE_HOST` and
`LATENTGUARD_REMOTE_REPO`. `LATENTGUARD_REMOTE_BRANCH` and
`LATENTGUARD_REMOTE_COMMIT` are available when explicit flags are inconvenient;
`LATENTGUARD_REMOTE_SSH_EXECUTABLE` and
`LATENTGUARD_REMOTE_CONNECT_TIMEOUT` are optional local settings. Explicit CLI
values take precedence. Pass the branch and expected full 40-character commit
explicitly, or resolve them from the local repository in a later workflow.

Copy `.env.remote.example` only to an ignored local file. Never commit a real
host/IP, sensitive username, password, token, SSH key, or private dataset path.
The command does not read or accept passwords.

Preview a plan without opening a network connection:

```text
latentguard remote-sync --dry-run \
  --host example-training-host \
  --repo-dir /example/latentguard-vla \
  --branch codex/m0-data-contract \
  --commit 0000000000000000000000000000000000000000
```

Dry-run output is sanitized and shows the target alias, repository, branch,
expected commit, and intended non-destructive steps. It does not change local
or remote files.

## Synchronization guarantees

Before a real synchronization, the local side refuses to continue unless its
working tree is clean, the branch has an upstream, local `HEAD` equals upstream
`HEAD`, and the expected commit equals local `HEAD`. It invokes the system SSH
client with a subprocess argument list, never a shell-expanded command. SSH
runs in non-interactive batch mode with standard input disabled, and each
subprocess is bounded to ten minutes so credential prompts or stalled commands
cannot block automation indefinitely.

The remote operation performs the equivalent of:

```text
cd <remote-repository>
fail if tracked modifications exist
git fetch --prune origin
git checkout <branch>
git pull --ff-only origin <branch>
test "$(git rev-parse HEAD)" = "<expected-full-sha>"
```

Any tracked-file dirty checkout, missing repository/Python, unavailable revision,
non-fast-forward pull, or SHA mismatch fails with a nonzero result. The
operation never uses destructive cleanup, `git reset --hard`, `git clean`, a
forced checkout, history rewriting, or force push.

## Runs and artifacts

Later training milestones must write outputs outside tracked source, preferably
under `LATENTGUARD_REMOTE_RUN_ROOT`, with an ID such as
`<date>-<time>_<experiment>_<short-sha>_<seed>`. A run manifest records commit,
branch, resolved configuration, dataset version/manifest, seed, hostname, GPU,
Python/PyTorch/CUDA versions, start time, and launch command.

Checkpoints, optimizer states, raw datasets, replay buffers, video, TensorBoard
events, downloaded models, caches, and large traces stay outside Git. Small
metrics, compact tables, resolved configs, environment manifests, small plots,
and Markdown summaries may be retrieved and committed locally after review.

M2B-Core's deterministic replay fixture must not use this synchronization path.
It is not a simulator and its weak evidence is not physically meaningful. M2C
may add a real ManiSkill reference adapter, and a later LangMani adapter may use
the same generic core; any remote execution must still follow the exact-revision
lifecycle. Simulator verification is permitted only after an explicitly trusted
real adapter restores and verifies the exact source state twice, passes the
original-action baseline, and completes the corrupted replay. The M0 and M1
bundles alone cannot make that claim.
