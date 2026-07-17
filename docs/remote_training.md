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
real adapter validates the exact archived source state, independently restores
it twice under its compatibility-bound full-state comparison contract, passes
the original-action baseline, and completes the corrupted replay. The M0 and M1
bundles alone cannot make that claim. Numeric runtime restoration is authorized
only for an adapter with an explicit versioned contract; it never weakens exact
archive content and inventory validation.

## M2C remote exact-replay smoke

The PickCube smoke is execution rather than training. It uses one compatible
GPU and a runtime environment outside the Git checkout. Before every gate, the
local tree must be clean and equal to upstream; the remote checkout must then be
clean, fast-forwarded, and equal to the requested full SHA.

Authentication is through a dedicated key with `BatchMode=yes`; connection
details remain outside Git. Password automation, `sshpass`, embedded
credentials, and committed credential files are prohibited. The M2C key-only
path and repository-scoped read-only deploy key have been verified, but no
remote acceptance is claimed until all simulator gates pass.

Gates run in order: dependency/CUDA/Vulkan smoke, compatibility probe, state
round trip, one bounded action, one official solver source, independent
baseline replay, six-source collection, M1 generation, bounded paired replay,
reload, and zero-duplicate resume. Failure stops the sequence. Contract changes
found by a probe are made locally and pushed as a new exact revision; tracked
files are never patched remotely.

Runtime state, NPY leaves, trajectories, HDF5, video, caches, and full logs stay
under `LATENTGUARD_REMOTE_RUN_ROOT`. Only reviewed sanitized manifests,
compatibility/action/collection/replay summaries, compact evidence, failure
counts, state statistics, resume results, and a final report return under
`reports/m2c/<run-id>/`.

## M3B single-GPU verifier benchmark

M3B reuses the same exact-revision synchronization contract, but performs no
simulator execution. Before starting a paid GPU, verify that the persistent M3A
full dataset is present and independently reloads with digest
`sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`.
If it is missing, rerun the accepted full M3A pipeline; never substitute the
smoke dataset or reconstruct samples manually.

Remote gates run on one RTX 5090 in this order: exact SHA sync, full dataset
validation, CPU loader, one-batch GPU forward and backward, tiny overfit for all
four models, checkpoint reload and resume, bounded 50-step runs, the fixed
four-model/five-seed benchmark, validation-only architecture selection, frozen
test evaluation, calibration and group metrics, then artifact reload. A failed
gate stops the sequence. Test data is not evaluated by smoke gates.

Smoke and full execution use distinct immutable output roots, for example
`<run-id>-smoke` and `<run-id>-full`. Execution mode participates in the
orchestration identity, so a completed smoke summary must never be overwritten
or reused as the full-benchmark root.

The benchmark identity binds the exact Git SHA, accepted-dataset and acceptance
report digests, split and preprocessing digests, benchmark/training
configuration digests, mode, and complete 20-run plan. A completed run is reused
only after its manifest, complete scalar history, summary, completion marker,
best/final/periodic checkpoint bytes, validation evaluation and predictions,
and exact directory inventory pass again. Interrupted matching runs resume from
periodic checkpoints; explicit reruns use a new root. Calibration and thresholds
bind the selected best checkpoint and its validation predictions, and those
predictions are recomputed before any test inference. The completed benchmark
also re-parses the typed selection/calibration/threshold records, recomputes
test metrics from compact predictions, and binds every evaluation artifact
digest plus the generated Markdown summary.

Runs use external immutable identities derived from content digests and write
under `LATENTGUARD_REMOTE_RUN_ROOT`. Retrieve only compact resolved configs,
manifests, histories, metrics, selection/calibration/threshold records,
coverage/group/slice tables, sanitized predictions, comparisons, and summaries
to `reports/m3b/<benchmark-id>/`. Do not retrieve datasets, checkpoints,
optimizer state, vectors, action chunks, raw simulator state, event files, or
large traces. After verified retrieval and the pushed result commit, shut down
the paid GPU server.

The remote M3B job consumes the already accepted compact M3A dataset; it does
not start ManiSkill, LangMani, a VLM, or any online simulator rollout. The fixed
models run sequentially on one GPU rather than distributed or multi-GPU
training.

## M3C single-GPU blind selection and replay

M3C begins from a clean pushed implementation revision. Persistent storage is
audited for the accepted M3A dataset and the 15 required M3B checkpoints plus
preprocessing, calibration, and threshold artifacts. Every available artifact
is digest-validated before use. If checkpoints are absent, only the three fixed
five-seed families are reconstructed from the exact M3B configurations and
accepted M3A data; architecture selection is not rerun and no M3C outcome is
read. If the accepted dataset is absent, the accepted M3A pipeline is rerun and
must reproduce its digest.

The committed compact M3B benchmark summary authorizes the outer strict-report
digest of every calibration and threshold artifact. A runtime file that was
changed and then made internally self-consistent still fails this gate. Each
seed's complete ordered validation logits and labels must also reproduce its
stored validation-prediction digest before the six temporal ensemble policies
are prepared. The accepted Stage A inventory is exactly eleven selectors backed
by exactly three five-seed bundles; the oracle is created only after Stage C.

Remote gates are sequential on one RTX 5090: exact SHA synchronization,
artifact validation/reconstruction, CPU and GPU inference smoke, six-trajectory
mechanics smoke, local configuration freeze if required, 60 untouched full
sources, candidate pool construction, blind manifest finalization, selected
union replay, complementary full-pool replay, metrics/bootstrap, and
zero-duplicate resume. No full outcome may exist before the full Stage A
manifest is final.

The CPU and GPU gates use `prepare-selection-checkpoints
--inference-smoke-output <path>` to execute the fixed outcome-free forward
fixture before source collection. A successful checkpoint load without a
forward pass is not an inference smoke.

Stage A runs with a capability allowlist, not merely a directory convention.
Its command accepts only the safely serialized blinded pool and frozen-selector
inputs—not the full pool and not any evidence, outcome, or replay path.
Complete-pool replay/evidence/outcome artifacts are first generated or loaded
by Stage B/C. Stage C is not allowed to create a
simulator session until the selected-union dataset has been strictly reloaded as
complete, strong, error-free, and bound to the same envelope, pool, source,
configuration, and seed.

Smoke and full collection use different fixed seed windows and output roots.
Full construction must load the smoke pool and prove no overlap in reset seed,
trajectory ID, split-group ID, or complete state-tree digest. If smoke changes a
configuration, the fix is made locally, committed and pushed, and the full run
uses a new untouched source set at the new exact SHA.

Production source collection bounds native ManiSkill session churn with a
sequential supervisor. It starts the same public collection CLI for exactly one
seed and one attempt in each short-lived subprocess, strictly reloads and
cross-checks the paired state/reference archives, and aggregates accepted
episodes in declared seed order. Normal source rejection is accepted only from
an exact one-attempt incomplete summary. A signal, unexpected status, missing or
malformed summary, partial archive, or identity mismatch aborts collection
before final publication. The parent then runs each requested complete
fresh-state audit in a separate short-lived subprocess and strictly rebuilds the
content-bound audit records. This is sequential failure containment on one GPU,
not multiprocessing parallelism, distributed execution, or a relaxed evidence
gate.

CPU and GPU inference/profile gates each write a compact report containing
per-candidate/group p50/p95/p99, throughput, peak allocated memory, bundle/model
loading time, and joint-versus-temporal cost differences. Raw predictions stay
outside Git and profiling cannot alter selections. Final result binding uses
only persisted phase timestamps and requires
`selection < selected start <= selected finish < remainder start <= remainder
finish`.
`evaluate-candidate-selection` requires both compact reports through
`--cpu-latency-report` and `--gpu-latency-report`; the result binds their strict
report digests and summaries.

Large checkpoints, datasets, state/action arrays, raw simulator states, and raw
predictions stay under the external run root. Only compact bundle, pool,
selection, metric, ranking, coverage, latency, bootstrap, evidence-digest,
resume, and human-review reports return to `reports/m3c/<run-id>/`. The paid
server is shut down only after those artifacts are reloaded, committed, pushed,
and local/upstream SHA equality is confirmed.

Stage A always retains its own validated latency report and atomically publishes
the manifest, selector configuration, and latency file together. External
latency copies are made afterward and can be recreated by `--resume`. Selected
and remainder replay resumes write sibling strict reports with zero evaluated,
recovered, and retried attempts; final evaluation requires both reports before
publishing `resume-summary.json`. It also emits `review.md`, whose exact bytes
are bound by the compact candidate-selection report.

M3C uses one RTX 5090 sequentially. It starts no VLM/LLM, image encoder,
LangMani job, distributed process, multi-GPU job, or new model training. The
completed full run `20260716T152012Z_m3c-full_a632a70_seed271828` found all 15
accepted M3B checkpoints and validated their digests, so no checkpoint
reconstruction occurred. It finalized Stage A before outcomes, replayed 1,168
selected and 1,712 remainder candidates, bound all 2,880 outcomes, and completed
independent selected/remainder zero-work resumes plus a final strict reload.
Compact CPU/GPU latency and outcome reports are stored under
`reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828/`; arrays, raw states,
datasets, evidence payloads, and checkpoints remain outside Git.
