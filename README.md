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

Milestone M3C adds a blind, one-shot candidate-selection protocol around the
frozen M3B verifiers. It finalizes immutable rankings before simulator outcomes
exist, excludes the original source action, and reuses the exact-state replay
ledger to evaluate selected candidates before the complete pool. See
[the blind candidate-selection contract](docs/blind_candidate_selection.md).

The current M2C checkout has passed a trusted real-runtime probe and accepted
six official ManiSkill source trajectories with six independent successful
baseline replays. The first 12-proposal paired-replay smoke was rejected because
the public Gym wrapper had not been initialized before stepping; its 12 runtime
errors were preserved without fabricating task failures. The adapter-local
`1.1.1` correction binds the archived source seed and initializes each fresh
wrapper before exact state restoration. The corrected 12-proposal rerun produced
eight conclusive successes and four conclusive task failures, with 12 valid
baselines, 12 strong simulator-verified evidence records, and zero execution
errors. Resume reused all 12 records without rerunning them. No training is
included.

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

M3B neural baselines use an optional PyTorch dependency and remain isolated from
the simulator-independent core:

```text
python -m pip install -e ".[dev,training]"
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

## Train the structured-state verifier baselines

M3B adds strict, leakage-resistant Direct Action Verifier training over the
accepted M3A dataset. The four learned baselines use only the verifier state,
candidate action chunk, and action mask; provenance and corruption metadata are
reporting-only. Inspect a bounded run without training with:

```text
latentguard train-action-verifier --help
latentguard evaluate-action-verifier --help
latentguard benchmark-action-verifier --help
```

The full protocol, model-selection boundary, metrics, calibration, and scope
limitations are documented in `docs/direct_action_verifier.md`. A frozen
selection authorizes only its exact best-checkpoint bytes and validation
prediction digest; calibration and thresholds are bound to that same identity
before the test split can be inferred. Training, checkpoints, and authoritative
predictions belong outside the repository and require the exact accepted M3A
dataset digest. M3B performs no simulator rollout, LangMani execution, or VLM
training. Completed benchmarks revalidate all run and evaluation artifacts and
emit a compact content-bound Markdown result summary.

## Evaluate blind candidate selection

M3C exposes six strict orchestration commands:

```text
latentguard prepare-selection-checkpoints --help
latentguard build-blind-candidate-pools --help
latentguard select-action-candidates --help
latentguard replay-selected-candidates --help
latentguard replay-complete-candidate-pools --help
latentguard evaluate-candidate-selection --help
```

`select-action-candidates` deliberately accepts no evidence, outcome, or replay
path. The candidate pool, selector configuration, checkpoint bundles, ensemble
validation thresholds, and full seed range are frozen before full evaluation.
Simulator execution remains a single-GPU remote operation after exact revision
synchronization; ordinary tests are CPU-only and network-free.

Stage A is isolated at the capability boundary: its process receives only a
strict blinded pool containing opaque group/candidate IDs and the deployable
state/action/mask tensors. It receives neither the full pool's corruption,
severity, distribution, or source provenance nor any evidence manifest, replay
output, or outcome-dataset capability. This is more precise than assuming the
whole machine contains no unrelated outcome file.
Complete-pool replay evidence and outcomes are first generated or loaded by
Stage B/C, after the immutable Stage A envelope exists.

The frozen Stage A inventory is exactly eleven selectors: deterministic random,
the M3B action-magnitude heuristic, three learned five-seed ensembles
(action-only, state+action MLP, and temporal), and six temporal policies (two
validation operating points plus approximately 90/80/70/50 percent coverage).
Only those three learned families form verifier bundles, for 15 checkpoints in
total. Oracle is not one of the eleven; it is analysis-only after Stage C.

Preparation validates calibration and threshold strict-report envelope digests
against the committed M3B compact benchmark summary, rather than trusting a
self-consistent rewritten runtime file. It also recomputes each ordered M3B
validation prediction digest before fitting the frozen ensemble policies.
Selector configuration, policy report, pool, bundle, and preparation-report
digests are reloaded before replay.

Preparation additionally executes a fixed outcome-free CPU/GPU forward smoke
before new source collection and can persist its digest-only proof with
`--inference-smoke-output`. Loading checkpoints without running a forward pass
does not satisfy that gate.

Stage ordering is proven with persisted timestamps, requiring selection before
the selected-union run and that completed run before the complementary run; the
bound outcome time is the actual remainder-run completion time, never a
synthesized timestamp. CPU and GPU inference runs emit compact latency reports
with per-candidate/group p50/p95/p99, throughput, peak allocated memory, bundle
loading time, and the joint-versus-temporal difference, without raw
predictions.

Stage A is atomically published only after its manifest, selector
configuration, and internal latency report all reload. Replay identity excludes
host/path/time metadata while a separate archive-audit digest still binds the
exact selected/remainder datasets. Completed resumes produce zero-work strict
reports, and final evaluation emits a byte-bound `review.md` plus a combined
`resume-summary.json`.

The one-RTX-5090 M3C run at `a632a702c709edb1fc21e702c83e30964652ff79`
completed 60 disjoint source trajectories, 360 blind groups, and all 2,880
strong simulator-verified outcomes. The frozen temporal ensemble selected a
successful candidate in 354/360 groups versus 320/360 for deterministic random;
its trajectory-bootstrap success difference was 0.09444 with a 95% interval of
[0.06389, 0.12500]. The joint MLP reached 358/360 and was not statistically
inferior to temporal, so no temporal-over-joint claim is made. Compact results
are under
[`reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828`](reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828).
M3C remains one-shot selection followed by the archived fixed continuation, not
receding-horizon control; VLM and LangMani remain outside this milestone.

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
Vulkan, or CUDA. M2C uses a separate probe requirement file whose
discovery-verified pins are `mani_skill==3.0.1`, `mplib==0.1.1`, and
`sapien==3.0.3`, and then runs:

```text
latentguard probe-maniskill-pickcube --help
latentguard collect-maniskill-pickcube --help
latentguard replay-maniskill-pickcube --help
```

The successful schema-1.1 discovery and trusted probes resolved and confirmed
the dependency, compatibility, and action contracts without guessing them.
Key-only access and six-source collection are verified. The first paired replay
correctly exposed a wrapper-initialization runtime error; the pushed `1.1.1`
correction then passed the one-proposal gate, the 12-proposal class-balance
smoke, and idempotent resume. See the
[PickCube reference integration](docs/maniskill_pickcube_reference.md).

Preview an exact-revision remote synchronization without network access:

```text
latentguard remote-sync --dry-run --host example-training-host --repo-dir /example/latentguard-vla --branch codex/m0-data-contract --commit 0000000000000000000000000000000000000000
```

## Audit, summarize, and inspect Episodes offline

The Robot Episode Toolkit adds read-only quality checks, candidate-denominator
metrics, and exact-index offline iteration without invoking a simulator:

```text
latentguard audit-data --input-dir .tmp/m0-sanity --output .tmp/audit.json
latentguard summarize-data --input-dir .tmp/m0-sanity --output .tmp/metrics.json
latentguard replay-episode --input-dir .tmp/m0-sanity --episode-id synthetic-s00000042-e0000 --candidate-id synthetic-s00000042-e0000-candidate-000 --output .tmp/replay.jsonl
```

Offline iteration is not M2B simulator replay and produces no physical or
simulator-verified evidence. See [the Robot Episode Toolkit contract](docs/robot_episode_toolkit.md).

The remote-sync example values are placeholders. Store machine-specific values
in environment variables or an ignored local file; never commit credentials
or private paths. See [the remote workflow](docs/remote_training.md),
[architecture](docs/architecture.md), and [data contract](docs/data_contract.md).
