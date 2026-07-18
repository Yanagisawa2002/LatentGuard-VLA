# M4B Multi-View Visual Action Verifier Plan

## Milestone boundary

M4B trains and evaluates a PickCube visual action verifier from the accepted
M4A RGB packets and candidate action chunks. The deployable student input is
limited to three ordered RGB views, a `[16, 8]` action chunk, and a `[16]`
action mask. Privileged structured state is available only through a
content-bound teacher-target cache during training and is absent at student
inference.

M4B does not render images, replay a simulator, regenerate candidates or
outcomes, refit M3B structured models, use a VLM or LangMani, process video, or
use more than one GPU. Raw datasets, feature arrays, teacher arrays,
checkpoints, optimizer state, and per-sample predictions remain outside Git.

## Initial repository state

- Starting revision:
  `6e2c573d55c721158d174f8d76e7d8478a767835`.
- Starting branch: `codex/m4a-multiview-visual-dataset`.
- M4B branch: `codex/m4b-visual-action-verifier`.
- The starting working tree was clean, local `HEAD` equaled its upstream, and
  the GitHub branch resolved to the same full SHA.
- The existing remote server was reachable, its tracked checkout was clean,
  the single RTX 5090 was idle, and persistent storage retained the prior
  milestone run roots. The remote checkout will not be advanced until a clean
  local implementation revision has passed validation and been pushed.

## Bound accepted artifacts

- M4A execution SHA:
  `7a2a073666e455b36da8e72a2b87350a2baf3582`.
- M4A result SHA:
  `6e2c573d55c721158d174f8d76e7d8478a767835`.
- M3A visual development dataset:
  `sha256:f79be2b357a9de68e1deb80132f7915717158bec506884554536882e3adbd339`.
- M3C external visual dataset:
  `sha256:62e762acf6a441579c494e1aab6e873e868594cead40f256be2d809abcbeccef`.
- M3A structured dataset:
  `sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d`.
- Visual compatibility identity:
  `sha256:d4d4b18156eeb390541b0fd184f0a0e0ad67c36450f45c69e36c8c3426d03312`.
- Camera-rig digest:
  `sha256:f6345d8c01554a8def7a84f2cd01707523192338d1082578cc27652d7ee42f1f`.
- Render-domain configuration digest:
  `sha256:ce16fe22b8b143a714f10b43653a30f86d409402d756cc55b60de9fee158abc2`.
- M3B checkpoint, preprocessing, calibration, model, split, and benchmark
  identities and the M3C candidate-pool, blind-manifest, and outcome identities
  will be loaded and verified from their committed strict reports rather than
  copied from the milestone prompt.

## Implementation plan

1. Extend the existing training, checkpoint, calibration, evaluation, and
   blind-selection contracts instead of creating parallel lifecycle machinery.
2. Add strict M4B configuration and project-owned serializable models for the
   ResNet-18 contract, feature and teacher caches, input allowlist, domain
   cycling, four screening architectures, promotion policy, run identities,
   model freeze, and outcome-free external selection.
3. Add safe transactional NPY feature/teacher cache publication with exact
   inventory reload, tamper detection, path-independent identities, and
   zero-duplicate resume.
4. Reuse one shared temporal action encoder across all four visual models. Use
   the fixed torchvision ResNet-18 ImageNet-1K V1 family for pretrained models
   and the same ResNet-18 architecture with random initialization for the
   end-to-end baseline.
5. Add the eight required CLI entry points with dry-run, bounded smoke, resume,
   explicit seed/device/output, checkpoint, manifest, and peak-memory support.
6. Add CPU-only fake fixtures for cache, teacher, model, training, promotion,
   calibration, bootstrap, and blind-selection contracts. GPU and real-weight
   tests remain explicitly marked.
7. Run the complete local suite and required M4B smoke gates, review every
   staged file, commit, push, and verify local/upstream/GitHub equality before
   any remote GPU execution.

## Remote execution plan

1. Synchronize the clean remote checkout to the exact pushed implementation
   SHA and validate every accepted source artifact and structured teacher
   checkpoint identity.
2. Reuse the existing Python/CUDA environment when validation permits. Prepare
   and digest the official ResNet-18 ImageNet-1K V1 weights once.
3. Extract and strictly reload the M3A frozen feature cache and M3A teacher
   targets once, then pass live-versus-cache equivalence.
4. Screen exactly four model families at seed `0` using validation only. Freeze
   an immutable promotion record.
5. Reuse seed `0` and run seeds `1` and `2` only for the validation winner,
   strongest direct non-distilled baseline, and at most one essential ablation.
   Do not run seeds `3` or `4`.
6. Freeze checkpoints, preprocessing, calibration, thresholds, and ensemble
   semantics before opening internal test or M3C external visual artifacts.
7. Evaluate internal test once, then create the M3C feature cache, outcome-free
   per-domain selection manifests, and only afterward join accepted outcomes.
8. Verify cache/checkpoint zero-work resumes, run one final complete Linux
   suite on the accepted execution revision, retrieve only compact sanitized
   reports, validate locally, and commit/push the result closeout.

## Server lifecycle amendment

The existing paid server remains powered on throughout M4B and after M4B
completion. No command, failure, idle period, result retrieval, milestone
completion, or inherited documentation authorizes shutdown, suspension,
deallocation, termination, environment removal, artifact deletion, or creation
of an automatic shutdown task. Failures preserve resumable state and are fixed
locally before exact-SHA resynchronization. The final state must retain SSH
availability, clean tracked source, accepted environments, datasets, caches,
checkpoints, reports, and run roots for the next explicitly authorized task.

## Validation and closeout

Before each implementation or result push, run:

```text
python -m pytest
ruff check .
ruff format --check .
mypy src
git diff --check
```

Also run CPU forward/backward, fake cache and teacher round trips, four-model
tiny overfit, checkpoint save/reload/resume, seed-promotion, blind-selection,
all eight CLI help paths, sanitization, staged-scope review, and exact SHA
verification. Weak or negative model results remain valid engineering results;
test or external outcomes never authorize tuning or seed escalation.

## Current status

Repository/server discovery and the local M4B implementation are complete.
The strict configurations, eight CLI commands, metadata-isolated data join,
four models, feature/teacher caches, exact checkpoints, staged promotion,
internal evaluation, and outcome-free external join have passed targeted CPU
tests plus the complete local test/lint/type suite. Remote synchronization,
cache creation, GPU smokes, screening, promotion, evaluation, compact result
retrieval, and result closeout remain pending until the implementation commit
is reviewed, pushed, and verified at the exact SHA.
