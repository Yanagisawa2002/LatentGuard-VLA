# LatentGuard-VLA Repository Instructions

## Project goal

LatentGuard-VLA is an action-conditioned verification, evaluation, and failure-analysis framework for robot policies. Keep the core independent of LangMani, ManiSkill, LeRobot, and any specific simulator. Add integrations only through explicit, typed adapters.

Use this lifecycle: develop and validate locally; commit and push each completed milestone; make the remote training server pull that exact revision; run training/evaluation remotely; retrieve only small artifacts and summaries; then make any fixes locally and repeat the cycle.

## Source of truth and remote execution

- The local repository is authoritative for all tracked source, tests, configuration, documentation, experiment definitions, migration scripts, and evaluation logic.
- Make every tracked-file modification locally. The remote server is an execution environment only: GPU training/evaluation, simulation data generation, profiling, checkpoints, and long-running benchmarks.
- Never edit, commit, or push tracked source files on the remote server.
- If a remote run exposes a bug, preserve its logs/configuration, fix and validate it locally, commit and push the fix, synchronize the remote checkout to the new revision, and restart the run. A remote-only hotfix is invalid.

## Secrets and remote configuration

- Keep remote-specific settings in environment variables or an ignored local configuration file. The repo may include `.env.remote.example`.
- Typical variables: `LATENTGUARD_REMOTE_HOST`, `LATENTGUARD_REMOTE_REPO`, `LATENTGUARD_REMOTE_RUN_ROOT`, `LATENTGUARD_REMOTE_PYTHON`, and `LATENTGUARD_REMOTE_ENV_COMMAND`.
- Never commit real host/IP, sensitive username, passwords, keys, tokens, credentials, or private absolute dataset paths.
- Do not modify SSH configuration unless explicitly requested. Never request, print, log, or commit SSH passwords or private keys; prefer an existing SSH host alias.

## Git and milestone protocol

At the start of each milestone, inspect the branch, remote, and working tree; fetch remote metadata when available. Use the user-designated feature branch, or otherwise create `codex/<milestone>-<slug>`. Do not force-push, rewrite published history, or rebase a pushed branch without explicit instruction. Do not push directly to a protected branch unless the user has placed the repo there for this workflow.

For every completed milestone:

1. Create or update `docs/plans/` with the milestone plan, assumptions, and exclusions.
2. Implement only the requested scope; add/update tests and documentation.
3. Run required validation and milestone-specific CLI smoke tests.
4. Review `git status --short`, all modified/untracked files, `git diff --check`, and the staged diff/summary. Do not stage credentials, datasets, checkpoints, caches, or large generated outputs.
5. Stage intended changes with `git add -A`, commit with a concise milestone-oriented message, and push using `git push -u origin HEAD` without seeking an extra confirmation.
6. Confirm the tree is clean, local `HEAD` matches upstream, and record the branch plus full commit SHA.

A milestone is incomplete if its commit cannot be pushed. Preserve a local commit when pushing fails, report the exact failed command/error, and do not use destructive Git operations. Examples: `feat(m0): establish repository and data contract`, `feat(m1): add provenance-aware corruption engine`, `fix(training): restore deterministic checkpoint resume`.

## Remote synchronization and runs

Start remote training only from a pushed revision. Before a run, confirm the local tree is clean and matches upstream; record its branch and full SHA. On the remote, require a clean tracked-file checkout, fetch, check out the requested branch, fast-forward pull, and verify exact SHA:

```text
git fetch --prune origin
git checkout <branch>
git pull --ff-only origin <branch>
test "$(git rev-parse HEAD)" = "<expected-full-sha>"
```

Fail synchronization if the checkout is modified, fast-forward is impossible, the revision is unavailable or mismatched, Python is unavailable, or the target repo does not exist. Do not use `git reset --hard`, `git clean -fd`, or destructive checkout operations unless the remote checkout is explicitly disposable.

Each remote run must record commit SHA, branch, resolved config, dataset version/manifest, seed, hostname, GPU model, Python/PyTorch/CUDA versions, start time, and launch command. Write outputs outside tracked source directories, preferably under `LATENTGUARD_REMOTE_RUN_ROOT`, using an identifier such as `<date>-<time>_<experiment-name>_<short-sha>_<seed>`.

Never commit checkpoints, optimizer states, raw datasets, replay buffers, videos, TensorBoard events, downloaded models, caches, or large profiling traces. Small reviewable artifacts (e.g. `metrics.json`, compact CSVs, resolved config, environment manifest, small plots, Markdown summaries, and failure taxonomies) may be added locally and committed locally.

## Training safety

Every training entry point must support `--dry-run`, `--max-steps`, `--limit-samples`, explicit output directory and seed, checkpoint save/resume, periodic evaluation, practical graceful interruption, resolved-config and run-manifest export, and peak GPU-memory reporting.

Before a full remote run, pass these gates in order: configuration validation; CPU data-loading smoke test; one-batch GPU forward pass; one-batch forward/backward pass; short overfit or max-steps run; checkpoint save/resume test; then full training. Do not start an expensive run if an earlier gate fails. Never silently shrink model/data settings to make a run succeed; record and justify changes.

## Cost-aware model screening

Future model milestones screen each proposed architecture once with seed `0`,
full training data, validation-only selection, ordinary early stopping, and
checkpoint/resume validation. Only the best validation model, strongest direct
baseline, and one technically essential ablation may advance to final seeds
`0, 1, 2`. Add seeds `3` and `4` only when the top validation gap is below
`0.02`, three-seed standard deviation exceeds `0.03`, rankings change across
seeds, a seed collapses, or a publication-level statistical claim requires it.
Poor external-test performance must never trigger extra seeds or test tuning.
Cache reusable frozen embeddings and teacher logits once when a milestone
authorizes them; retain a raw-input end-to-end inference check. This policy
reduces redundant execution without reducing required baselines, ablations, or
external evaluation.

## Learned-model evaluation rules

- Begin training only from an independently validated accepted dataset. Bind the
  exact dataset digest and trajectory-level split assignment to every run.
- Explicitly allowlist deployable model inputs. Provenance, corruption metadata,
  candidate type, and outcome-derived fields are reporting-only unless a future
  milestone explicitly authorizes them as deployable inputs.
- Fit preprocessing, class weights, calibration, thresholds, checkpoint
  selection, and model selection without test data. Checkpoint selection uses
  validation metrics only.
- Compare every learned baseline with non-learned baselines and aggregate learned
  performance across multiple fixed seeds. Preserve and report weak, failed, or
  negative results rather than hiding them.
- Keep checkpoints and raw predictions outside Git. Only compact metrics and
  reviewed sanitized reports may be committed locally.
- M3B learned-model claims are limited to the fixed PickCube structured-state
  scope and do not imply visual, language, cross-task, general safety, or
  real-robot performance.

## Blind candidate-selection rules

- Finalize and content-bind every action-selection manifest before simulator
  outcomes are generated or loaded. A blind selection run must prove that its
  process had no outcome dataset, evidence, or replay output available.
- Keep candidate-pool generation and selection as separate phases. The original
  source action is prohibited from selectable pools unless a future deployment
  contract explicitly authorizes it. Stage A receives only opaque IDs and the
  allowlisted state, action, and mask tensors, never full candidate provenance,
  family, severity, or distribution metadata.
- Freeze selector configurations, checkpoint identities, calibration, and
  thresholds before full evaluation. All selectors compare identical candidate
  pools.
- Smoke trajectories may validate mechanics only; full-evaluation trajectories
  remain untouched. Once full outcomes exist, any configuration change requires
  a new local commit and a new untouched trajectory set.
- Report abstention as reduced coverage, never as success or task failure.
  Simulator outcomes must not trigger remote-only configuration changes.

Runtime state-restoration verification

Serialized and archived simulator state remains subject to exact structural and
content-integrity validation. Stored manifests, paths, dtypes, shapes, array
bytes, leaf inventories, and archive digests must match exactly. Archive
tampering or content drift is never tolerated.

Runtime state restoration is verified according to the state-comparison
contract explicitly bound by the adapter and its compatibility identity.

For an adapter that declares tolerance-based numeric restoration, restored state
may be treated as verified only when all of the following hold:

the archived source state passes exact digest and inventory validation;
the restored state has exactly the same complete tree structure and leaf paths;
every leaf has the expected dtype and shape;
the compared component count exactly matches the archived state;
all compared numeric values are finite;
no expected or observed component is missing, added, reordered, clipped,
normalized, or otherwise repaired;
the maximum absolute error does not exceed the fixed tolerance recorded in
both the compatibility report and the adapter’s semantic configuration;
the comparison semantic and tolerance participate in the adapter
configuration digest, replay-case identity, and evidence;
baseline and corrupted sessions independently restore and verify the same
content-bound state reference under the same comparison contract;
the observed maximum error and compared component count are recorded in replay
evidence.

For maniskill_pickcube_v1, the authorized runtime comparison semantic is
tolerance_verified_full_state_v1 with a maximum absolute tolerance of 1e-6,
subject to successful compatibility probing. The currently observed maximum
round-trip error is 1.1920929e-7 across repeated restoration and fresh
environment instances.

Passing this contract permits the restoration gate to support strong,
simulator-verified evidence, provided every other exact-replay gate also passes:
source identity validation, successful baseline replay, independent corrupted
session restoration, complete corrupted execution, and complete terminal task
evaluation.

This authorization does not apply automatically to other adapters. Other
adapters remain byte-exact unless their own versioned, compatibility-bound
comparison contract is explicitly reviewed and authorized.

Approximate scene reconstruction, partial-state comparison, omitted leaves,
schema coercion, inferred defaults, and unbound or dynamically relaxed
tolerances remain invalid and must never be described as verified simulator
restoration.

## Visual data rules

- Render visual observations only from content-bound exact states, without
  advancing physics, and reject any render whose complete post-render state or
  restored-boundary task projection changes.
- Follow every render with the adapter-bound complete-state comparison and
  exact verifier-state verification; visual validity never overrides a failed
  physical-integrity gate.
- Keep every image variant for one physical anchor in the source trajectory's
  existing split. Camera IDs and render-domain IDs are reporting metadata, not
  deployable model inputs unless a later milestone explicitly authorizes them.
- Treat M3C-derived visual observations as external evaluation-only data. Freeze
  the camera rig and rendering domains before external rendering and never tune
  them from external images or outcomes.
- Content-bind visual packets and exact image bytes while excluding runtime
  paths, hosts, process IDs, and timestamps from semantic identities. Keep raw
  RGB datasets outside Git.
- Do not train a visual model until the complete M4A dataset reload, integrity,
  split, and cross-dataset leakage gates pass.
- Report repeated-render nondeterminism exactly. Do not hide pixel drift behind
  an unreviewed tolerance.
- Visual training loaders may consume only development datasets with
  `training_allowed=true`; M3C external visual data is unavailable during
  training, selection, calibration, threshold fitting, and model promotion.
- Compute content-bound frozen image features and privileged teacher targets
  once, and reuse the identical caches for every repeated seed.
- Single-seed screening is not a stability claim. Only validation-promoted
  models may advance to seeds `0, 1, 2`; seeds `3` and `4` require explicit
  authorization.
- Camera/domain IDs, corruption metadata, candidate types, evidence/trajectory
  IDs, and outcome metadata are reporting-only and never learned inputs.
- Freeze every visual-model configuration before internal test or external
  evaluation. External evaluation must finalize an outcome-free selection
  manifest before joining the existing outcomes.
- Keep raw visual datasets, embeddings, teacher targets, and checkpoints outside
  Git.

## Receding-horizon shield rules

- M4C reuses accepted frozen M3B and M4B ensembles; it must not train, tune,
  calibrate, select, or mutate a model, checkpoint, threshold, renderer, or
  candidate configuration.
- At every boundary all selectors receive the same eight deterministic M3C
  candidates. The exact source continuation is prohibited, and candidate
  generation must remain independent of selector identity and outcomes.
- Persist the candidate pool and finalized ranking before action execution.
  Recovery restores the last committed full state and reuses that exact
  decision without rescoring. A complete episode resume must execute zero work.
- Keep horizon 16, stride 4, the last-full-window then nominal-residual tail,
  per-action task checks, and distinct success, task-failure, unsafe,
  horizon-exhausted, and execution-error outcomes.
- Visual selection uses exactly three current state-preserving RGB views in
  slots 0..2 of a fixed 128-image round-robin batch and consumes only the first
  three feature rows. Variable batches and privileged visual inputs are invalid.
- The full benchmark is exactly 60 new disjoint source plans and 600 paired
  selector/domain episodes. Use the fixed 2,000-resample source-trajectory
  bootstrap and do not create an equivalent duplicate full benchmark.

## Conservative fallback shield rules

- A reliable nominal action is accepted by default unless the complete frozen gate fires. Never describe routine nominal acceptance as an intervention.
- Candidate outcomes from the same development or evaluation set must not tune a gate. Fault identity and injected/not-injected labels are reporting-only and must never reach a learned scorer or gate.
- Bind clean and fault-injected nominal schedules to source trajectory, scenario, decision ordinal, fixed seed, and a versioned semantic before execution. Every comparison must retain the accepted fixed-primary fallback.
- Report success, unsuccessful outcomes, interventions, fixed-fallback invocations, and other-alternative invocations separately. Matching fixed-fallback performance with fewer interventions may be useful even when success does not improve.
- M4C decisions and outcomes are retrospective observed development evidence only; they are not M4D untouched evaluation and do not establish causal intervention effects.
- M4D performs no model training, threshold fitting from final outcomes, extra seeds, or candidate-pool changes.
- After a compatible remote server is restarted for M4D, keep it powered on and SSH-ready after completion unless the user explicitly authorizes shutdown in the current task.

## Engineering and data rules

- Use Python 3.11 and a `src`-layout package.
- All public functions/classes need type annotations and concise docstrings.
- Core tests run on CPU without network access. Tests must not download models/datasets; GPU and network/SSH tests require explicit pytest markers or mocked subprocesses.
- M2A fixture evaluation, evidence validation, serialization, resume, and CLI validation are local, CPU-only, and network-free; they must not use SSH, a simulator, LangMani, ManiSkill, or a GPU.
- Keep model, data, storage, simulator, policy, remote-execution, and training interfaces decoupled. Add only dependencies needed for the current milestone.
- Do not silently repair malformed data; raise descriptive validation errors. All random operations take an explicit seed or generator, and all configs are serializable.
- Preserve complete provenance for generated/transformed samples. Split source episodes before deriving samples, and keep every source episode plus derivatives in the same split.
- Group and split training examples by original source trajectory. States, anchors, corruptions, proposals, and evidence from one trajectory may not cross dataset splits.
- Action-chunk corruption may modify only its declared step window. The source continuation after that window must remain byte-identical, and its source policy identity must be recorded.
- Every state-indexed replay requires a successful remaining-trajectory baseline restored from the same archived state before any proposal from that anchor may enter training data.
- Keep intermediate-state archives outside Git. Content-bind every training example to its replay evidence, and do not create scale or class balance through manual relabeling.
- Model training is prohibited until the final training-dataset reload, evidence, split, leakage, and integrity validation gate passes.
- Keep uninterrupted-trajectory task annotations separate from post-restoration model inputs. A restoration-derived verifier vector must be captured after a complete verified fresh-session state round trip and before any action; never substitute a source-time contact flag or advance physics to manufacture one.
- A heuristic corruption is not a label. Any heuristic outcome label assigned by a later evaluator is weak evidence and is not simulator verification.
- Corruption generation does not determine task outcome. Corrupted actions remain unlabeled proposals until a later evaluator attaches evidence; never represent heuristic corruption as simulator verification.
- Evidence and outcome labels are distinct. An evaluator may return indeterminate evidence, and missing evidence must never be replaced with default task values.
- Runtime failure is not task failure. Record evaluator exceptions without fabricating success, progress, safety, or failure outcomes.
- Only complete, conclusive evidence may be projected into an `OutcomeLabel`.
- Simulator verification requires exact state restoration followed by successful replay; approximate reconstruction is not verification.
- Repeated and resumed evaluation must be idempotent, and source corruption datasets and proposals must remain immutable.
- Deterministic fixture evaluators are infrastructure tests only and must never support training, benchmark, or research claims.
- Exact-replay integrations must implement the generic replay protocols; simulator- and framework-native objects must not enter core replay models.
- Replay identities bind path-independent source and corruption content digests; runtime paths, hosts, process IDs, and timestamps must not participate.
- Original source-action replay is a mandatory validity gate. A failed baseline is invalid context, not corrupted task failure.
- Baseline and corrupted actions require independently created or independently reset sessions, both restored from the same content-bound state reference with verified state round trips.
- Replay adapter exceptions remain execution errors. Restoration mismatch and action-contract mismatch remain invalid context rather than task failure.
- Replay fixture adapters are non-physical infrastructure tests. Only an explicitly trusted real simulator adapter may emit strong, simulator-verified evidence after every exact-replay gate passes.
- Real simulator adapters must pin and verify dependency versions and record a compatibility probe before trusted replay; a package version alone is insufficient when task, controller, solver source, or public behavior can drift.
- Content-bind the source solver, task implementation, action contract, and state contract for real simulator replay. Runtime paths, package installation paths, hosts, and probe timestamps must not enter semantic identity.
- Accept a real simulator source trajectory only after an independent baseline replay from its verified initial state succeeds. Store integration runtime state archives outside Git.
- Permit exact simulator evidence only after both state restorations, baseline success, complete corrupted execution, and terminal evaluation pass for that attempt.
- Commit and push real simulator configuration changes locally before using them remotely, and sanitize runtime probe output before retrieval.
- Preserve complete single-source provenance for every corruption. Multi-source corruption is prohibited until the schema can represent every parent completely.
- A transformation must never silently clip, normalize, reshape, repair, retarget, or otherwise change its configured meaning. Applicability failures must be explicit skips or descriptive errors.
- Do not commit machine-specific absolute paths or credentials. Copy training configurations into run outputs.

Every derived sample must identify source episode, source policy, source task, transformation/corruption type and parameters, seed, split-group ID, and schema version. Evaluated derivatives must additionally record label source, strength, and simulator-replay status; unlabeled proposals must not fabricate those fields.

## Required validation and completion report

Before completing a milestone, run:

```bash
python -m pytest
ruff check .
ruff format --check .
mypy src
```

Also run milestone-specific CLI smoke tests. If a required tool is not yet configured, configure it during the milestone or document a temporary exception in the milestone plan. Treat repository warnings as failures unless narrowly justified and filtered.

The completion report must state: completed scope; changed files; validation commands/results; branch; full commit SHA; push result; known limitations; whether remote training was performed; and remote run IDs when applicable. Do not speculate beyond the current milestone or silently choose research-level architecture, loss, backbone, scale, or evaluation claims unless the milestone delegates the choice.
