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

## Engineering and data rules

- Use Python 3.11 and a `src`-layout package.
- All public functions/classes need type annotations and concise docstrings.
- Core tests run on CPU without network access. Tests must not download models/datasets; GPU and network/SSH tests require explicit pytest markers or mocked subprocesses.
- M2A fixture evaluation, evidence validation, serialization, resume, and CLI validation are local, CPU-only, and network-free; they must not use SSH, a simulator, LangMani, ManiSkill, or a GPU.
- Keep model, data, storage, simulator, policy, remote-execution, and training interfaces decoupled. Add only dependencies needed for the current milestone.
- Do not silently repair malformed data; raise descriptive validation errors. All random operations take an explicit seed or generator, and all configs are serializable.
- Preserve complete provenance for generated/transformed samples. Split source episodes before deriving samples, and keep every source episode plus derivatives in the same split.
- A heuristic corruption is not a label. Any heuristic outcome label assigned by a later evaluator is weak evidence and is not simulator verification.
- Corruption generation does not determine task outcome. Corrupted actions remain unlabeled proposals until a later evaluator attaches evidence; never represent heuristic corruption as simulator verification.
- Evidence and outcome labels are distinct. An evaluator may return indeterminate evidence, and missing evidence must never be replaced with default task values.
- Runtime failure is not task failure. Record evaluator exceptions without fabricating success, progress, safety, or failure outcomes.
- Only complete, conclusive evidence may be projected into an `OutcomeLabel`.
- Simulator verification requires exact state restoration followed by successful replay; approximate reconstruction is not verification.
- Repeated and resumed evaluation must be idempotent, and source corruption datasets and proposals must remain immutable.
- Deterministic fixture evaluators are infrastructure tests only and must never support training, benchmark, or research claims.
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
