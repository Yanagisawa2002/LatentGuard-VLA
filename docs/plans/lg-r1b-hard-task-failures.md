# LG-R1b hard-task natural failures and held-out-task SARM plan

## Objective and claim boundary

LG-R1b measures two things independently:

1. whether the frozen LG-R1 SARM-style progress model generalizes zero-shot to
   pre-registered unseen standard LIBERO tasks; and
2. whether the unchanged frozen VLA-JEPA policy naturally produces enough
   diverse failures on those tasks to support a later LG-R2 milestone.

This milestone does not train a failure head, tune an outcome classifier,
generate or rank candidates, intervene in actions, corrupt policy actions,
fine-tune VLA-JEPA/Qwen/the policy, access RoboLab, or modify LangMani. Progress
signals are descriptive and must not be called a safety detector.

## Immutable inputs

- baseline branch: `codex/lg-r1-sarm-progress`
- baseline commit: `c6ae31acaf07cfaa9147c1c8c705adcaafa3f35e`
- source policy revision:
  `735d9f692981e286ade093b5046627eda876e5d0`
- LeRobot revision: `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- frozen LG-R1 SARM: validation-selected `sarm-style-small/best.pt`
- historical development seeds: `1000..1009`, `2000..2019`
- sealed final seeds: `900000..900099`
- action execution: horizon 7, relative control, no action transformation

LG-R0 and LG-R1 results, splits, checkpoints, and artifacts remain read-only.

## Pre-registered tasks

The primary group has eight unseen tasks: six LIBERO-10 tasks and two
LIBERO-Goal tasks. It includes the four required scenes: LIBERO-10 task 2
(`KITCHEN_SCENE3`), task 9 (`KITCHEN_SCENE6`), task 8
(`KITCHEN_SCENE8`), and task 4 (`LIVING_ROOM_SCENE5`). The other tasks add
two-object basket placement, relational placement, object-target transfer, and
precision rack placement.

The secondary group has eight unseen LIBERO-90 tasks and is activated only by
Pilot C. It covers drawer placement/closure, open-then-place, relational
stacking, and toggle-plus-pan placement. It was selected by task structure,
not observed success rates.

The content-bound source is `configs/lg_r1b/task_registry.yaml`; the public
outcome-free serialization is `artifacts/lg_r1b/task_seed_registry.json`.

## Seed and task isolation

Seeds are unique across task experiments:

- primary pilot: `3000..3079`, ten task-local seeds per task;
- primary uniform expansion round 1: `4000..4079`;
- primary uniform expansion round 2: `4080..4159`;
- primary uniform expansion round 3: `4160..4239`;
- secondary pilot: `5000..5079`;
- secondary completion to 200 cumulative episodes: `6000..6039`.

The registry validator rejects historical overlap, final-seed access, reuse of
one seed by two task experiments, historical task overlap, standard-horizon
drift, an unfrozen task, and any authorization for synthetic failures.

## Pilot and conditional expansion

Run the primary pilot as exactly 8 tasks × 10 episodes. Every episode is
recorded atomically with its LeRobot dataset, policy action chunks, executed
actions, processor/checkpoint identities, and privileged supervision sidecar.

The frozen decision rule is:

- Pilot A, failures at least 10: expand every primary task uniformly. Target
  at least 30 natural failures, three failed tasks, and 300 failure windows.
- Pilot B, failures 3–9: expand every primary task uniformly to 20, 30, then
  at most 40 episodes per task. Never expand only failed tasks.
- Pilot C, failures below 3: do not repeat the primary tasks; activate the
  secondary group uniformly. If needed, add the pre-registered five episodes
  per secondary task to reach 200 cumulative standard episodes.
- If two task groups and 200 episodes still yield fewer than 10 natural
  failures, emit `STANDARD_LIBERO_FAILURE_YIELD_TOO_LOW` and stop adding
  LIBERO episodes.

An incomplete pilot or any infrastructure error selects no research branch.

## Stage and progress protocol

Each task resolves to a reviewed declarative adapter: `pick_place`,
`open_place`, `place_close`, `multi_place`, or `toggle_place`. Labels use
current task predicates, current entity relations, contact, poses, and the
episode-initial state. Step index is not a stage. Success is progress 1;
failure is not forced to zero; timeout is not a failure stage; real regression
is preserved; future terminal outcomes are never backfilled.

Structured QA reviews at least two episodes per task. For every task with
failures, it reviews at least two successes and two failures when available,
or all failures when fewer than two exist. Gates are zero critical errors,
zero deterministic mismatches, and at most 5% minor disagreement. A shortage
of two success examples for a failed task is explicit and fails the strict QA
gate rather than being silently repaired.

## Frozen SARM zero-shot evaluation

Before any adaptation, load the exact LG-R1 validation-selected checkpoint and
perform zero optimizer steps. Evaluate a deterministic balanced frame sample
from every active unseen task. Report progress MAE/Spearman, stage
accuracy/macro-F1, pairwise accuracy, stagnation and regression measures,
monotonicity violation, and per-task metrics. Preserve the result and its hash
before checking the adaptation trigger.

Compare it directly with the committed LG-R1 in-distribution test metrics.
Stage accuracy is interpreted cautiously because task-specific stage names can
change while the eight-position ontology remains fixed.

## Optional task-level adaptation

Adaptation runs only if any pre-registered zero-shot condition fires:

- progress MAE above `0.08`;
- progress Spearman below `0.80`; or
- pairwise progress accuracy below `0.70`.

Primary tasks are frozen into four adaptation-train tasks, two held-out
validation tasks, and two untouched held-out test tasks. Checkpoint selection
uses validation progress MAE only. Only `StageTransformer` and
`SubtaskTransformer` are trainable. VLA-JEPA, Qwen, the policy, processor, and
CLIP remain frozen. Zero-shot results are always reported beside adapted
results.

## Natural-failure evidence

The taxonomy includes horizon exhaustion, stagnation, stage regression, wrong
object/target, missed grasp, object drop, failed placement, failed open/close,
subgoal-order error, false progress, environment error, and unknown.
Unsupported fine-grained categories remain absent rather than inferred.
Environment/runtime errors are excluded from natural policy failures.

Failure windows use short, medium, long, and terminal views. They are matched
without replacement to successes from the same task and task split with the
same stage and bounded progress/remaining-horizon distance. This avoids the
trivial comparison of failure endings against success beginnings.

Frozen SARM signal analysis uses pre-registered `0.02` stagnation/regression
rules and reports precision, recall, false-alarm rate, and sampled-frame lead
time. VLA-JEPA diagnostics use the official frozen training-style interface
and report representation norms, variance, temporal change, current/predicted
distance, and Qwen action-token norms on matched-window anchors. These are
descriptive associations, not a failure head.

## LG-R2 gates

Full authorization requires at least 250 valid rollouts, eight tasks, two
suites, 20 natural failures, three failed tasks, 250 failure windows, 250
matched success windows, passed stage QA, complete held-out zero-shot
evaluation, zero episode/seed leakage, complete identities, and excluded
infrastructure errors.

The limited exception requires at least 10 failures, three failed tasks, 150
failure windows, at least two nontrivial failure categories, and a failure in
the untouched task-level test split, plus all common integrity gates. Failures
from one task, only horizon exhaustion, train-only failures, synthetic
windows, no held-out test failure, or failed stage QA cannot authorize LG-R2.

## Remote execution gates and ETA

Tracked source is authored, validated, committed, and pushed locally first.
The remote checkout must be clean and match the exact pushed revision. Every
session sources `/etc/network_turbo`; downloads use Aliyun mirrors by default.
The server performs only rollout, CUDA evaluation, optional SARM-head
adaptation, and frozen diagnostics. It never edits tracked source.

Measured LG-R1 throughput was about 15.6 seconds per episode. Expected remote
time:

- sync and smoke gates: 10–20 minutes;
- 80-episode pilot: 25–40 minutes;
- each 80-episode uniform round: 25–40 minutes;
- stage build/QA and dataset reload: 10–25 minutes;
- frozen CLIP cache plus zero-shot SARM: 15–35 minutes;
- optional adaptation: 10–25 minutes;
- failure windows, signal analysis, and frozen VLA-JEPA sample: 15–35 minutes.

Pilot-only ETA is about 1.5–2.5 hours. A 320-episode primary path is about
3–5 hours. The Pilot-C two-group/200-episode path is about 2.5–4 hours.
Checkpoint, video, raw dataset, and feature caches remain outside Git. The
server stays powered on after completion.

## Validation and completion

Before the first remote rollout: full pytest, Ruff check/format, strict mypy,
wheel and sdist build, registry/config smoke tests, Git diff review, commit,
push, clean-tree confirmation, and upstream equality.

After remote work: Linux/CUDA smoke, JSON parsing, artifact hashes, source and
sensitive-path validation, compact artifact retrieval, evidence documents,
full local validation, final diff review, commit, push, clean-tree/upstream
confirmation, and a final server-online check.

## Completion record

LG-R1b followed the pre-registered Pilot B path. The 80-episode pilot had 7
natural failures, so all eight primary tasks were expanded uniformly in three
additional 80-episode rounds. The completed run has 320 valid episodes, 27
natural failures and four failed tasks. The secondary task group was never
activated.

The remote run identifier is
`20260723T174522Z_lg-r1b-hard-tasks_c2d87c2_seed3000-3079`. The identifier
retains the revision at directory creation; the final execution audit binds
the completed artifacts to exact execution revision
`6965f1619fa4d9c40f4fc252bd35d2e336cec849`.

Recorded per-episode rollout time totals 6,821.24 seconds (1.895 GPU-hours):

| Phase | Episodes | Recorded time |
|---|---:|---:|
| Primary pilot | 80 | 0.470 h |
| Expansion 1 | 80 | 0.466 h |
| Expansion 2 | 80 | 0.478 h |
| Expansion 3 | 80 | 0.481 h |

The median episode time was 20.67 seconds. Stage construction, zero-shot
evaluation, adaptation and diagnostics completed within the original
3–5-hour wall-clock ETA, but their exact GPU-active duration was not persisted
as a content-bound metric; no more precise total GPU-hour claim is made.

The data-readiness gate passes, while frozen SARM held-out-task
generalization is Result C. The mechanical gate artifact preserves
`LG_R2_AUTHORIZED=true` according to the pre-registered data thresholds, but
direct LG-R2 execution remains withheld until a new milestone addresses the
task-specific progress/stage failure.

## Reproducible command sequence

The remote environment supplied the path variables below. Real hostnames,
credentials and machine-specific paths are intentionally not tracked.

```bash
source /etc/network_turbo
export PYTHONPATH=src
export LG_R1B_NETWORK_TURBO_SOURCED=1
cd "$LATENTGUARD_REMOTE_REPO"

# Registry and local/CPU protocol checks
python scripts/lg_r1b_build_task_registry.py \
  --config configs/lg_r1b/task_registry.yaml

# One-episode execution gate, then frozen pilot
python scripts/lg_r1b_collect_rollouts.py \
  --config configs/lg_r1b/pilot.yaml \
  --phase primary_pilot --max-episodes 1 --resume
python scripts/lg_r1b_collect_rollouts.py \
  --config configs/lg_r1b/pilot.yaml \
  --phase primary_pilot --resume
python scripts/lg_r1b_check_pilot_gate.py \
  --manifest "$LG_R1B_OUTPUT_ROOT/pilot_manifest.json"

# Pilot-B uniform expansion
for phase in primary_expansion_1 primary_expansion_2 primary_expansion_3; do
  python scripts/lg_r1b_collect_rollouts.py \
    --config configs/lg_r1b/expansion.yaml \
    --phase "$phase" --resume
done

# Stage labels and dataset
python scripts/lg_r1b_build_stage_labels.py \
  --config configs/lg_r1b/stages.yaml
python scripts/lg_r1b_validate_stage_labels.py \
  --report "$LG_R1B_OUTPUT_ROOT/stage_annotation_report.json"
python scripts/lg_r1b_build_progress_dataset.py \
  --runtime-root "$LG_R1B_OUTPUT_ROOT"

# Frozen zero-shot evaluation before adaptation
python scripts/lg_r1b_evaluate_frozen_sarm.py \
  --config configs/lg_r1b/eval_zero_shot.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" --dry-run
python scripts/lg_r1b_evaluate_frozen_sarm.py \
  --config configs/lg_r1b/eval_zero_shot.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT"

# Triggered lightweight adaptation: dry-run, one-step save, resume, full run
python scripts/lg_r1b_adapt_sarm.py \
  --config configs/lg_r1b/adaptation.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" --dry-run
python scripts/lg_r1b_adapt_sarm.py \
  --config configs/lg_r1b/adaptation.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" \
  --output-dir "$LG_R1B_OUTPUT_ROOT/sarm-adaptation-smoke" \
  --result "$LG_R1B_OUTPUT_ROOT/sarm-adaptation-smoke/results.json" \
  --max-steps 1 --limit-samples 64
python scripts/lg_r1b_adapt_sarm.py \
  --config configs/lg_r1b/adaptation.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" \
  --output-dir "$LG_R1B_OUTPUT_ROOT/sarm-adaptation-smoke" \
  --result "$LG_R1B_OUTPUT_ROOT/sarm-adaptation-smoke/resume-results.json" \
  --max-steps 2 --limit-samples 64 --resume
python scripts/lg_r1b_adapt_sarm.py \
  --config configs/lg_r1b/adaptation.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT"

# Natural-failure analysis and frozen VLA-JEPA diagnostic
python scripts/lg_r1b_build_failure_windows.py \
  --config configs/lg_r1b/failure_windows.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT"
python scripts/lg_r1b_analyze_failure_signals.py \
  --config configs/lg_r1b/analysis.yaml \
  --runtime-root "$LG_R1B_OUTPUT_ROOT"
python scripts/lg_r1b_analyze_vlajepa_descriptive.py \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" --max-samples 64
python scripts/lg_r1b_check_lg_r2_gate.py \
  --manifest "$LG_R1B_OUTPUT_ROOT/dataset_manifest.json" \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" \
  --output "$LG_R1B_OUTPUT_ROOT/lg_r2_gate.json"

# Final exact-revision/source/audit validation
python scripts/lg_r1b_validate_sources.py \
  --expected-commit 6965f1619fa4d9c40f4fc252bd35d2e336cec849 \
  --artifact-root "$LG_R1B_OUTPUT_ROOT" \
  --output "$LG_R1B_OUTPUT_ROOT/source_validation.json"
python -m pytest -q
python scripts/lg_r1b_write_remote_audit.py \
  --expected-commit 6965f1619fa4d9c40f4fc252bd35d2e336cec849 \
  --runtime-root "$LG_R1B_OUTPUT_ROOT" \
  --output "$LG_R1B_OUTPUT_ROOT/remote_execution_audit.json"
```

All remote sessions sourced `/etc/network_turbo`. No additional package
download was required; Ruff, mypy and package builds remained authoritative
local checks. The server remained online after the final audit.

## Validation record

- Local Windows: 1,715 passed and 3 platform-capability symlink tests skipped.
- Remote Linux: 1,718 passed.
- Ruff check: passed.
- Ruff format check: 420 files already formatted.
- mypy: no issues in 238 source files.
- sdist and wheel: built successfully.
- Task registry and all four executed phases: validated at exactly 80 scheduled
  episodes per phase.
- Local LG-R2 recomputation: semantically identical to the retrieved remote
  artifact. Byte serialization differs under Windows newline handling; the
  retrieved files themselves were verified byte-for-byte against remote
  SHA-256 values.
- Remote source validation and execution audit: passed at exact revision
  `6965f1619fa4d9c40f4fc252bd35d2e336cec849`.
- JSON parse, artifact hash, sensitive content and machine-path scans: passed.
- Remote audit confirms no final-seed access, synthetic failure generation,
  policy/failure-head/VLA-JEPA training, intervention, RoboLab execution,
  LangMani modification or tracked remote source edit.
