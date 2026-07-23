# LG-R1 remote execution and audit record

## Run identity

- run ID:
  `20260723T010000Z_lg-r1-rollouts_b7105e8_seed2000-2019`;
- branch: `codex/lg-r1-sarm-progress`;
- rollout code:
  `c9a34f02dea1743e357619e6e7c2b19c28ddfdd1`;
- accepted stage-label code:
  `46fc8aaad4b939ed4dcb49d47f28912f2ec78e7a`;
- final execution audit:
  `6c131b409e7c59a2cd6d8ad2459e10bc13bf76a3`;
- GPU: NVIDIA GeForce RTX 5090;
- Python/PyTorch/CUDA: 3.12.3 / 2.11.0 / 12.8;
- server state after completion: online and SSH-ready.

The run-root name retains the first implementation short SHA. It is only a
locator; each manifest records the actual commit that produced it, and the
compact execution summary binds those revisions.

## Execution protocol

Every remote session sourced `/etc/network_turbo`. Aliyun package indexes were
the default download source. The exact CLIP Hugging Face revision had no
verified Alibaba model mirror, so the official Hugging Face repository was
used with Xet disabled after the Xet endpoint returned an authorization error.
No floating or unreviewed model revision was accepted.

The remote GitHub HTTPS checkout had no usable authentication. Instead of
editing source remotely or spoofing history, each already-pushed local revision
was imported through a SHA-256-verified incremental Git bundle and checked
against its full commit ID. The tracked remote tree was clean at every
execution gate.

## Sanitized command sequence

The following records the executable sequence without host credentials or
machine-private absolute paths. `REMOTE_REPO`, `RUN_ROOT`, and `REMOTE_PYTHON`
were environment-specific locations outside tracked source.

```bash
source /etc/network_turbo
export LG_R1_NETWORK_TURBO_SOURCED=1
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export LG_R1_OUTPUT_ROOT="${RUN_ROOT}"
cd "${REMOTE_REPO}"

git status --porcelain
test "$(git rev-parse HEAD)" = "${EXPECTED_COMMIT}"

"${REMOTE_PYTHON}" scripts/lg_r1_validate_sources.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --artifact-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/source_validation.json"

"${REMOTE_PYTHON}" scripts/lg_r1_audit_sarm_stack.py \
  --config configs/lg_r1/sarm.yaml \
  --output "${RUN_ROOT}/sarm_stack_manifest.json"

"${REMOTE_PYTHON}" scripts/lg_r1_build_task_registry.py \
  --config configs/lg_r1/task_registry.yaml \
  --output "${RUN_ROOT}/task_seed_registry.json"

"${REMOTE_PYTHON}" scripts/lg_r1_collect_rollouts.py \
  --config configs/lg_r1/rollout_collection.yaml \
  --seed-registry "${RUN_ROOT}/task_seed_registry.json" \
  --output "${RUN_ROOT}/rollout_manifest.json" \
  --include-extension --record-all --resume

"${REMOTE_PYTHON}" scripts/lg_r1_build_stage_labels.py \
  --config configs/lg_r1/stage_labels.yaml \
  --rollout-manifest "${RUN_ROOT}/rollout_manifest.json" \
  --output-root "${RUN_ROOT}" \
  --report "${RUN_ROOT}/stage_annotation_report.json"

"${REMOTE_PYTHON}" scripts/lg_r1_validate_stage_labels.py \
  --manifest "${RUN_ROOT}/stage_annotation_report.json" \
  --review "${RUN_ROOT}/stage_review_decisions.json"

"${REMOTE_PYTHON}" scripts/lg_r1_build_progress_dataset.py \
  --config configs/lg_r1/dataset.yaml \
  --output-root "${RUN_ROOT}"

"${REMOTE_PYTHON}" scripts/lg_r1_eval_time_baseline.py \
  --config configs/lg_r1/eval.yaml \
  --output-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/time_baseline_results.json"

"${REMOTE_PYTHON}" scripts/lg_r1_train_representation_probe.py \
  --config configs/lg_r1/representation_probe.yaml \
  --runtime-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/runs/representation-probe" \
  --result "${RUN_ROOT}/representation_probe_results.json"

"${REMOTE_PYTHON}" scripts/lg_r1_train_or_eval_sarm.py \
  --config configs/lg_r1/sarm.yaml \
  --runtime-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/runs/sarm-style-small" \
  --result "${RUN_ROOT}/sarm_results.json"

"${REMOTE_PYTHON}" scripts/lg_r1_evaluate_progress.py \
  --config configs/lg_r1/eval.yaml \
  --runtime-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/progress_evaluation.json" \
  --max-vlajepa-samples 32

"${REMOTE_PYTHON}" scripts/lg_r1_check_lg_r2_gate.py \
  --manifest "${RUN_ROOT}/dataset_manifest.json" \
  --sarm-results "${RUN_ROOT}/sarm_results.json" \
  --output "${RUN_ROOT}/lg_r2_gate.json"

"${REMOTE_PYTHON}" scripts/lg_r1_write_remote_audit.py \
  --expected-commit "${EXPECTED_COMMIT}" \
  --runtime-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/remote_execution_audit.json"
```

Before the full head runs, the same entry points were exercised through
configuration validation, bounded smoke inputs, one-step optimization, and
checkpoint resume. The final artifacts record the selected steps and resume
status.

## Final audit

The audit passed:

- exact expected commit and clean remote checkout;
- `latentguard` imported from this repository's `src/latentguard`;
- network turbo sourced;
- all work executed remotely;
- no remote tracked-source edit;
- no final-seed access;
- no synthetic failure;
- no policy training or VLA-JEPA fine-tuning;
- no shutdown request.

The only non-pass artifact status is the intentional
`FAILURE_DATA_GATE_NOT_MET` result for LG-R2.

## Artifact retrieval

Only compact evidence was retrieved. The transfer archive matched SHA-256
`56a624194227d607d22a8f585723d782380bd2b00490be64890b015b7493d4f9`
on both hosts. Raw rollout data, videos, feature caches, model predictions,
optimizer states, and checkpoints remain outside Git.
