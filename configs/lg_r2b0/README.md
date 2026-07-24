# LG-R2b0 execution configuration

These files are frozen before candidate and branch outcomes. Runtime outputs
must be outside the repository and selected explicitly:

```bash
export LG_R2B0_OUTPUT_ROOT=/path/to/ignored/run
export PYTHONPATH="$PWD/src:$PWD/scripts"
EXPECTED_COMMIT="$(git rev-parse HEAD)"
```

Run in order, stopping on the first failure:

```bash
python scripts/lg_r2b0_audit_candidate_source.py \
  --config configs/lg_r2b0/candidates.yaml \
  --expected-commit "$EXPECTED_COMMIT"
python scripts/lg_r2b0_validate_state_restore.py \
  --config configs/lg_r2b0/state_restore.yaml \
  --expected-commit "$EXPECTED_COMMIT"
python scripts/lg_r2b0_build_anchor_registry.py \
  --config configs/lg_r2b0/anchors.yaml \
  --expected-commit "$EXPECTED_COMMIT"
python scripts/lg_r2b0_generate_candidates.py \
  --config configs/lg_r2b0/candidates.yaml \
  --expected-commit "$EXPECTED_COMMIT" --resume
python scripts/lg_r2b0_validate_candidate_diversity.py \
  --manifest "$LG_R2B0_OUTPUT_ROOT/candidate_diversity_report.json" \
  --expected-commit "$EXPECTED_COMMIT"
python scripts/lg_r2b0_collect_counterfactuals.py \
  --config configs/lg_r2b0/collection.yaml \
  --expected-commit "$EXPECTED_COMMIT" --resume
python scripts/lg_r2b0_validate_dataset.py \
  --manifest "$LG_R2B0_OUTPUT_ROOT/dataset_manifest.json" \
  --expected-commit "$EXPECTED_COMMIT"
python scripts/lg_r2b0_analyze_identifiability.py \
  --config configs/lg_r2b0/analysis.yaml \
  --expected-commit "$EXPECTED_COMMIT"
python scripts/lg_r2b0_check_gate.py \
  --summary "$LG_R2B0_OUTPUT_ROOT/outcome_diversity_report.json" \
  --expected-commit "$EXPECTED_COMMIT"
```

`--resume` reuses only completed content-bound units. It does not change the
registered task, seed, candidate, continuation, or gate configuration.
