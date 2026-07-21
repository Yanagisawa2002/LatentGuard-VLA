# WM-v0 D2 Pilot Report

## Result

**Result B at the policy asset audit gate. No pilot was executed.**

No policy was assigned `ACCEPTED_COMPATIBLE`, so the fail-closed pipeline did not
load a controller on PickCube-v1, generate candidates, launch a simulator, or
attempt terminalization. This is the intended safe outcome, not an empty pilot
presented as evidence.

## Pilot inventory

| Measure | Result |
| --- | ---: |
| accepted compatible policies | 0 |
| policy smoke executions | 0 |
| independent episodes | 0 |
| independent anchors | 0 |
| policy-generated futures | 0 |
| candidate diversity evaluations | 0 |
| terminal success | 0 |
| terminal failure | 0 |
| unresolved horizon | 0 |
| simulator error | 0 |
| mixed-outcome anchors | 0 |

The D1 corpus remains unchanged and is not included in any D2 total. In
particular, its 120 synthetic corruptions were not renamed or counted as
policy-generated candidates.

## Gate

The gate fails accepted policy count, 100% policy-generated ratio, sample count,
anchor count, both terminal class counts, terminalized ratio, meaningful
diversity ratio, mixed-outcome anchors, future completeness, and policy metadata
completeness. Zero mismatch, leakage, and duplicate counts only mean no D2 data
was produced; they do not establish successful execution.

Therefore D2 does **not** authorize D3 expansion, formal training, or closed-loop
work.

## Reproduction commands

Run from an environment where the package is installed, or set `PYTHONPATH=src`:

```bash
python scripts/audit_policy_assets.py \
  --config configs/wm_v0_d2/policy_registry.yaml

python scripts/build_policy_registry.py \
  --config configs/wm_v0_d2/policy_registry.yaml

python scripts/validate_policy_registry.py \
  --registry artifacts/wm_v0_d2/policy_registry.json \
  --allow-blocked

python scripts/check_wm_v0_d2_gate.py \
  --manifest artifacts/wm_v0_d2/dataset_manifest.json \
  --allow-failed
```

Without the diagnostic flags, registry validation and the D2 gate return a
nonzero status, preventing accidental collection or promotion.

## Evidence

- `artifacts/wm_v0_d2/policy_registry.json`
- `artifacts/wm_v0_d2/dataset_manifest.json`
- `artifacts/wm_v0_d2/candidate_diversity_report.json`
- `artifacts/wm_v0_d2/terminalization_report.json`
- `artifacts/wm_v0_d2/data_quality_report.json`
- `docs/wm_v0_d2_policy_asset_audit.md`
- `docs/wm_v0_d2_terminalization_design.md`

The compact remote execution audit is limited to exact source revision,
environment/GPU inventory, and absence of compatible policy runtime assets. It
contains no checkpoint bytes, credentials, machine paths, raw data, or simulator
outcomes.
