# LG-RB0.1 patch design

## Contract

The patch implements RoboLab's documented rule: recorded configuration
restores values; the current source tree supplies structure, types, assets, and
callable code. It applies only to exact upstream commit
`0aef241fb088ca21bb4ebd24448940ed56620d17`.

## Overlay classifications

- `EXPECTED_RUNTIME_PRESERVATION`: a known live runtime/code field was
  intentionally retained. This is the only non-fatal skip class.
- `SCHEMA_DRIFT`: a missing or incompatible field is outside the explicit
  declarative-data policy. This is fatal to replay promotion.
- `INVALID_RECORDED_VALUE`: the recorded value cannot safely represent the
  live target. This is fatal.

Unknown or legacy unclassified skips are also fatal in LatentGuard.

For an existing callable target, the patch may use RoboLab's official
`string_to_callable()` only to compare an importable stable identity; the live
callable remains installed. Unimportable or different identities fail closed.
A live list or tuple containing callable code is retained as one runtime unit,
because the recorded JSON representation is lossy.

Missing keys are not accepted generically. Recorded-only fields are admitted
only under explicit declarative parameter namespaces. Missing condition/code
keys are not injected. Classification is based on live schema, namespace, and
target type rather than string appearance alone.

`/_instruction_variants` is always preserved as runtime metadata. The resolved
recorded `/instruction` remains restorable.

No `eval`, `exec`, arbitrary string execution, task-specific repair, or
callable reconstruction is used.

## State canonicalization

LatentGuard independently canonicalizes `/deformable_object` and `/gripper`
symmetrically. An allowlisted mapping must be exactly empty. Canonicalization
records before/after paths and content digests and does not change any numeric
leaf. The strict numeric tolerance remains `1e-6`; RoboLab's official
`StateValidator` remains separately reported at `0.01`; pixel tolerance is
zero.

## Ordered execution

The exact patch is tested on a clean exact-base checkout. Faithful replay must
then pass 30/30 before prefix replay is launched. Prefix replay must pass
globally before same-suffix branches are launched. Same-suffix determinism must
pass globally before A-B-A/B-A-B isolation is launched.

Branch isolation binds simulation state and replay semantics plus action
buffers, recorder state, task latches, Python/NumPy/Torch RNG state, resolved
config, and known condition caches. Failure stops the counterfactual route; it
does not authorize additional restore fields or a second patch layer.

## Gate outcome

The patch passed its bounded overlay and provenance tests and allowed 30/30
replays to complete. The faithful gate nevertheless failed because repeats one
and two drifted beyond both the strict and official state tolerances. Prefix,
same-suffix, and isolation gates were therefore not run. This is Result C, and
the design explicitly forbids extending the patch to chase the later replay
state.
