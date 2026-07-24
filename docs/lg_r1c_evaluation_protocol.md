# LG-R1c evaluation protocol

LG-R1c reuses the exact LG-R1b 320-episode dataset, task registry, seed
registry, task-level train/validation/test split, progress and stage labels,
27 natural failures, 7,025 failure windows, and 2,315 matched success windows.
No episode is re-split and no new or synthetic failure is generated.

Each episode contributes endpoints at 25%, 50%, 75%, and 100%. At every
endpoint, short, medium, and long windows span 8, 32, and 64 control steps.
Every common comparison uses the same eight uniformly sampled frames, camera,
instruction, endpoint, and temporal range. TOPReward's 16-frame native
diagnostic is restricted to the same long endpoints.

The failure benchmark contains all 2,315 existing matched success windows and
their referenced failure windows. Strict matching is fixed at progress
distance <= 0.025, stage distance 0, and remaining-horizon distance <= 32.
Results are reported for all matched windows, strict matches, the original
split, equal task macro, failure-task macro, FAILED_PLACEMENT, OBJECT_DROP,
stage-balanced strata, and leave-LIBERO-10-task-6-out.

Zero-shot ROBOMETER and TOPReward predictions are content-hashed before any
calibration. Only validation labels fit clipped affine progress calibration,
scalar logistic outcome calibration, and the three-scalar task-agnostic
ensemble. Task ID, stage ID, privileged state, and future terminal outcome are
not calibrator inputs. Test labels are read once after the method is fixed.

The promotion gate is pre-registered in
`configs/lg_r1c/evaluation.yaml`. Accuracy alone is never a failure metric.
Episode and window AUROC/AUPRC, natural prevalence, precision and FPR at fixed
recall, early-warning lead, false alarms, task macro, and task-6 exclusion are
all mandatory. A progress or probe improvement does not establish policy task
success, safety, or intervention efficacy.
