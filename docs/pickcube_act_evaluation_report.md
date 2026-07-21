# PickCube-v1 Native ACT Evaluation Report

## Result

WM-v0 P0 closes as **Result B**. The expert gate, 500-episode dataset gate,
formal ACT training, checkpoint integrity, deterministic resume, and independent
development evaluation all completed. No trained checkpoint is accepted for
PickCube-v1 because every evaluated checkpoint produced an out-of-contract raw
action before the simulator could execute it.

The action verifier remained fail-closed. It did not clip, normalize, repair,
or otherwise alter the learned action to obtain a rollout.

## Bound identities

- Training source commit: `ef24b14529317ee8fc240307be77fcbe94a0cf31`
- Evaluation source commit: `c490be5f77de1cbafae7bd8c686e186b02e3febf`
- Dataset digest:
  `sha256:234b82709359f1f2fde513ab26c015f4ebfb262a16b1f16206c54eddbde44800`
- Normalization digest:
  `sha256:8093f5bba3916dff218e73789739b18e05e9eb96cfc200fa18747f010959532a`
- ACT contract digest:
  `sha256:dd29f16cf295de24c9da990399dea1d036f6f2bee47a08f80a46f7d4c9ecd83f`
- Best-validation checkpoint: `step-00020000`
- Best-validation checkpoint digest:
  `sha256:b5df38aef975aad127fd5d5e65cf047b14c5eacbb019a16acdb42cf87d75e54c`

## Development evaluation

All 16 genuine training checkpoints were evaluated on the same 30 independent
development seeds, `800000..800029`. The aggregate result was:

| Checkpoints | Episodes | Successes | Action-contract violations | Simulator errors | Workspace violations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 480 | 0 | 480 | 0 | 0 |

Every checkpoint had the same termination taxonomy: 30
`action_contract_violation` outcomes. The best-validation checkpoint therefore
has success rate `0/30 = 0.0` with Wilson 95% interval
`[0.0, 0.11351339317396876]`. Its fixed-seed rerun reproduced both the raw
action digest and episode outcome.

The best-validation checkpoint allocated 232,948,224 GPU bytes during
inference loading. Query latency and action smoothness are not reported because
the first generated action chunk failed the fixed action contract before any
action was executed; reporting a fabricated execution metric would be invalid.

## Promotion and untouched final evaluation

The frozen promotion gate required zero action-contract violations, zero
simulator errors, zero workspace violations, and fixed-seed reproduction. The
checkpoint passed the latter three conditions but failed the exact action gate.
It was therefore not promoted. The reserved final seeds beginning at `900000`
remain untouched and the final episode count is zero.

The primary and secondary policy lists are both empty. No weak checkpoint is
retained as a candidate because none can legally control the environment.

## Checkpoint diversity

The early, middle, best-validation, and final roles were compared on 20 real
validation anchors. Their raw finite chunks are not all identical, but no role
is contract-valid across the anchors:

| Role | Checkpoint | Violating anchors | Maximum bound exceedance |
| --- | --- | ---: | ---: |
| early | step 1,000 | 20/20 | 0.5375752449 |
| mid | step 10,000 | 20/20 | 0.1140766144 |
| best-validation | step 20,000 | 20/20 | 0.0740650892 |
| final | step 20,000 | 20/20 | 0.0740650892 |

Best-validation and final are the same genuine step-20,000 checkpoint, so an
exact duplicate role pair exists. There are zero contract-valid roles and no
pair of legal, meaningfully distinct policy sources. P0 therefore does not
unlock policy-generated D2 counterfactual collection.

## Package and registry

No `PolicyPackage` is built for the rejected checkpoint. The strict D2 registry
is still emitted as a complete Result B audit entry with status `REJECTED`, a
null package, and zero `ACCEPTED_COMPATIBLE` policies. Validation is run with
the explicit diagnostic `--allow-blocked` option; ordinary acceptance
validation continues to fail closed.

## Concrete next repair

The next revision should declare a bounded action-output semantic in the
training and policy contracts and train it end to end, for example by mapping
network outputs differentiably into the exact per-dimension native bounds.
That semantic, its inverse target transform, processors, and hashes must be
content-bound before training. Post-hoc clipping, runtime repair, or tolerance
relaxation is not an acceptable fix. The existing final seeds must remain
untouched until a new checkpoint passes a fresh development gate.

## Remote execution

- Expert gate run:
  `20260721T094113Z_wm-v0-p0-expert-rrt-fallback_1e633f6_seed600000`
- Dataset run:
  `20260721T095704Z_wm-v0-p0-demos500_c5537a2_seed700000`
- Formal training run:
  `20260721T104200Z_wm-v0-p0-act-full_ef24b14_seed0`
- Evaluation run:
  `20260721T111400Z_wm-v0-p0-evaluation_81fe8a8_seed800000`

The remote server remains online as required.
